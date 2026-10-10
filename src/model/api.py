# model/api.py
"""
DeepFluHGT  —  model public interface
================================
Provides:
  - DeepFluHGTModel: the full model (15-step forward pipeline)
  - build_model():   build a model from HGTConfig
  - forward():       convenience forward pass
  - test():          self-test

v6.0 adaptations
---------
PE input format: [CLS]R1[SEP]R2[SEP] = 303 tokens
R2 processing: backbone splits hidden states at the SEP boundary
vocab_size: 10 (single nucleotide)
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

# ── Project path ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.contracts import HGTConfig, HGTOutput
from data_pipeline.contracts import HGTBatch

# ── modelcomponent ──
from model.impl.backbone import GlobalEncoder, LocalEncoder, DualEncoderBackbone
from model.impl.bio_features import BiologicalFeatureEncoder, EvoEncoder
from model.impl.fusion import (
    FeatureSplitBlock,
    RoleGatedCrossAttention,
    SequenceFPN,
    BatchGraphInteraction,
)
from model.impl.heads import (
    EvidentialHead,
    DirectionHead,
    RoleHead,
    BreakpointRefiner,
)
from model.impl.early_exit import EarlyExitManager


class DeepFluHGTModel(nn.Module):
    """
    The full DeepFluHGT model.

    15-step forward pipeline:
      1.  backbone(input_ids, attention_mask)
      2.  local_query = local_hidden or global_hidden
      3.  splitter(global_hidden, mask) → donor/recipient + gates
      4.  role_logit = role_head.pre_graph(global_pooled)
      5.  cross_attn(local_query, donor, recipient, role_logit, mask) → fused
      6.  fpn(fused) → fpn_output
      7.  [early_exit fusion check]
      8.  graph(fpn_pooled) → graph_pooled
      9.  bio_encoder(bio_features) → bio_encoded
      10. [early_exit graph check]
      11. cls_head(graph_pooled) → cls_logit
          [PE: extract R2 pooled → cls_head → rc_cls_logits]
      12. role_head.post_graph(graph_pooled, bio_encoded) → role_logits
      13. direction_head(donor_pooled, recipient_pooled) → direction_logits
      14. bp_refiner(fpn_output, mask, n_events) → bp_heatmaps
      15. Assemble HGTOutput

    Parameters
    ----------
    config : HGTConfig
    """

    def __init__(self, config: HGTConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim

        global_layers = config._smoke_global_layers or 12
        local_layers = config._smoke_local_layers or 16

        # ── 1. Dual-encoder backbone ──
        # In contig mode, sequences can reach max_contig_length + 2 (CLS+SEP)
        global_max_pos = max(2048, config.max_contig_length + 64)

        global_enc = GlobalEncoder(
            vocab_size=config.vocab_size,
            hidden_dim=D,
            num_layers=global_layers,
            num_heads=config.num_attention_heads,
            intermediate_dim=D * 4,
            dropout=config.dropout,
            max_position_embeddings=global_max_pos,
            pretrained_path=config.pretrained_nt_path,
            use_flash_attention=config.use_flash_attention,
        )

        local_enc = None
        if config.use_local_encoder:
            local_enc = LocalEncoder(
                vocab_size=config.vocab_size,
                hidden_dim=config.local_hidden_dim,
                num_layers=local_layers,
                intermediate_dim=config.local_hidden_dim * 4,
                dropout=config.dropout,
                max_position_embeddings=global_max_pos,
                pretrained_path=config.pretrained_caduceus_path,
                output_dim=D,
            )

        self.backbone = DualEncoderBackbone(
            global_encoder=global_enc,
            local_encoder=local_enc,
            hidden_dim=D,
            read_length=config.read_length,
            data_mode=config.data_mode,
        )

        # ── 2. Biological feature encoder ──
        self.bio_encoder = BiologicalFeatureEncoder(
            bio_feature_dim=config.bio_feature_dim,
            hidden_dim=D,
            bio_encoder_hidden=config.bio_encoder_hidden,
            dropout=config.dropout,
        )

        # ── 2b. v7: Evo2 offline-embedding encoder + decision-layer injection gate ──
        self.evo_encoder = EvoEncoder(
            evo_dim=config.pretrained_evo_dim,
            hidden_dim=D,
            evo_hidden=config.evo_hidden,
            dropout=config.dropout,
        ) if config.use_evo else None
        # Zero-initialized residual gate: initially head_pooled = graph_pooled (consistent with
        # the dual-encoder-free baseline); the injection strength of the Evo2 global prior is
        # learned adaptively during training (tanh(0)=0).
        self.fuse_alpha = nn.Parameter(torch.zeros(1)) if config.use_evo else None

        # ── 3. Disentangling split ──
        self.splitter = FeatureSplitBlock(
            hidden_dim=D,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )

        # ── 4. Role-gated cross-attention ──
        self.cross_attn = RoleGatedCrossAttention(
            hidden_dim=D,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )

        # ── 5. Sequence feature pyramid ──
        self.fpn = SequenceFPN(
            hidden_dim=D,
            dilations=config.num_fpn_dilations,
            kernel_size=3,
            dropout=config.dropout,
        )

        # ── 6. Batch graph interaction ──
        self.graph = BatchGraphInteraction(
            hidden_dim=D,
            top_k=config.graph_top_k,
            dropout=config.dropout,
        )

        # ── 7. Task heads ──
        self.cls_head = EvidentialHead(
            hidden_dim=D,
            evidential_hidden_dim=config.evidential_hidden_dim,
            num_classes=2,
            dropout=config.dropout,
            use_evidential=config.use_evidential,
        )
        self.role_head = RoleHead(hidden_dim=D)
        self.direction_head = DirectionHead(hidden_dim=D, dropout=config.dropout)
        self.bp_refiner = BreakpointRefiner(
            hidden_dim=D,
            refiner_window=config.refiner_window,
            max_peaks=3,
        )

        # ── 8. Early-exit manager ──
        self.early_exit_mgr = None
        if config.early_exit_threshold < 1.0:
            self.early_exit_mgr = EarlyExitManager(
                hidden_dim=D,
                threshold=config.early_exit_threshold,
                dropout=config.dropout,
            )

    def forward(self, batch: HGTBatch, early_exit: bool = False) -> HGTOutput:
        """
        Full forward pass: HGTBatch → HGTOutput.

        Parameters
        ----------
        batch : HGTBatch
        early_exit : bool
            Whether to enable early exit at inference time

        Returns
        -------
        HGTOutput
        """
        B = batch.input_ids.size(0)

        # ── Step 1: backbone encoding ──
        backbone_out = self.backbone(
            batch.input_ids,
            batch.attention_mask,
        )
        global_hidden = backbone_out["global_hidden"]   # [B, L, D]
        global_pooled = backbone_out["global_pooled"]   # [B, D]
        local_hidden = backbone_out["local_hidden"]     # [B, L, D] or None
        r2_global_hidden = backbone_out.get("r2_global_hidden")  # [B, L_r2, D] or None
        r2_global_pooled = backbone_out.get("r2_global_pooled")  # [B, D] or None

        # ── v7: Evo2 offline embedding → decision-layer injection (dual foundation models) ──
        # Main architecture: NTv3-pre (online global token encoder) + Evo2 7B (offline pooled global prior).
        # Evo2 is projected to hidden_dim by EvoEncoder and injected into the final decision
        # representation through the zero-initialized residual gate;
        # the token-level breakpoint path is driven entirely by NTv3-pre. Enabled only when
        # the batch carries evo_embeddings.
        evo_encoded = None
        evo_emb = getattr(batch, "evo_embeddings", None)
        if self.evo_encoder is not None and evo_emb is not None and evo_emb.numel() > 0:
            evo_encoded = self.evo_encoder(evo_emb)   # [B, D]

        # ── Step 2: cross-attention query ──
        local_query = local_hidden if local_hidden is not None else global_hidden

        # ── Step 3: disentangling split ──
        split_out = self.splitter(global_hidden, batch.attention_mask)
        donor_hidden = split_out["donor_hidden"]       # [B, L, D]
        recipient_hidden = split_out["recipient_hidden"]  # [B, L, D]
        donor_pooled = split_out["donor_pooled"]        # [B, D]
        recipient_pooled = split_out["recipient_pooled"]  # [B, D]
        gates = split_out["gates"]                      # [B, L]

        # ── Step 4: role pre-graph ──
        role_logit = self.role_head.forward_pre_graph(global_pooled)  # [B, 1]

        # ── Step 5: role-gated cross-attention ──
        cross_out = self.cross_attn(
            local_query,
            donor_hidden,
            recipient_hidden,
            role_logit,
            attention_mask=batch.attention_mask,
            local_hidden_b=r2_global_hidden,
        )
        fused_a = cross_out["fused_a"]  # [B, L, D]

        # ── Step 6: FPN ──
        fpn_output = self.fpn(fused_a)  # [B, L, D]

        # ── Step 7: fusion-layer early exit (optional) ──
        # BUG-003 fix: enable the early-exit short-circuit at inference, skipping subsequent computation.
        exit_logits_fusion = None
        _early_exit_mask = None   # [B] bool: samples that already early-exited at the fusion layer
        _early_exit_cls = None    # [B, 1]: cls_logits of early-exited samples

        if self.early_exit_mgr is not None:
            fpn_pooled = fpn_output[:, 0, :]
            exit_logits_fusion, should_exit_f, _ = self.early_exit_mgr.forward_fusion(
                fpn_pooled, early_exit=early_exit,
            )
            if early_exit and should_exit_f.any():
                exit_dict = self.early_exit_mgr.build_early_exit_output(
                    exit_logits_fusion, exit_layer=7,
                    seq_len=fpn_output.size(1),
                )
                if should_exit_f.all():
                    # All samples early-exit → return directly, skip Steps 8-15
                    return HGTOutput(
                        cls_logits=exit_dict["cls_logits"],
                        bp_heatmaps=exit_dict["bp_heatmaps"],
                        role_logits=exit_dict["role_logits"],
                        direction_logits=exit_dict["direction_logits"],
                        bp_refined=exit_dict["bp_refined"],
                        split_gates=gates,
                        donor_pool=donor_pooled,
                        recipient_pool=recipient_pooled,
                        hidden_states=fpn_output,
                        early_exit_fusion=exit_logits_fusion,
                        early_exit_graph=None,
                        exited_at=7,
                    )
                # Partial early exit: record the mask, continue executing Steps 8-15 normally,
                # and at Step 15 override the cls_logits of samples that already early-exited
                _early_exit_mask = should_exit_f
                _early_exit_cls = exit_dict["cls_logits"]

        # ── Step 8: graph interaction ──
        fpn_pooled = fpn_output[:, 0, :]  # [B, D] CLS
        graph_pooled = self.graph(fpn_pooled)  # [B, D]

        # ── Step 9: biological feature encoding ──
        bio_encoded = self.bio_encoder(batch.bio_features)  # [B, D]

        # ── Step 10: graph-layer early exit (optional) ──
        exit_logits_graph = None
        if self.early_exit_mgr is not None:
            exit_logits_graph, _, _ = self.early_exit_mgr.forward_graph(
                graph_pooled, early_exit=early_exit,
            )

        # ── Step 11: classification head ──
        # v7: the Evo2 global prior is injected into the decision representation
        # through the zero-initialized residual gate
        head_pooled = graph_pooled
        if evo_encoded is not None:
            head_pooled = graph_pooled + torch.tanh(self.fuse_alpha) * evo_encoded
        cls_out = self.cls_head(head_pooled)
        cls_logit = cls_out["cls_logit"]  # [B, 1]

        # ── Step 11b: PE RC consistency ──
        rc_cls_logits = None
        if r2_global_pooled is not None:
            rc_out = self.cls_head(r2_global_pooled)
            rc_cls_logits = rc_out["cls_logit"]  # [B, 1]

        # ── Step 12: role post-graph ──
        role_logits = self.role_head.forward_post_graph(head_pooled, bio_encoded)

        # ── Step 13: direction prediction ──
        direction_logits = self.direction_head(donor_pooled, recipient_pooled)

        # ── Step 14: breakpoint refinement ──
        bp_out = self.bp_refiner(fpn_output, batch.attention_mask, batch.n_events)
        bp_heatmaps = bp_out["bp_heatmaps"]    # [B, L]
        bp_refined = bp_out["bp_refined"]      # [B, 2W]

        # ── Step 15: assemble output ──
        # BUG-003 fix: override the cls_logits of partially early-exited samples
        if _early_exit_mask is not None and _early_exit_cls is not None:
            cls_logit = cls_logit.clone()
            cls_logit[_early_exit_mask] = _early_exit_cls[_early_exit_mask]

        return HGTOutput(
            cls_logits=cls_logit,
            bp_heatmaps=bp_heatmaps,
            role_logits=role_logits,
            direction_logits=direction_logits,
            bp_refined=bp_refined,
            split_gates=gates,
            donor_pool=donor_pooled,
            recipient_pool=recipient_pooled,
            hidden_states=fpn_output,
            rc_cls_logits=rc_cls_logits,
            early_exit_fusion=exit_logits_fusion,
            early_exit_graph=exit_logits_graph,
            exited_at=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════ #

def build_model(config: HGTConfig) -> DeepFluHGTModel:
    """Build the full model from a config."""
    model = DeepFluHGTModel(config)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[build_model] params: {n_params:,} total, {n_trainable:,} trainable")
    return model


def forward(batch: HGTBatch, config: Optional[HGTConfig] = None) -> HGTOutput:
    """
    Convenience forward pass (for testing/debugging only).

    ⚠ Every call rebuilds the model (build_model); in production use model(batch) directly.

    Parameters
    ----------
    batch : HGTBatch
    config : HGTConfig, optional (uses default config)

    Returns
    -------
    HGTOutput
    """
    if config is None:
        config = HGTConfig()
    model = build_model(config)
    model.eval()
    with torch.no_grad():
        return model(batch)


def test() -> bool:
    """
    Model self-test.

    Uses a smoke config (few layers) to validate:
      1. PE mode forward pass shape
      2. Contig mode forward pass shape
      3. Gradient propagation
    """
    print("=" * 60)
    print("DeepFluHGT  —  Model Self-Test")
    print("=" * 60)

    results = []
    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"

    def log(name, passed, detail=""):
        sym = PASS if passed else FAIL
        msg = f"  {sym} {name}"
        if detail:
            msg += f"  [{detail}]"
        print(msg)
        results.append(passed)

    # ── Smoke config (few layers for speed) ──
    pe_config = HGTConfig(
        data_mode="paired_end",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        early_exit_threshold=0.95,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )

    contig_config = HGTConfig(
        data_mode="contig",
        _smoke_global_layers=2,
        _smoke_local_layers=2,
        early_exit_threshold=0.95,
        pretrained_nt_path="",
        pretrained_caduceus_path="",
    )

    # ── 1. PE mode shape test ──
    print("\n[1] PE Mode  —  Forward Shape Test")
    model_pe = build_model(pe_config)
    model_pe.eval()

    B = 4
    pe_batch = HGTBatch(
        input_ids=torch.randint(0, 10, (B, 303)),
        attention_mask=torch.ones(B, 303, dtype=torch.long),
        bio_features=torch.randn(B, 260),
        labels=torch.tensor([1, 0, 1, 0]),
        breakpoint_targets=torch.zeros(B, 303),
        region_targets=torch.zeros(B, 303),
        pe_case=torch.zeros(B, dtype=torch.long),
        hgt_mode=torch.zeros(B, dtype=torch.long),
        n_events=torch.tensor([1, 0, 2, 0]),
    )

    with torch.no_grad():
        out_pe = model_pe(pe_batch)

    log("PE: cls_logits shape", out_pe.cls_logits.shape == (B, 1),
        f"got {list(out_pe.cls_logits.shape)}")
    log("PE: bp_heatmaps shape", out_pe.bp_heatmaps.shape == (B, 303),
        f"got {list(out_pe.bp_heatmaps.shape)}")
    log("PE: role_logits shape", out_pe.role_logits.shape == (B, 1))
    log("PE: direction_logits shape", out_pe.direction_logits.shape == (B, 1))
    log("PE: bp_refined shape", out_pe.bp_refined.shape == (B, 100),
        f"got {list(out_pe.bp_refined.shape)}")
    log("PE: split_gates shape", out_pe.split_gates.shape == (B, 303))
    log("PE: donor_pool shape", out_pe.donor_pool.shape == (B, 768))
    log("PE: recipient_pool shape", out_pe.recipient_pool.shape == (B, 768))
    log("PE: hidden_states shape", out_pe.hidden_states.shape == (B, 303, 768))
    log("PE: rc_cls_logits shape", out_pe.rc_cls_logits.shape == (B, 1))
    log("PE: early_exit_fusion present", out_pe.early_exit_fusion is not None)
    log("PE: early_exit_graph present", out_pe.early_exit_graph is not None)

    # ── 2. Contig mode shape test ──
    print("\n[2] Contig Mode  —  Forward Shape Test")
    model_ctg = build_model(contig_config)
    model_ctg.eval()

    L_ctg = 402  # 400bp + CLS + SEP
    ctg_batch = HGTBatch(
        input_ids=torch.randint(0, 10, (B, L_ctg)),
        attention_mask=torch.ones(B, L_ctg, dtype=torch.long),
        bio_features=torch.randn(B, 260),
        labels=torch.tensor([1, 0, 1, 0]),
        breakpoint_targets=torch.zeros(B, L_ctg),
        region_targets=torch.zeros(B, L_ctg),
        pe_case=torch.zeros(B, dtype=torch.long),
        hgt_mode=torch.zeros(B, dtype=torch.long),
        n_events=torch.tensor([1, 0, 1, 0]),
    )

    with torch.no_grad():
        out_ctg = model_ctg(ctg_batch)

    log("Contig: cls_logits shape", out_ctg.cls_logits.shape == (B, 1),
        f"got {list(out_ctg.cls_logits.shape)}")
    log("Contig: bp_heatmaps shape", out_ctg.bp_heatmaps.shape == (B, L_ctg),
        f"got {list(out_ctg.bp_heatmaps.shape)}")
    log("Contig: rc_cls_logits is None", out_ctg.rc_cls_logits is None)

    # ── 3. Gradient propagation test ──
    print("\n[3] Gradient Propagation Test")
    model_pe.train()
    pe_batch_train = HGTBatch(
        input_ids=torch.randint(0, 10, (2, 303)),
        attention_mask=torch.ones(2, 303, dtype=torch.long),
        bio_features=torch.randn(2, 260),
        labels=torch.tensor([1, 0]),
        breakpoint_targets=torch.zeros(2, 303),
        region_targets=torch.zeros(2, 303),
        pe_case=torch.zeros(2, dtype=torch.long),
        hgt_mode=torch.zeros(2, dtype=torch.long),
        n_events=torch.tensor([1, 0]),
    )

    out_train = model_pe(pe_batch_train)
    loss = (
        out_train.cls_logits.sum()
        + out_train.role_logits.sum()
        + out_train.direction_logits.sum()
        + out_train.bp_heatmaps.sum()
    )
    loss.backward()

    # Conditionally skip parameters (only involved in computation on specific paths)
    _skip = (
        "cross_attn.pair_attn_",
        "cross_attn.pair_norm_",
        "cross_attn.local_b_to_",
        "cross_attn.norm_b",
        "bp_refiner.conv_stack",
        "bp_refiner.fine_mlp",
        "cls_head.mlp",
        "early_exit_mgr",
        # v7: the Evo2 branch is injected through the zero-initialized residual gate;
        # initially this branch's gradient is 0 (unlocked once fuse_alpha is learned),
        # so the gradient check skips it conditionally.
        "evo_encoder",
        "fuse_alpha",
    )
    has_grad = all(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for name, p in model_pe.named_parameters()
        if p.requires_grad and not any(name.startswith(s) for s in _skip)
    )
    log("Core parameters have gradients", has_grad)

    no_nan = all(
        not torch.isnan(p.grad).any()
        for p in model_pe.parameters()
        if p.requires_grad and p.grad is not None
    )
    log("No NaN in gradients", no_nan)

    # ── Summary ──
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"TOTAL: {total}  |  PASSED: {passed}  |  FAILED: {failed}")
    print("=" * 60)

    return failed == 0
