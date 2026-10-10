# model/impl/backbone.py
"""
DeepFluHGT  —  dual-encoder backbone network
====================================
v5.0 → 0 adaptations:
  - vocab_size: 4100 (6-mer) → 10 (single nucleotide)
  - Token IDs: canonical (0-9) + remap to NTv3/Caduceus
  - PE input: concatenate [CLS]R1[SEP]R2[SEP] instead of separate R1/R2 tensors
  - R2 processing: split hidden states at the SEP boundary

Components
----
1. BaseDNAEncoder   —  unified encoder interface
2. GlobalEncoder    —  NTv3 global encoder (canonical→NTv3 remap)
3. LocalEncoder     —  Caduceus local encoder (canonical→Caduceus remap)
4. DualEncoderBackbone  —  dual-encoder composition
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# ── Canonical → Model remap tables (from data_pipeline tokenizer) ──
NTV3_REMAP = {0: 1, 1: 3, 2: 4, 3: 2, 4: 0, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}
CADUCEUS_REMAP = {0: 4, 1: 0, 2: 1, 3: 3, 4: 6, 5: 7, 6: 10, 7: 8, 8: 9, 9: 11}


# ═══════════════════════════════════════════════════════════════════════════════ #
#  BaseDNAEncoder  —  unified encoder interface
# ═══════════════════════════════════════════════════════════════════════════════ #

class BaseDNAEncoder(nn.Module, ABC):
    """Abstract base class for DNA sequence encoders."""

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        encode DNA token sequence.

        Returns
        -------
        hidden_states : [B, L, D]
        pooled : [B, D]
        """
        ...

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the encoder output dimension D."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════ #
#  GlobalEncoder  —  NTv3 global encoder (v6.0: vocab=10, remap)
# ═══════════════════════════════════════════════════════════════════════════════ #

class GlobalEncoder(BaseDNAEncoder):
    """
    NTv3 global encoder.

    v6.0 changes:
      - vocab_size=10 (single nucleotide)
      - Accepts canonical token IDs, remapped internally to NTv3 IDs
      - Falls back to a trainable Transformer when the pretrained model is unavailable

    Parameters
    ----------
    vocab_size : int
        Canonical vocab size (default 10)
    hidden_dim : int
        Output hidden dimension (default 768)
    num_layers : int
        Number of Transformer layers (default 12)
    num_heads : int
        Number of attention heads (default 12)
    intermediate_dim : int
        FFN intermediate dimension (default 3072)
    dropout : float
        Dropout rate (default 0.1)
    max_position_embeddings : int
        Maximum positional encoding length (default 2048)
    pretrained_path : str
        Path to NTv3 pretrained weights
    use_flash_attention : bool
        Whether to use FlashAttention-2
    """

    def __init__(
        self,
        vocab_size: int = 10,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_dim: int = 3072,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        pretrained_path: str = "",
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # ── Remap table (canonical → NTv3) ──
        self.register_buffer(
            "_remap_table",
            torch.tensor([NTV3_REMAP.get(i, 0) for i in range(vocab_size)],
                         dtype=torch.long),
        )

        # ── Try to load NTv3 ──
        # v7: use the NTv3-pre variant (config.json auto_map → NTv3PreTrained),
        # which neither accepts nor needs species_ids, eliminating the v8.0
        # species_id=6 (clownfish) bug at its root.
        self._nt_model = None
        self._use_fallback = True

        if pretrained_path and os.path.isdir(pretrained_path):
            try:
                import sys
                if pretrained_path not in sys.path:
                    sys.path.insert(0, pretrained_path)
                # v7: import the NTv3-pre class directly from local files (local_files_only offline load).
                # Avoids the "InstaDeepAI/ntv3_base_model--" prefix in config.json auto_map
                # triggering a network fetch toward HuggingFace (with no external network this
                # machine would silently fall back to random initialization).
                from configuration_ntv3_pretrained import Ntv3PreTrainedConfig
                from modeling_ntv3_pretrained import NTv3PreTrained
                nt_config = Ntv3PreTrainedConfig.from_pretrained(
                    pretrained_path, local_files_only=True)
                self._nt_model = NTv3PreTrained.from_pretrained(
                    pretrained_path, config=nt_config, local_files_only=True,
                    ignore_mismatched_sizes=True,
                )
                nt_dim = getattr(nt_config, 'hidden_size',
                                 getattr(nt_config, 'embed_dim', hidden_dim))
                if nt_dim != hidden_dim:
                    self._proj = nn.Sequential(
                        nn.LayerNorm(nt_dim),
                        nn.Linear(nt_dim, hidden_dim),
                    )
                else:
                    self._proj = nn.Identity()
                self._use_fallback = False
                print(f"[GlobalEncoder] load NTv3-pre weights: {pretrained_path} "
                      f"(hidden={nt_dim}→{hidden_dim})")
            except Exception as e:
                import os
                if os.environ.get("DEEPFLUHGT_ALLOW_FALLBACK", "").lower() not in ("1", "true", "yes"):
                    raise RuntimeError(
                        f"[GlobalEncoder] NTv3 pretrained load failed: {e}. "
                        f"Refusing to silently fall back to a random-initialized transformer. "
                        f"Set DEEPFLUHGT_ALLOW_FALLBACK=1 to explicitly allow non-production fallback."
                    ) from e
                print("[GlobalEncoder] *** NON-PRODUCTION FALLBACK MODE *** "
                      "Results must NOT be used for paper. "
                      f"NTv3 load failed ({e}), using random init per explicit override.")

        if self._use_fallback:
            self._build_fallback_transformer(
                vocab_size, hidden_dim, num_layers, num_heads,
                intermediate_dim, dropout, max_position_embeddings,
            )
            print(f"[GlobalEncoder] Using trainable Transformer (random init): "
                  f"layers={num_layers}, dim={hidden_dim}, heads={num_heads}")

    def _build_fallback_transformer(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        intermediate_dim: int,
        dropout: float,
        max_position_embeddings: int,
    ):
        """Build the trainable Transformer fallback."""
        self._embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self._pos_embedding = nn.Embedding(max_position_embeddings, hidden_dim)
        self._embed_dropout = nn.Dropout(dropout)
        self._layer_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

    def _remap_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Canonical token IDs → NTv3 IDs."""
        return self._remap_table[input_ids]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """encode DNA token sequence → ([B, L, D], [B, D])."""
        if not self._use_fallback:
            return self._forward_nt(input_ids, attention_mask)
        return self._forward_fallback(input_ids, attention_mask)

    def _forward_nt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """NTv3-pre forward pass (remap IDs, pad to a multiple of 128).

        v7: NTv3-pre (MaskedLM) does not accept species_ids (the v6.8.0 species_id=6
        clownfish bug is already eliminated). attention_mask is accepted by signature
        but ignored inside the model; it is used only for external pooling.
        """
        mapped_ids = self._remap_ids(input_ids)
        B, L = mapped_ids.shape

        # NTv3 conv tower requires padding to a multiple of 128
        pad_mult = 128
        if L % pad_mult != 0:
            pad_len = pad_mult - (L % pad_mult)
            # NTv3 pad token id = 1
            pad_tensor = torch.full((B, pad_len), 1, dtype=mapped_ids.dtype,
                                    device=mapped_ids.device)
            mapped_ids = torch.cat([mapped_ids, pad_tensor], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(B, pad_len, dtype=attention_mask.dtype,
                                            device=attention_mask.device)], dim=1)

        # NTv3-pre forward: return MaskedLMOutput (loss/logits/hidden_states/attentions)
        outputs = self._nt_model(
            input_ids=mapped_ids,
            output_hidden_states=True,
            attention_mask=attention_mask,
        )

        # hidden_states[-1] = deconv tower output (B, L_padded, nt_dim), already restored
        # to full resolution. Crop back to the original (pre-pad) length L.
        hidden = outputs.hidden_states[-1][:, :L, :]

        # fp32 projection: prevents cumulative fp16 overflow → NaN in [B,L,1536]x[1536,768] under AMP
        with torch.amp.autocast(device_type=hidden.device.type, enabled=False):
            hidden = self._proj(hidden.float())

        # Pool: mean pooling with the attention mask
        orig_mask = attention_mask[:, :L].unsqueeze(-1).float()
        pooled = (hidden * orig_mask).sum(dim=1) / orig_mask.sum(dim=1).clamp(min=1)

        return hidden, pooled

    def _forward_fallback(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback Transformer forward pass (uses canonical IDs directly)."""
        B, L = input_ids.shape

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self._embedding(input_ids) + self._pos_embedding(positions)
        x = self._embed_dropout(x)

        key_padding_mask = (attention_mask == 0)
        hidden_states = self._transformer(x, src_key_padding_mask=key_padding_mask)
        hidden_states = self._layer_norm(hidden_states)

        # BUG-011 fix: use masked mean pooling instead of CLS pooling,
        # consistent with the NT path, so the fallback path produces a pooled
        # representation with the same semantics.
        mask_exp = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return hidden_states, pooled

    def get_output_dim(self) -> int:
        return self.hidden_dim


# ═══════════════════════════════════════════════════════════════════════════════ #
#  LocalEncoder  —  Caduceus local encoder (v6.0: vocab=10, remap)
# ═══════════════════════════════════════════════════════════════════════════════ #

class LocalEncoder(BaseDNAEncoder):
    """
    Caduceus local encoder.

    v6.0 changes:
      - vocab_size=10
      - Accepts canonical token IDs, remapped internally to Caduceus IDs
      - Fallback Transformer uses vocab_size=10

    Parameters
    ----------
    vocab_size : int
        Canonical vocab size (default 10)
    hidden_dim : int
        Hidden dimension (default 256, Caduceus d_model)
    num_layers : int
        Number of layers (default 16)
    intermediate_dim : int
        FFN intermediate dimension (default 1024)
    dropout : float
        Dropout rate
    max_position_embeddings : int
        Maximum sequence length (default 512)
    pretrained_path : str
        Path to Caduceus pretrained weights
    output_dim : int
        Output projection dimension
    """

    def __init__(
        self,
        vocab_size: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 16,
        intermediate_dim: int = 1024,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
        pretrained_path: str = "",
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self._use_fallback = True
        self._caduceus_model = None

        # ── Remap table (canonical → Caduceus) ──
        self.register_buffer(
            "_remap_table",
            torch.tensor([CADUCEUS_REMAP.get(i, 4) for i in range(vocab_size)],
                         dtype=torch.long),
        )

        # ── Try to load Caduceus ──
        if pretrained_path and os.path.isdir(pretrained_path):
            try:
                import sys
                if pretrained_path not in sys.path:
                    sys.path.insert(0, pretrained_path)
                from transformers import AutoModel
                self._caduceus_model = AutoModel.from_pretrained(
                    pretrained_path,
                    trust_remote_code=True,
                )
                # d_model may differ from the actual output dimension (SSM expand factor)
                _expand = 1
                if hasattr(self._caduceus_model.config, 'ssm_cfg') and self._caduceus_model.config.ssm_cfg:
                    _expand = self._caduceus_model.config.ssm_cfg.get('expand', 1)
                cad_dim = self._caduceus_model.config.d_model * _expand
                if cad_dim != self.output_dim:
                    self._proj = nn.Linear(cad_dim, self.output_dim)
                else:
                    self._proj = nn.Identity()
                self._use_fallback = False
                print(f"[LocalEncoder] load Caduceus weights: {pretrained_path}")
            except Exception as e:
                print(f"[LocalEncoder] Caduceus load failed ({e}), falling back to random init")

        if self._use_fallback:
            self._build_fallback(
                vocab_size, hidden_dim, num_layers,
                intermediate_dim, dropout, max_position_embeddings,
            )
            print(f"[LocalEncoder] Using trainable Transformer (random init): "
                  f"layers={num_layers}, dim={hidden_dim}")

        # ── Output projection ──
        if self.hidden_dim != self.output_dim and self._use_fallback:
            self._out_proj = nn.Linear(self.hidden_dim, self.output_dim)
        else:
            self._out_proj = nn.Identity()

    def _build_fallback(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        intermediate_dim: int,
        dropout: float,
        max_position_embeddings: int,
    ):
        """Build the fallback Transformer."""
        self._embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self._pos_embedding = nn.Embedding(max_position_embeddings, hidden_dim)
        self._embed_dropout = nn.Dropout(dropout)
        self._layer_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, hidden_dim // 64),
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

    def _remap_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Canonical token IDs → Caduceus IDs."""
        return self._remap_table[input_ids]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """encode DNA token sequence → ([B, L, output_dim], [B, output_dim])."""
        if not self._use_fallback:
            return self._forward_caduceus(input_ids, attention_mask)
        return self._forward_fallback(input_ids, attention_mask)

    def _forward_caduceus(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Caduceus forward pass (remap IDs). Caduceus does not support attention_mask."""
        mapped_ids = self._remap_ids(input_ids)
        outputs = self._caduceus_model(input_ids=mapped_ids)
        # fp32 projection: prevents cumulative fp16 overflow under AMP
        with torch.amp.autocast(device_type=outputs.last_hidden_state.device.type, enabled=False):
            hidden_states = self._proj(outputs.last_hidden_state.float())
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return hidden_states, pooled

    def _forward_fallback(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback Transformer forward pass."""
        B, L = input_ids.shape

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self._embedding(input_ids) + self._pos_embedding(positions)
        x = self._embed_dropout(x)

        key_padding_mask = (attention_mask == 0)
        hidden_states = self._transformer(x, src_key_padding_mask=key_padding_mask)
        hidden_states = self._layer_norm(hidden_states)
        hidden_states = self._out_proj(hidden_states)

        # BUG-011 fix: use masked mean pooling instead of CLS pooling,
        # consistent with the Caduceus path.
        mask_exp = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return hidden_states, pooled

    def get_output_dim(self) -> int:
        return self.output_dim


# ═══════════════════════════════════════════════════════════════════════════════ #
#  DualEncoderBackbone  —  dual-encoder composition
# ═══════════════════════════════════════════════════════════════════════════════ #

class DualEncoderBackbone(nn.Module):
    """
    Dual-encoder backbone network (v6.0).

    v6.0 changes:
      - No longer accepts separate r2_input_ids / r2_attention_mask
      - In PE mode, R1+R2 are already concatenated in input_ids: [CLS]R1[SEP]R2[SEP]
      - forward() splits hidden states at the SEP boundary to get the R2 part

    Parameters
    ----------
    global_encoder : GlobalEncoder
    local_encoder : LocalEncoder, optional
    hidden_dim : int
        Final output dimension
    read_length : int
        PE read length (default 150)
    data_mode : str
        "paired_end" | "contig"
    """

    def __init__(
        self,
        global_encoder: GlobalEncoder,
        local_encoder: Optional[LocalEncoder] = None,
        hidden_dim: int = 768,
        read_length: int = 150,
        data_mode: str = "paired_end",
    ):
        super().__init__()
        self.global_encoder = global_encoder
        self.local_encoder = local_encoder
        self.hidden_dim = hidden_dim
        self.read_length = read_length
        self.data_mode = data_mode

        # ── Local encoder projection ──
        if local_encoder is not None:
            local_dim = local_encoder.get_output_dim()
            if local_dim != hidden_dim:
                self._local_proj = nn.Sequential(
                    nn.LayerNorm(local_dim),
                    nn.Linear(local_dim, hidden_dim),
                )
            else:
                self._local_proj = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Dual-encoder forward pass.

        v6.0 PE input format: [CLS]R1[0..149][SEP]R2[0..149][SEP] = 303 tokens
        v6.0 Contig input format: [CLS]seq[0..L-1][SEP]

        Returns
        -------
        dict with keys:
            "global_hidden" : [B, L, D]
            "global_pooled" : [B, D]
            "local_hidden"  : [B, L, D] or None
            "local_pooled"  : [B, D] or None
            "r2_global_hidden" : [B, L_r2, D] or None  (PE only, from SEP split)
            "r2_global_pooled" : [B, D] or None
        """
        # ── Global encoder ──
        global_hidden, global_pooled = self.global_encoder(input_ids, attention_mask)

        result = {
            "global_hidden": global_hidden,
            "global_pooled": global_pooled,
            "local_hidden": None,
            "local_pooled": None,
            "r2_global_hidden": None,
            "r2_global_pooled": None,
        }

        # ── Local encoder (optional) ──
        if self.local_encoder is not None:
            local_hidden, local_pooled = self.local_encoder(input_ids, attention_mask)
            with torch.amp.autocast(device_type=local_hidden.device.type, enabled=False):
                local_hidden = self._local_proj(local_hidden.float())
                # m-9 fix (2026-06-22): removed the Identity double projection of local_pooled
                # _local_proj=Identity (local_dim=768=hidden_dim); local_pooled is read by api.py but not consumed
            result["local_hidden"] = local_hidden
            result["local_pooled"] = local_pooled

        # ── PE mode: split out the R2 part from the concatenated sequence ──
        # PE layout: [CLS]R1[SEP]R2[SEP]
        # R1 at positions [1, read_length+1)
        # SEP at position read_length+1
        # R2 at positions [read_length+2, 2*read_length+2)
        # SEP at position 2*read_length+2
        if self.data_mode == "paired_end":
            r2_start = self.read_length + 2   # +2: CLS(1) + SEP(1)
            r2_end = 2 * self.read_length + 2
            r2_global_hidden = global_hidden[:, r2_start:r2_end, :]  # [B, read_length, D]

            # R2 mean pooling (protected by attention_mask)
            r2_mask = attention_mask[:, r2_start:r2_end].unsqueeze(-1).float()
            r2_global_pooled = (r2_global_hidden * r2_mask).sum(dim=1) / r2_mask.sum(dim=1).clamp(min=1)

            result["r2_global_hidden"] = r2_global_hidden
            result["r2_global_pooled"] = r2_global_pooled

        return result

    def get_output_dim(self) -> int:
        return self.hidden_dim
