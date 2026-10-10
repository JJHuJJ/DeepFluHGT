#!/usr/bin/env python3
# DeepFluHGT.py
"""
DeepFluHGT  —  unified entry point
============================
usage:
  python DeepFluHGT.py simulate   [options]
  python DeepFluHGT.py train      [options]
  python DeepFluHGT.py eval       [options]
  python DeepFluHGT.py predict    [options]
  python DeepFluHGT.py test       [module]
"""

import argparse
import os
import sys
import time

# Ensure the project root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema import MODULES


def load_config(config_path):
    """Load a YAML configuration."""
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            print("[WARN] PyYAML not installed, using defaults")
    return {}


# ═══════════════════════════════════════════════════════════════════════════════ #
#  simulate
# ═══════════════════════════════════════════════════════════════════════════════ #

def cmd_simulate(args):
    """Run the simulator module. Priority order: CLI > YAML > SimulatorConfig defaults."""
    from simulator.contracts import SimulatorConfig
    from simulator.api import simulate_dataset, validate_output

    # Load YAML config
    yaml_cfg = load_config(args.config)
    sim_cfg = yaml_cfg.get("simulator", {})

    # Helper: CLI > YAML > hard-coded defaults (from SimulatorConfig)
    def _v(arg_val, yaml_key, default):
        return arg_val if arg_val is not None else sim_cfg.get(yaml_key, default)

    config = SimulatorConfig(
        mode=args.mode,
        virus_fasta=args.virus_fasta,
        microbe_dir=args.microbe_dir,
        output_dir=_v(args.output_dir, "output_dir", "./simulated_data"),
        seed=_v(args.seed, "seed", 42),
        n_pos=_v(args.n_pos, "n_pos", 10000),
        n_neg=_v(args.n_neg, "n_neg", 10000),
        read_len=_v(args.read_len, "read_len", 150),
        contig_min_length=_v(args.contig_min, "contig_min_length", 300),
        contig_max_length=_v(args.contig_max, "contig_max_length", 100000),
        error_rate=_v(args.error_rate, "error_rate", 0.005),
        gc_bias=_v(args.gc_bias, "gc_bias", False),
        use_coverage=_v(args.use_coverage, "use_coverage", False),
        processes=_v(args.processes, "processes", 1),
        do_split=not _v(args.no_split, "no_split", False),
        train_ratio=_v(args.train_ratio, "train_ratio", 0.8),
        val_ratio=_v(args.val_ratio, "val_ratio", 0.1),
        test_ratio=_v(args.test_ratio, "test_ratio", 0.1),
        insertion_prob=_v(args.insertion_prob, "insertion_prob", 0.6),
        replacement_prob=_v(args.replacement_prob, "replacement_prob", 0.3),
        deletion_prob=_v(args.deletion_prob, "deletion_prob", 0.1),
        multi_event_enabled=_v(args.multi_event, "multi_event_enabled", True),
        multi_event_prob=_v(args.multi_event_prob, "multi_event_prob", 0.2),
        multi_event_max=_v(args.multi_event_max, "multi_event_max", 3),
        multi_event_min_gap=_v(args.multi_event_min_gap, "multi_event_min_gap", 20),
        sandwich_enabled=_v(args.sandwich, "sandwich_enabled", True),
        sandwich_prob=_v(args.sandwich_prob, "sandwich_prob", 0.15),
        sandwich_min_gap=_v(args.sandwich_min_gap, "sandwich_min_gap", 5),
        virus_fragment_min=_v(args.virus_frag_min, "virus_fragment_min", 20),
        virus_fragment_max=_v(args.virus_frag_max, "virus_fragment_max", 200),
        virus_fragments_per_genome=_v(args.virus_frag_per_genome,
                                      "virus_fragments_per_genome", 500),
        multi_source_host_enabled=_v(args.multi_host, "multi_source_host_enabled", True),
        multi_source_host_prob=_v(args.multi_host_prob, "multi_source_host_prob", 0.2),
        multi_source_host_max=_v(args.multi_host_max, "multi_source_host_max", 3),
        # Stratification/distribution settings (read from YAML only)
        virus_frag_tiers_pe=sim_cfg.get("virus_frag_tiers", {}).get("paired_end"),
        virus_frag_tiers_contig=sim_cfg.get("virus_frag_tiers", {}).get("contig"),
        pe_case_ratios=sim_cfg.get("pe_case_ratios"),
        neg_type_ratios=sim_cfg.get("neg_type_ratios"),
        contig_hgt_mode_ranges=sim_cfg.get("contig_hgt_mode_ranges"),
        # Biological realism enhancement parameters
        seq_platform=_v(args.seq_platform, "seq_platform", "illumina"),
        coverage_distribution=_v(args.coverage_distribution, "coverage_distribution", "lognormal"),
        coverage_zipf_alpha=_v(args.coverage_zipf_alpha, "coverage_zipf_alpha", 1.0),
        batch_effect_sigma=_v(args.batch_effect_sigma, "batch_effect_sigma", 0.0),
        contamination_rate=_v(args.contamination_rate, "contamination_rate", 0.0),
        contaminant_dir=_v(args.contaminant_dir, "contaminant_dir", ""),
        assembly_artifact_rate=_v(args.assembly_artifact_rate, "assembly_artifact_rate", 0.0),
        label_noise=_v(args.label_noise, "label_noise", 0.0),
        boundary_shift_max=_v(args.boundary_shift_max, "boundary_shift_max", 0),
        evolutionary_distance=_v(args.evolutionary_distance, "evolutionary_distance", 0.0),
        evolutionary_distance_tiers=sim_cfg.get("evolutionary_distance_tiers"),
        host_mutation_rate=_v(args.host_mutation_rate, "host_mutation_rate", 0.0),
        open_set_virus=_v(args.open_set_virus, "open_set_virus", False),
        gc_amelioration=_v(args.gc_amelioration, "gc_amelioration", False),
        evo_distance_distribution=_v(args.evo_distance_distribution, "evo_distance_distribution", "uniform"),
        evo_distance_gamma_shape=_v(args.evo_distance_gamma_shape, "evo_distance_gamma_shape", 2.0),
        microhomology_len=_v(args.microhomology_len, "microhomology_len", 0),
        # Genome isolation
        genome_isolation=_v(args.genome_isolation, "genome_isolation", True),
        genome_isolation_microbe_ratios=sim_cfg.get("genome_isolation_microbe_ratios"),
        genome_isolation_virus_ratios=sim_cfg.get("genome_isolation_virus_ratios"),
    )

    output = simulate_dataset(config)
    ok = validate_output(output)
    if not ok:
        print("[ERROR] Output validation failed!")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════ #
#  train
# ═══════════════════════════════════════════════════════════════════════════════ #


def _resolve_resume_checkpoint(resume_path):
    """
    parse resume path, returnactual checkpoint filepath.

    Supports three kinds of input:
      - file path (used directly)
      - directory path (auto-selects last_model.pt > best_model.pt > latest epoch_N.pt)
      - None (return None)
    """
    if not resume_path:
        return None

    # Case 1: directly a file
    if os.path.isfile(resume_path):
        return resume_path

    # Case 2: a directory, auto-find the most recent checkpoint
    if os.path.isdir(resume_path):
        # Priority order: last_model.pt > best_model.pt > latest epoch_N.pt
        for name in ["last_model.pt", "best_model.pt"]:
            candidate = os.path.join(resume_path, name)
            if os.path.isfile(candidate):
                print(f"[Resume] Found {name} in directory {resume_path}")
                return candidate

        # Find the latest epoch_N.pt
        epoch_ckpts = []
        for f in os.listdir(resume_path):
            if f.startswith("epoch_") and f.endswith(".pt"):
                try:
                    ep = int(f.replace("epoch_", "").replace(".pt", ""))
                    epoch_ckpts.append((ep, os.path.join(resume_path, f)))
                except ValueError:
                    pass
        if epoch_ckpts:
            epoch_ckpts.sort(reverse=True)
            print(f"[Resume] Found {os.path.basename(epoch_ckpts[0][1])} in directory {resume_path}")
            return epoch_ckpts[0][1]

        print(f"[WARN] No checkpoint file found in directory {resume_path}")
        return None

    # Case 3: path does not exist
    print(f"[WARN] Resume pathdoes not exist: {resume_path}")
    return None

def cmd_train(args):
    """Training entry point  —  auto-detects train/val/test and outputs complete training information."""
    import torch
    from torch.utils.data import DataLoader
    from model.contracts import HGTConfig
    from model.api import build_model
    from loss.contracts import LossConfig
    from loss.api import build_loss
    from training_engine.contracts import TrainingConfig
    from training_engine.api import TrainingEngine
    from data_pipeline.api import load_splits, collate
    from data_pipeline.impl.collator import HGTCollator

    # Load YAML config
    yaml_cfg = load_config(args.config)
    data_cfg = yaml_cfg.get("data", {})
    train_cfg = yaml_cfg.get("training", {})
    model_cfg = yaml_cfg.get("model", {})
    loss_cfg = yaml_cfg.get("loss", {})

    # override: CLI parameters > YAML > defaultvalue
    data_mode = args.mode or data_cfg.get("mode", "paired_end")
    lr = args.lr or train_cfg.get("learning_rate", 3e-5)
    epochs = args.epochs or train_cfg.get("max_epochs", 30)
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    data_dir = args.data_dir
    max_length = args.max_length or data_cfg.get("max_contig_length", 10000)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Global seed control (BUG-026 fix) ──
    from utils.impl.seed import seed_all
    seed = train_cfg.get("seed", 42)
    seed_all(seed)

    print("=" * 70)
    print("DeepFluHGT  —  Training")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Data: {data_mode}  LR: {lr}  Epochs: {epochs}  Batch: {batch_size}  Seed: {seed}")
    print("=" * 70)

    # Build model
    model_config = HGTConfig(
        data_mode=data_mode,
        hidden_dim=model_cfg.get("hidden_dim", 768),
        local_hidden_dim=model_cfg.get("local_hidden_dim", 256),
        num_attention_heads=model_cfg.get("num_attention_heads", 12),
        dropout=model_cfg.get("dropout", 0.1),
        graph_top_k=model_cfg.get("graph_top_k", 5),
        bio_feature_dim=model_cfg.get("bio_feature_dim", 260),
        pretrained_nt_path=model_cfg.get("pretrained_nt_path", ""),
        pretrained_caduceus_path=model_cfg.get("pretrained_caduceus_path", ""),
        use_local_encoder=model_cfg.get("use_local_encoder", True),
        use_evo=model_cfg.get("use_evo", True),
        pretrained_evo_dim=model_cfg.get("pretrained_evo_dim", 4096),
        evo_hidden=model_cfg.get("evo_hidden", 512),
    )
    model = build_model(model_config)

    # Build loss
    loss_config = LossConfig(
        use_uncertainty_weight=loss_cfg.get("use_uncertainty_weight", False),  # BUG-023 fix: alignment dataclass default
        region_pos_weight=loss_cfg.get("region_pos_weight", 10.0),
        cascade_gate_slope=loss_cfg.get("cascade_gate_slope", 10.0),
        # ── Focal Loss ──
        use_focal_loss=loss_cfg.get("use_focal_loss", False),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        # ── Manual weights (effective when use_uncertainty_weight=False) ──
        cls_weight=loss_cfg.get("cls_weight", 1.0),
        bp_weight=loss_cfg.get("bp_weight", 0.3),
        role_weight=loss_cfg.get("role_weight", 0.2),
        direction_weight=loss_cfg.get("direction_weight", 0.2),
        breakpoint_sigma=loss_cfg.get("breakpoint_sigma", 5.0),
        # BUG-023 fix: the 4 weights below were previously not passed through from YAML;
        # they always used the dataclass defaults
        disentangle_weight=loss_cfg.get("disentangle_weight", 0.1),
        rc_consistency_weight=loss_cfg.get("rc_consistency_weight", 0.1),
        refiner_weight=loss_cfg.get("refiner_weight", 0.5),
        early_exit_weight=loss_cfg.get("early_exit_weight", 0.25),
        neg_bp_reg_weight=loss_cfg.get("neg_bp_reg_weight", 0.1),
    )
    loss_fn = build_loss(loss_config)

    # Build training config
    training_config = TrainingConfig(
        learning_rate=lr,
        batch_size=batch_size,
        backbone_lr_ratio=train_cfg.get("backbone_lr_ratio", 0.1),
        lora_lr_multiplier=train_cfg.get("lora_lr_multiplier", 2.0),
        weight_decay=train_cfg.get("weight_decay", 0.05),  # M-04 fix: alignment dataclass/YAML
        max_epochs=epochs,
        gradient_clip_norm=train_cfg.get("gradient_clip_norm", 5.0),
        accumulation_steps=train_cfg.get("accumulation_steps", 1),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.06),
        schedule_type=train_cfg.get("schedule_type", "cosine"),
        use_lora=train_cfg.get("use_lora", False),
        lora_rank=train_cfg.get("lora_rank", 8),
        lora_alpha=train_cfg.get("lora_alpha", 16.0),
        use_amp=train_cfg.get("use_amp", device == "cuda"),  # BUG-024 fix: read YAML
        amp_dtype=train_cfg.get("amp_dtype", "float16"),  # bf16 switch (Contig fp16 overflow fix)
        device=device,
        checkpoint_dir=args.ckpt_dir or train_cfg.get("checkpoint_dir", "./checkpoints"),
        # fix: save_epochs was previously not passed in from YAML (always default 5),
        # breaking the epoch_4/8/12 snapshot schedule. Bugfix-level change (the key
        # previously had no effect), not a design change.
        save_epochs=train_cfg.get("save_epochs", 5),
        mixup_alpha=train_cfg.get("mixup_alpha", 0.2),
        early_stop_patience=(
            train_cfg.get("early_stop_patience")
            or yaml_cfg.get("utils", {}).get("early_stop_patience", 5)
        ),
        freeze_epochs=train_cfg.get("freeze_epochs", 2),
        log_interval=train_cfg.get("log_interval", 50),
        global_lr_ratio=train_cfg.get("global_lr_ratio", 0.1),
        local_lr_ratio=train_cfg.get("local_lr_ratio", 0.2),
        # ── FGM adversarial training ──
        use_fgm=train_cfg.get("use_fgm", False),
        fgm_epsilon=train_cfg.get("fgm_epsilon", 1.0),
        fgm_start_epoch=train_cfg.get("fgm_start_epoch", 3),
        # ── LoRA / Mixup additions ──
        lora_dropout=train_cfg.get("lora_dropout", 0.0),
        mixup_warmup_epochs=train_cfg.get("mixup_warmup_epochs", 2),
    )

    # ── Contract provenance: config and dataset hashes (persisted with every checkpoint) ──
    import hashlib as _hashlib
    try:
        with open(args.config, "rb") as _f:
            training_config.config_sha256 = _hashlib.sha256(_f.read()).hexdigest()
    except Exception:
        pass
    if data_dir:
        try:
            _h = _hashlib.sha256()
            for _sp in ("train", "val", "test"):
                _h.update(_sp.encode())
                with open(os.path.join(data_dir, _sp, "breakpoints.bed"), "rb") as _f:
                    _h.update(_f.read())
            training_config.dataset_manifest_sha256 = _h.hexdigest()
        except Exception:
            pass

    # ── dataload ──
    if args.dry_run or not data_dir:
        # Synthetic data demo
        print("\n[Data] Using synthetic demo data (dry_run mode)")
        from data_pipeline.contracts import HGTBatch

        N = 100
        torch.manual_seed(42)
        labels = torch.cat([torch.ones(50), torch.zeros(50)]).long()
        perm = torch.randperm(100)
        labels = labels[perm]

        pos_pattern = torch.tensor([5, 6, 5, 6, 5, 6, 5, 6, 5, 6])
        neg_pattern = torch.tensor([7, 8, 7, 8, 7, 8, 7, 8, 7, 8])
        input_ids = torch.zeros(N, 303, dtype=torch.long)
        for i in range(N):
            pat = pos_pattern if labels[i] == 1 else neg_pattern
            input_ids[i] = pat.repeat((303 + 9) // 10)[:303]

        hgt_modes = torch.zeros(N, dtype=torch.long)
        for i in range(N):
            if labels[i] == 1:
                hgt_modes[i] = torch.randint(0, 3, (1,)).item()
            else:
                hgt_modes[i] = 3

        n_events = torch.zeros(N, dtype=torch.long)
        n_events[labels == 1] = torch.randint(1, 3, (50,))

        # Split 8:1:1
        n_train = 80
        n_val = 10

        train_batches = []
        val_batches = []
        test_batches = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b = end - start
            batch = HGTBatch(
                input_ids=input_ids[start:end],
                attention_mask=torch.ones(b, 303, dtype=torch.long),
                bio_features=torch.randn(b, 260),
                labels=labels[start:end],
                breakpoint_targets=torch.zeros(b, 303),
                region_targets=torch.zeros(b, 303),
                pe_case=torch.zeros(b, dtype=torch.long),
                hgt_mode=hgt_modes[start:end],
                n_events=n_events[start:end],
            )
            if start < n_train:
                train_batches.append(batch)
            elif start < n_train + n_val:
                val_batches.append(batch)
            else:
                test_batches.append(batch)

        print(f"  Synthetic data: train={n_train}, val={n_val}, test={N - n_train - n_val}")

        # Build engine
        engine = TrainingEngine(model, loss_fn, training_config)
        engine.setup()

        # trainingloop
        _run_training_loop(
            engine, train_batches, val_batches, test_batches,
            epochs, device, args.dry_run, training_config.checkpoint_dir,
        )
    else:
        # Real data: auto-detect train/val/test
        print(f"\n[Data] Loading from: {data_dir}")

        # ── Sliding-window mode (Contig) ──
        use_sliding_window = (data_mode == "contig" and
                              data_cfg.get("use_sliding_window", False))

        if use_sliding_window:
            from data_pipeline.api import load_splits_sliding_window

            sw_window_size = data_cfg.get("sw_window_size", 1500)
            sw_overlap = data_cfg.get("sw_overlap", 300)
            sw_max_neg = data_cfg.get("sw_max_neg_windows", 2)
            sw_max_eval = data_cfg.get("sw_max_eval_windows", 10)

            print(f"  [Sliding Window] window={sw_window_size}bp, "
                  f"overlap={sw_overlap}bp, max_neg_windows={sw_max_neg}, "
                  f"max_eval_windows={sw_max_eval}")

            # Adjust model max_contig_length to the window size
            model_config = HGTConfig(
                data_mode=data_mode,
                max_contig_length=sw_window_size,
                hidden_dim=model_cfg.get("hidden_dim", 768),
                local_hidden_dim=model_cfg.get("local_hidden_dim", 256),
                num_attention_heads=model_cfg.get("num_attention_heads", 12),
                dropout=model_cfg.get("dropout", 0.1),
                graph_top_k=model_cfg.get("graph_top_k", 5),
                bio_feature_dim=model_cfg.get("bio_feature_dim", 260),
                pretrained_nt_path=model_cfg.get("pretrained_nt_path", ""),
                pretrained_caduceus_path=model_cfg.get("pretrained_caduceus_path", ""),
                use_local_encoder=model_cfg.get("use_local_encoder", True),
                use_evo=model_cfg.get("use_evo", True),
                pretrained_evo_dim=model_cfg.get("pretrained_evo_dim", 4096),
                evo_hidden=model_cfg.get("evo_hidden", 512),
            )
            model = build_model(model_config)

            splits = load_splits_sliding_window(
                data_dir=data_dir,
                window_size=sw_window_size,
                overlap=sw_overlap,
                max_neg_windows=sw_max_neg,
                max_eval_windows=sw_max_eval,
                read_len=data_cfg.get("read_length", 150),
                evo_cache_path=data_cfg.get("evo_cache_path", ""),
                rc_aug_prob=data_cfg.get("rc_aug_prob", 0.0),
            )
        else:
            splits = load_splits(
                data_dir=data_dir,
                mode=data_mode,
                read_len=data_cfg.get("read_length", 150),
                evo_cache_path=data_cfg.get("evo_cache_path", ""),
                rc_aug_prob=data_cfg.get("rc_aug_prob", 0.0),
                max_length=max_length,
            )

        train_ds = splits.get("train")
        val_ds = splits.get("val")
        test_ds = splits.get("test")

        # Create DataLoader (BUG-026 fix: reproducible shuffle)
        from utils.impl.seed import worker_init_fn as _worker_init
        collator = HGTCollator(pad_id=0)
        _utils_cfg = yaml_cfg.get("utils", {})
        # DATALOADER-001 fix: dual-key compatible read
        # YAML configs use num_workers (no prefix), legacy code uses dataloader_num_workers (prefixed)
        # Prefer the prefixed key, fall back to the unprefixed key, then the default value
        _dl_workers = _utils_cfg.get("dataloader_num_workers",
                      _utils_cfg.get("num_workers", 10))
        _dl_prefetch = _utils_cfg.get("dataloader_prefetch_factor",
                       _utils_cfg.get("prefetch_factor", 2))
        _dl_persistent = _utils_cfg.get("dataloader_persistent_workers",
                         _utils_cfg.get("persistent_workers", False))
        _g = torch.Generator()
        _g.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, collate_fn=collator,
                                  num_workers=_dl_workers, pin_memory=True,
                                  prefetch_factor=_dl_prefetch,
                                  persistent_workers=_dl_persistent,
                                  generator=_g,
                                  worker_init_fn=lambda wid: _worker_init(wid, seed))
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, collate_fn=collator,
                                num_workers=_dl_workers, pin_memory=True,
                                prefetch_factor=_dl_prefetch,
                                persistent_workers=_dl_persistent) if val_ds else None
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 shuffle=False, collate_fn=collator,
                                 num_workers=_dl_workers, pin_memory=True,
                                 prefetch_factor=_dl_prefetch,
                                 persistent_workers=_dl_persistent) if test_ds else None

        print(f"  train={len(train_ds)}, val={len(val_ds) if val_ds else 0}, "
              f"test={len(test_ds) if test_ds else 0}")

        # Build engine
        engine = TrainingEngine(model, loss_fn, training_config)

        # ── Resume from checkpoint ──
        resume_epoch = 0
        resume_was_early_stopped = False
        freeze_epochs = train_cfg.get("freeze_epochs", 2)
        resume_ckpt = _resolve_resume_checkpoint(args.resume)

        if resume_ckpt:
            import torch as _torch
            _ckpt = _torch.load(resume_ckpt, map_location="cpu", weights_only=False)
            resume_epoch = _ckpt.get("epoch", 0)
            best_epoch = _ckpt.get("best_epoch", 0)
            ckpt_patience = _ckpt.get("patience_counter", 0)
            ckpt_patience_max = _ckpt.get("early_stop_patience", 5)
            ckpt_batch_size = _ckpt.get("config", {}).get("batch_size")
            has_patience_info = "patience_counter" in _ckpt
            resume_was_early_stopped = ckpt_patience >= ckpt_patience_max

            # Legacy checkpoint fallback: no early-stopping info but last_epoch >> best_epoch → suspected early stopping
            if not has_patience_info and resume_epoch > 0:
                epochs_since_best = resume_epoch - best_epoch
                # Ran >= patience epochs after best_epoch, very likely early stopping
                suspected_early_stop = epochs_since_best >= ckpt_patience_max
                if suspected_early_stop:
                    resume_was_early_stopped = True
                    ckpt_patience = ckpt_patience_max  # Mark as early-stopped

            # Pre-set the epoch so setup() builds the correct optimizer (frozen/unfrozen)
            engine.current_epoch = resume_epoch

            is_frozen = resume_epoch < freeze_epochs
            print(f"\n[Resume] checkpoint: {resume_ckpt}")
            print(f"  Completed epochs: {resume_epoch}  (best_epoch={best_epoch})")
            print(f"  freeze_epochs={freeze_epochs} → backbone is {'frozen' if is_frozen else 'unfrozen'}")

            # Detect early stopping + parameter changes
            params_changed = (
                (ckpt_batch_size is not None and batch_size != ckpt_batch_size)
                or (lr != train_cfg.get("learning_rate", 3e-5))
            )

            if resume_was_early_stopped:
                if has_patience_info:
                    es_info = f"patience={ckpt_patience}/{ckpt_patience_max}"
                else:
                    es_info = f"suspected early stopping (best_epoch={best_epoch} is {resume_epoch - best_epoch} epochs behind current)"

                if params_changed:
                    print(f"  ⚠ Previous run terminated by early stopping ({es_info})")
                    print(f"  Training parameters have changed (batch_size: {ckpt_batch_size}→{batch_size})")
                    print(f"  → Auto-resetting patience, continuing training")
                else:
                    print(f"  ⚠ Previous run terminated by early stopping ({es_info})")
                    print(f"  Training parameters unchanged; continuing may not improve results")
                    print(f"  Consider adjusting parameters (batch_size / lr) for better results")

        engine.setup(steps_per_epoch=len(train_loader))

        # ── Contract: encoder permanently frozen assertion ──
        # freeze_epochs >= max_epochs means permanently frozen (the unfreeze trigger
        # condition current_epoch == freeze_epochs can never be met during training).
        # Explicit assertion up front to prevent "config says frozen, code actually unfreezes".
        if training_config.freeze_epochs >= training_config.max_epochs:
            backbone_frozen = all(
                not p.requires_grad for p in model.backbone.parameters())
            assert backbone_frozen, \
                ("[Contract] encoder-freeze violation: freeze_epochs >= max_epochs "
                 "but backbone has parameters with requires_grad=True")
            opt_param_ids = {id(p) for g in engine.optimizer.param_groups
                             for p in g["params"]}
            leak = [n for n, p in model.backbone.named_parameters()
                    if id(p) in opt_param_ids]
            assert not leak, f"[Contract] backbone parameters leaked into the optimizer: {leak[:5]}"
            print("[Contract] encoder permanently frozen ✓ "
                  "(backbone requires_grad=False, not in the optimizer; "
                  "trainable = projection/fusion/heads)")

        else:
            # The contract requires permanently frozen; freeze_epochs < max_epochs means the
            # encoder unfreezes mid-training (v6.8 legacy path), which is incompatible with the
            # "frozen representation evaluation" methodological claim. Loud warning to prevent silent violation.
            print(f"[Contract] ⚠⚠ warning: freeze_epochs({training_config.freeze_epochs}) "
                  f"< max_epochs({training_config.max_epochs})  —  encoder will unfreeze at "
                  f"epoch {training_config.freeze_epochs}, violating the permanently-frozen contract!"
                  f" If this is not intentional, set freeze_epochs to max_epochs.")

        # Deployment preflight mode: exit after the contract assertion passes (one of three checks for M0 deployment on remote machines)
        if getattr(args, "preflight", False):
            print("[Preflight] dataload ✓ model build ✓ contract assert ✓  —  preflight passed, not training")
            return

        # ── Resilience: run_manifest.json (auto-generated on every start, immutable provenance) ──
        import json as _jm
        import datetime as _dt
        os.makedirs(training_config.checkpoint_dir, exist_ok=True)
        _manifest = {
            "run_id": f"{os.path.splitext(os.path.basename(args.config))[0]}"
                      f"_ep{engine.current_epoch}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": _dt.datetime.now().isoformat(timespec="seconds"),
            "config": args.config,
            "config_sha256": training_config.config_sha256,
            "dataset_manifest_sha256": training_config.dataset_manifest_sha256,
            "resume_from": str(getattr(args, "resume", None) or "none"),
            "gpu_uuid_env": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device": str(engine.device),
            "seed": locals().get("seed", 42),
            "max_epochs": training_config.max_epochs,
            "freeze_epochs": training_config.freeze_epochs,
            "effective_batch": training_config.batch_size
                               * training_config.accumulation_steps,
            "contract_encoder_frozen": True,
        }
        with open(os.path.join(training_config.checkpoint_dir, "run_manifest.json"), "w") as _mf:
            _jm.dump(_manifest, _mf, indent=2)
        print(f"[RunManifest] {os.path.join(training_config.checkpoint_dir, 'run_manifest.json')}")

        # Load checkpoint (model weights + optimizer + scheduler)
        if resume_ckpt:
            engine.load_checkpoint(resume_ckpt)

            # Audit: resume events auto-logged (logs/resume_events.jsonl)
            import hashlib as _rh, datetime as _rd
            try:
                with open(resume_ckpt, "rb") as _rf:
                    _ck_sha = _rh.sha256(_rf.read()).hexdigest()
                _st = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
                _ev = {"time": _rd.datetime.now().isoformat(timespec="seconds"),
                       "run": os.path.basename(args.config),
                       "resume_from": resume_ckpt,
                       "checkpoint_sha256": _ck_sha,
                       "prev_epoch": _st.get("epoch"), "prev_step": _st.get("global_step"),
                       "validation": "pending (run scripts/resume_smoke_test.py)"}
                os.makedirs("logs", exist_ok=True)
                with open("logs/resume_events.jsonl", "a") as _ef:
                    _ef.write(__import__("json").dumps(_ev) + "\n")
                print(f"[ResumeEvent] logged → logs/resume_events.jsonl")
            except Exception as _e:
                print(f"[ResumeEvent] Failed to log (non-blocking): {_e}")

        # trainingloop
        # Determine whether contig-level aggregation is needed
        use_sliding_window = data_cfg.get("use_sliding_window", False)
        aggregate_contig = (data_mode == "contig" and use_sliding_window)

        _run_training_loop_dataloader(
            engine, train_loader, val_loader, test_loader,
            epochs, device, args.dry_run, training_config.checkpoint_dir,
            aggregate_contig=aggregate_contig,
            resume_epoch=resume_epoch,
        )


def _run_training_loop(engine, train_batches, val_batches, test_batches,
                        epochs, device, dry_run, ckpt_dir):
    """Training loop  —  uses an HGTBatch list (synthetic data mode)."""
    import torch

    train_history = []
    val_history = []
    best_val_auprc = 0.0
    optimal_threshold = 0.5
    patience_counter = 0
    patience = engine.config.early_stop_patience
    min_delta = 0.001  # Minimum AUC improvement

    for epoch in range(epochs):
        t0 = time.time()

        # ── Train ──
        train_metrics = engine.train_epoch(train_batches)
        train_time = time.time() - t0

        train_history.append(train_metrics)

        # ── Validation ──
        val_metrics = {}
        if val_batches:
            val_metrics = engine.eval_epoch(val_batches)
        val_time = time.time() - t0 - train_time

        lr = train_metrics.get("lr", 0)

        # outputtraininginformation
        print(f"\n[Epoch {epoch + 1}/{epochs}] "
              f"train_loss={train_metrics.get('total', 0):.4f}  "
              f"cls={train_metrics.get('cls', 0):.4f}  "
              f"bp={train_metrics.get('bp', 0):.4f}  "
              f"lr={lr:.2e}  "
              f"time={train_time:.1f}s+{val_time:.1f}s")

        if val_metrics:
            _print_val_metrics(val_metrics)

            # Best-model selection & early stopping (by AUC, must exceed min_delta)
            # M-10 fix: use AUPRC as the model-selection metric (more reliable on imbalanced data)
            val_auprc = val_metrics.get("auprc", val_metrics.get("auc", 0.0))
            if val_auprc > best_val_auprc + min_delta:
                best_val_auprc = val_auprc
                optimal_threshold = val_metrics.get("optimal_threshold", 0.5)
                patience_counter = 0
                # Sync engine state (for resume and checkpoint use)
                engine.best_metric = best_val_auprc
                engine.patience_counter = patience_counter
                engine.optimal_threshold = optimal_threshold
                if not dry_run:
                    os.makedirs(ckpt_dir, exist_ok=True)
                    engine.save_checkpoint(
                        os.path.join(ckpt_dir, "best_model.pt"),
                        metrics=val_metrics)
            else:
                patience_counter += 1
                engine.patience_counter = patience_counter
                if patience_counter >= patience:
                    print(f"\n[Early Stop] No validation AUPRC improvement for {patience} consecutive epochs "
                          f"(best={best_val_auprc:.4f})")
                    break

    # ── Final test evaluation ──
    if test_batches:
        print("\n" + "=" * 70)
        print("Final Test Evaluation")
        print("=" * 70)
        # Load best model, then evaluate
        best_path = os.path.join(ckpt_dir, "best_model.pt")
        if not dry_run and os.path.exists(best_path):
            engine.load_checkpoint(best_path)
            print(f"  [Test] Using best_model.pt (Val AUPRC={best_val_auprc:.4f}, "
                  f"optimal threshold={optimal_threshold:.2f})")
        test_metrics = engine.eval_epoch(test_batches, threshold=optimal_threshold,
                                         find_optimal_threshold=False)
        print(f"  [Test] loss={test_metrics.get('eval_loss', 0):.4f}  "
              f"acc={test_metrics.get('accuracy', 0):.4f}  "
              f"auc={test_metrics.get('auc', 0):.4f}  "
              f"auprc={test_metrics.get('auprc', 0):.4f}  "
              f"f1={test_metrics.get('f1', 0):.4f}  "
              f"precision={test_metrics.get('precision', 0):.4f}  "
              f"recall={test_metrics.get('recall', 0):.4f}")
        hard_auc = test_metrics.get("strat_hard_neg_auc")
        if hard_auc is not None:
            print(f"  [Test] stratified: hard_neg_auc={hard_auc:.4f}  "
                  f"easy_neg_auc={test_metrics.get('strat_easy_neg_auc', 0):.4f}  "
                  f"pos_recall={test_metrics.get('strat_pos_recall', 0):.4f}")
        ece = test_metrics.get("cal_ece")
        if ece is not None:
            print(f"  [Test] Calibration: ECE={ece:.4f}  "
                  f"T={test_metrics.get('cal_optimal_temperature', 1.0):.2f}  "
                  f"ECE_cal={test_metrics.get('cal_ece_calibrated', 0):.4f}")
        # Window-level stratified metrics (RFC-CONTIG-STRAT-001 change C)
        win_hard = test_metrics.get("window_strat_hard_neg_auc")
        if win_hard is not None:
            print(f"  [Test] window-levelstratified: hard_neg_auc={win_hard:.4f}  "
                  f"easy_neg_auc={test_metrics.get('window_strat_easy_neg_auc', 0):.4f}  "
                  f"pos_recall={test_metrics.get('window_strat_pos_recall', 0):.4f}  "
                  f"ECE={test_metrics.get('window_cal_ece', 0):.4f}")

    # ── Training summary ──
    _print_training_summary(train_history, val_history, epochs)

    # Save final model
    if not dry_run:
        os.makedirs(ckpt_dir, exist_ok=True)
        engine.save_checkpoint(os.path.join(ckpt_dir, "last_model.pt"))
        print(f"\n[Done] Model saved to {ckpt_dir}")


def _run_training_loop_dataloader(engine, train_loader, val_loader, test_loader,
                                   epochs, device, dry_run, ckpt_dir,
                                   aggregate_contig=False, resume_epoch=0):
    """Training loop  —  uses a DataLoader (real data mode)."""
    train_history = []
    val_history = []
    best_val_auprc = 0.0
    optimal_threshold = 0.5
    patience_counter = 0
    patience = engine.config.early_stop_patience
    min_delta = 0.001  # Minimum AUC improvement

    start_epoch = resume_epoch  # 0-indexed: epoch 0 = epoch 1
    if start_epoch > 0:
        # Restore early-stopping state (from the engine's checkpoint load result)
        best_val_auprc = getattr(engine, 'best_metric', 0.0)
        patience_counter = getattr(engine, 'patience_counter', 0)
        optimal_threshold = getattr(engine, 'optimal_threshold', 0.5)
        print(f"\n[Resume] Skipping {start_epoch} completed epochs, starting from epoch {start_epoch + 1}")
        print(f"         Resumed early-stopping state: best_auc={best_val_auprc:.4f}, "
              f"patience={patience_counter}/{patience}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # ── Train ──
        train_metrics = engine.train_epoch(train_loader)
        train_time = time.time() - t0

        train_history.append(train_metrics)

        # ── Validation ──
        val_metrics = {}
        if val_loader:
            val_metrics = engine.eval_epoch(
                val_loader, aggregate_contig=aggregate_contig)
        val_time = time.time() - t0 - train_time

        lr = train_metrics.get("lr", 0)

        # outputtraininginformation
        print(f"\n[Epoch {epoch + 1}/{epochs}] "
              f"train_loss={train_metrics.get('total', 0):.4f}  "
              f"cls={train_metrics.get('cls', 0):.4f}  "
              f"bp={train_metrics.get('bp', 0):.4f}  "
              f"lr={lr:.2e}  "
              f"time={train_time:.1f}s+{val_time:.1f}s")

        if val_metrics:
            val_tag = " [contig]" if aggregate_contig else ""
            _print_val_metrics(val_metrics, tag=val_tag)

            # Best-model selection & early stopping (by AUC, must exceed min_delta)
            # M-10 fix: use AUPRC as the model-selection metric (more reliable on imbalanced data)
            val_auprc = val_metrics.get("auprc", val_metrics.get("auc", 0.0))
            if val_auprc > best_val_auprc + min_delta:
                best_val_auprc = val_auprc
                optimal_threshold = val_metrics.get("optimal_threshold", 0.5)
                patience_counter = 0
                # Sync engine state (for resume and checkpoint use)
                engine.best_metric = best_val_auprc
                engine.patience_counter = patience_counter
                engine.optimal_threshold = optimal_threshold
                if not dry_run:
                    os.makedirs(ckpt_dir, exist_ok=True)
                    engine.save_checkpoint(
                        os.path.join(ckpt_dir, "best_model.pt"),
                        metrics=val_metrics)
            else:
                patience_counter += 1
                engine.patience_counter = patience_counter
                if patience_counter >= patience:
                    print(f"\n[Early Stop] No validation AUPRC improvement for {patience} consecutive epochs "
                          f"(best={best_val_auprc:.4f})")
                    break

        # v7: per-epoch metric persistence (long-run monitoring + learning dynamics; logging level)
        if not dry_run and val_metrics:
            import json as _json
            os.makedirs(ckpt_dir, exist_ok=True)
            _em_path = os.path.join(ckpt_dir, "epoch_metrics.json")
            _hist = []
            if os.path.exists(_em_path):
                try:
                    with open(_em_path) as _f:
                        _hist = _json.load(_f)
                except Exception:
                    _hist = []
            # METRIC-DIR-001: keep direction_accuracy None for null (not evaluated);
            # do not fill with a 0 default (0.0 would be confused with a true accuracy of 0)
            _dir_acc = val_metrics.get("direction_accuracy")
            _hist.append({
                "metric_schema": "v2",
                "epoch": epoch + 1,
                "train_loss": float(train_metrics.get("total", 0)),
                "lr": float(lr),
                "train_time_s": round(train_time, 1),
                "val_auprc": float(val_metrics.get("auprc", 0)),
                "val_auroc": float(val_metrics.get("auc", 0)),
                "optimal_threshold": float(val_metrics.get("optimal_threshold", 0.5)),
                "f1_at_optimal": float(val_metrics.get("f1_at_optimal", 0)),
                "recall_at_p90": float(val_metrics.get("recall_at_p90", 0)),
                "recall_at_p95": float(val_metrics.get("recall_at_p95", 0)),
                "cal_ece": float(val_metrics.get("cal_ece", 0) or 0),
                "hard_neg_auc": float(val_metrics.get("strat_hard_neg_auc", 0) or 0),
                "easy_neg_auc": float(val_metrics.get("strat_easy_neg_auc", 0) or 0),
                "pos_recall": float(val_metrics.get("strat_pos_recall", 0) or 0),
                "direction_accuracy": float(_dir_acc) if _dir_acc is not None else None,
                "direction_n": int(val_metrics.get("direction_n", 0) or 0),
                "direction_correct": int(val_metrics.get("direction_correct", 0) or 0),
                "direction_status": val_metrics.get("direction_status", "NOT_APPLICABLE"),
                "best_val_auprc": float(best_val_auprc),
            })
            with open(_em_path, "w") as _f:
                _json.dump(_hist, _f, indent=2)

        # Periodic checkpoint saving (v7: save_epochs takes effect + epoch_1 always saved —
        # needed for learning-dynamics analysis; previously hard-coded %5 so the save_epochs
        # config had no effect — bugfix-level change)
        if not dry_run and ((epoch + 1) % getattr(engine.config, "save_epochs", 5) == 0
                            or (epoch + 1) == 1):
            os.makedirs(ckpt_dir, exist_ok=True)
            engine.save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt"))

        # Resilience: overwrite last_model.pt every epoch (full state: model/optimizer/
        # scheduler/scaler/RNG/epoch/step) — crash-safe resume point, worst case loses 1 epoch.
        # Previously last was saved only at the end of training, so a crash could only
        # roll back to an epoch snapshot.
        if not dry_run:
            os.makedirs(ckpt_dir, exist_ok=True)
            engine.save_checkpoint(os.path.join(ckpt_dir, "last_model.pt"))

    # ── Save final model (before loading the best model, keep the last epoch's weights) ──
    if not dry_run:
        engine.save_checkpoint(os.path.join(ckpt_dir, "last_model.pt"))

    # ── Final test evaluation ──
    if test_loader:
        print("\n" + "=" * 70)
        print("Final Test Evaluation")
        print("=" * 70)
        # Load best model, then evaluate (does not affect the already-saved last_model.pt)
        best_path = os.path.join(ckpt_dir, "best_model.pt")
        if not dry_run and os.path.exists(best_path):
            engine.load_checkpoint(best_path)
            print(f"  [Test] Using best_model.pt (Val AUPRC={best_val_auprc:.4f}, "
                  f"optimal threshold={optimal_threshold:.2f})")
        test_metrics = engine.eval_epoch(
            test_loader, aggregate_contig=aggregate_contig,
            threshold=optimal_threshold, find_optimal_threshold=False)
        print(f"  [Test] loss={test_metrics.get('eval_loss', 0):.4f}  "
              f"acc={test_metrics.get('accuracy', 0):.4f}  "
              f"auc={test_metrics.get('auc', 0):.4f}  "
              f"auprc={test_metrics.get('auprc', 0):.4f}  "
              f"f1={test_metrics.get('f1', 0):.4f}  "
              f"precision={test_metrics.get('precision', 0):.4f}  "
              f"recall={test_metrics.get('recall', 0):.4f}")
        hard_auc = test_metrics.get("strat_hard_neg_auc")
        if hard_auc is not None:
            print(f"  [Test] stratified: hard_neg_auc={hard_auc:.4f}  "
                  f"easy_neg_auc={test_metrics.get('strat_easy_neg_auc', 0):.4f}  "
                  f"pos_recall={test_metrics.get('strat_pos_recall', 0):.4f}")
        ece = test_metrics.get("cal_ece")
        if ece is not None:
            print(f"  [Test] Calibration: ECE={ece:.4f}  "
                  f"T={test_metrics.get('cal_optimal_temperature', 1.0):.2f}  "
                  f"ECE_cal={test_metrics.get('cal_ece_calibrated', 0):.4f}")
        # Window-level stratified metrics (RFC-CONTIG-STRAT-001 change C)
        win_hard = test_metrics.get("window_strat_hard_neg_auc")
        if win_hard is not None:
            print(f"  [Test] window-levelstratified: hard_neg_auc={win_hard:.4f}  "
                  f"easy_neg_auc={test_metrics.get('window_strat_easy_neg_auc', 0):.4f}  "
                  f"pos_recall={test_metrics.get('window_strat_pos_recall', 0):.4f}  "
                  f"ECE={test_metrics.get('window_cal_ece', 0):.4f}")

    # ── Training summary ──
    _print_training_summary(train_history, val_history, epochs)

    print(f"\n[Done] Model saved to {ckpt_dir}")


def _print_val_metrics(val_metrics, tag=""):
    """Print validation metrics (including optimal threshold, stratified evaluation, calibration)."""
    print(f"  [Val{tag}] loss={val_metrics.get('eval_loss', 0):.4f}  "
          f"acc={val_metrics.get('accuracy', 0):.4f}  "
          f"auc={val_metrics.get('auc', 0):.4f}  "
          f"auprc={val_metrics.get('auprc', 0):.4f}  "
          f"f1={val_metrics.get('f1', 0):.4f}")
    opt_thr = val_metrics.get("optimal_threshold")
    if opt_thr is not None:
        print(f"  [Val{tag}] Optimal threshold={opt_thr:.2f}  "
              f"F1@optimal={val_metrics.get('f1_at_optimal', 0):.4f}  "
              f"P@optimal={val_metrics.get('precision_at_optimal', 0):.4f}  "
              f"R@optimal={val_metrics.get('recall_at_optimal', 0):.4f}")
    # stratifiedevaluation
    # METRIC-DIR-001: None-safe printing of the direction metric (None=not evaluated, do not default-fill 0)
    dir_acc = val_metrics.get("direction_accuracy")
    if dir_acc is not None:
        print(f"  [Val{tag}] Direction: acc={dir_acc:.4f}  "
              f"n={val_metrics.get('direction_n', 0)}  "
              f"status={val_metrics.get('direction_status', 'OK')}")
    hard_auc = val_metrics.get("strat_hard_neg_auc")
    easy_auc = val_metrics.get("strat_easy_neg_auc")
    pos_recall = val_metrics.get("strat_pos_recall")
    if hard_auc is not None:
        print(f"  [Val{tag}] stratified: hard_neg_auc={hard_auc:.4f}  "
              f"easy_neg_auc={easy_auc:.4f}  "
              f"pos_recall={pos_recall:.4f}")
    # Temperature-scaling calibration
    ece = val_metrics.get("cal_ece")
    if ece is not None:
        print(f"  [Val{tag}] Calibration: ECE={ece:.4f}  "
              f"T={val_metrics.get('cal_optimal_temperature', 1.0):.2f}  "
              f"ECE_cal={val_metrics.get('cal_ece_calibrated', 0):.4f}")
    # Window-level stratified metrics (RFC-CONTIG-STRAT-001 change C: contig aggregation path only, dual reporting)
    win_hard = val_metrics.get("window_strat_hard_neg_auc")
    if win_hard is not None:
        print(f"  [Val{tag}] window-levelstratified: hard_neg_auc={win_hard:.4f}  "
              f"easy_neg_auc={val_metrics.get('window_strat_easy_neg_auc', 0):.4f}  "
              f"pos_recall={val_metrics.get('window_strat_pos_recall', 0):.4f}  "
              f"ECE={val_metrics.get('window_cal_ece', 0):.4f}")


def _print_training_summary(train_history, val_history, epochs):
    """Print the training summary report."""
    if not train_history:
        return

    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)

    total_epochs = len(train_history)
    losses = [h.get("total", 0) for h in train_history]
    lrs = [h.get("lr", 0) for h in train_history]
    cls_losses = [h.get("cls", 0) for h in train_history]

    print(f"\n  Epochs: {total_epochs}")
    print(f"  Train Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  Loss Change: {losses[-1] - losses[0]:.4f} "
          f"({'↓' if losses[-1] < losses[0] else '↑'})")
    print(f"  Best Train Loss: {min(losses):.4f} (epoch {losses.index(min(losses)) + 1})")
    print(f"  LR Range: {min(lrs):.2e} ~ {max(lrs):.2e}")

    # convergenceinformation
    if total_epochs >= 3:
        last_3 = losses[-3:]
        delta = max(last_3) - min(last_3)
        converged = delta < 0.01
        print(f"  Convergence: {'converged' if converged else 'not converged'} "
              f"(last-3-epoch loss spread: {delta:.4f})")

    if val_history:
        val_losses = [h.get("eval_loss", 0) for h in val_history if h]
        if val_losses:
            print(f"  Val Loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
            val_accs = [h.get("accuracy", 0) for h in val_history if h]
            if val_accs:
                print(f"  Val Accuracy: {val_accs[0]:.4f} → {val_accs[-1]:.4f}")
                print(f"  Best Val Accuracy: {max(val_accs):.4f}")

    # Loss history table
    print(f"\n  {'Epoch':>6}  {'Train Loss':>10}  {'CLS Loss':>10}  {'LR':>12}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}")
    for i, h in enumerate(train_history):
        print(f"  {i + 1:>6}  {h.get('total', 0):>10.4f}  "
              f"{h.get('cls', 0):>10.4f}  {h.get('lr', 0):>12.2e}")


# ═══════════════════════════════════════════════════════════════════════════════ #
#  eval
# ═══════════════════════════════════════════════════════════════════════════════ #

def cmd_eval(args):
    """Evaluation entry point: load real test data and evaluate."""
    import torch
    from torch.utils.data import DataLoader
    from model.contracts import HGTConfig
    from model.api import build_model
    from loss.contracts import LossConfig
    from loss.api import build_loss
    from training_engine.contracts import TrainingConfig
    from training_engine.api import TrainingEngine
    from data_pipeline.api import load_splits, load_splits_sliding_window
    from data_pipeline.impl.collator import HGTCollator

    yaml_cfg = load_config(args.config)
    data_cfg = yaml_cfg.get("data", {})
    model_cfg = yaml_cfg.get("model", {})
    loss_cfg = yaml_cfg.get("loss", {})
    eval_cfg = yaml_cfg.get("evaluation", {})
    train_cfg = yaml_cfg.get("training", {})

    data_mode = args.mode or data_cfg.get("mode", "paired_end")
    data_dir = args.data_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = float(eval_cfg.get("threshold", 0.5))
    batch_size = args.batch_size or 16

    # ── Global seed control (BUG-026 fix) ──
    from utils.impl.seed import seed_all
    seed_all(train_cfg.get("seed", 42))

    print("=" * 60)
    print("DeepFluHGT  —  Evaluation")
    print(f"Device: {device}")
    print("=" * 60)

    # ── loaddata ──
    if not data_dir:
        print("[ERROR] --data_dir is required for evaluation")
        return

    use_sliding_window = (data_mode == "contig" and
                          data_cfg.get("use_sliding_window", False))

    if use_sliding_window:
        sw_window_size = data_cfg.get("sw_window_size", 1500)
        sw_overlap = data_cfg.get("sw_overlap", 300)
        sw_max_neg = data_cfg.get("sw_max_neg_windows", 2)
        sw_max_eval = data_cfg.get("sw_max_eval_windows", 10)

        model_config = HGTConfig(
            data_mode=data_mode,
            max_contig_length=sw_window_size,
            hidden_dim=model_cfg.get("hidden_dim", 768),
            local_hidden_dim=model_cfg.get("local_hidden_dim", 256),
            num_attention_heads=model_cfg.get("num_attention_heads", 12),
            dropout=model_cfg.get("dropout", 0.1),
            graph_top_k=model_cfg.get("graph_top_k", 5),
            bio_feature_dim=model_cfg.get("bio_feature_dim", 260),
            pretrained_nt_path=model_cfg.get("pretrained_nt_path", ""),
            pretrained_caduceus_path=model_cfg.get("pretrained_caduceus_path", ""),
            use_local_encoder=model_cfg.get("use_local_encoder", True),
            use_evo=model_cfg.get("use_evo", True),
            pretrained_evo_dim=model_cfg.get("pretrained_evo_dim", 4096),
            evo_hidden=model_cfg.get("evo_hidden", 512),
        )

        splits = load_splits_sliding_window(
            data_dir=data_dir,
            window_size=sw_window_size,
            overlap=sw_overlap,
            max_neg_windows=sw_max_neg,
            max_eval_windows=sw_max_eval,
            read_len=data_cfg.get("read_length", 150),
            evo_cache_path=data_cfg.get("evo_cache_path", ""),
            rc_aug_prob=data_cfg.get("rc_aug_prob", 0.0),
        )
    else:
        model_config = HGTConfig(
            data_mode=data_mode,
            hidden_dim=model_cfg.get("hidden_dim", 768),
            local_hidden_dim=model_cfg.get("local_hidden_dim", 256),
            num_attention_heads=model_cfg.get("num_attention_heads", 12),
            dropout=model_cfg.get("dropout", 0.1),
            graph_top_k=model_cfg.get("graph_top_k", 5),
            bio_feature_dim=model_cfg.get("bio_feature_dim", 260),
            pretrained_nt_path=model_cfg.get("pretrained_nt_path", ""),
            pretrained_caduceus_path=model_cfg.get("pretrained_caduceus_path", ""),
            use_local_encoder=model_cfg.get("use_local_encoder", True),
            use_evo=model_cfg.get("use_evo", True),
            pretrained_evo_dim=model_cfg.get("pretrained_evo_dim", 4096),
            evo_hidden=model_cfg.get("evo_hidden", 512),
        )
        max_length = data_cfg.get("max_contig_length", 10000)
        splits = load_splits(
            data_dir=data_dir,
            mode=data_mode,
            read_len=data_cfg.get("read_length", 150),
            evo_cache_path=data_cfg.get("evo_cache_path", ""),
            rc_aug_prob=data_cfg.get("rc_aug_prob", 0.0),
            max_length=max_length,
        )

    test_ds = splits.get("test")
    if not test_ds:
        print("[ERROR] No test split found in data_dir")
        return

    print(f"  Data: {data_dir} (test={len(test_ds)} samples)")

    collator = HGTCollator(pad_id=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, collate_fn=collator,
                             num_workers=10, pin_memory=True)

    # ── Build model ── (BUG-028 fix: pass YAML parameters through completely)
    model = build_model(model_config).to(device)
    # M-03 fix: cmd_eval LossConfig fully aligned with cmd_train
    loss_config = LossConfig(
        use_uncertainty_weight=loss_cfg.get("use_uncertainty_weight", False),
        region_pos_weight=loss_cfg.get("region_pos_weight", 10.0),
        cascade_gate_slope=loss_cfg.get("cascade_gate_slope", 10.0),
        use_focal_loss=loss_cfg.get("use_focal_loss", False),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        cls_weight=loss_cfg.get("cls_weight", 1.0),
        bp_weight=loss_cfg.get("bp_weight", 0.3),
        role_weight=loss_cfg.get("role_weight", 0.2),
        direction_weight=loss_cfg.get("direction_weight", 0.2),
        breakpoint_sigma=loss_cfg.get("breakpoint_sigma", 5.0),
        disentangle_weight=loss_cfg.get("disentangle_weight", 0.1),
        rc_consistency_weight=loss_cfg.get("rc_consistency_weight", 0.1),
        refiner_weight=loss_cfg.get("refiner_weight", 0.5),
        early_exit_weight=loss_cfg.get("early_exit_weight", 0.25),
        neg_bp_reg_weight=loss_cfg.get("neg_bp_reg_weight", 0.1),
    )
    loss_fn = build_loss(loss_config).to(device)
    training_config = TrainingConfig(
        learning_rate=train_cfg.get("learning_rate", 3e-5),
        batch_size=batch_size,
        use_amp=train_cfg.get("use_amp", True),
        amp_dtype=train_cfg.get("amp_dtype", "float16"),
        device=device,
        accumulation_steps=train_cfg.get("accumulation_steps", 1),
        gradient_clip_norm=train_cfg.get("gradient_clip_norm", 5.0),
    )

    # ── load checkpoint ──
    optimal_threshold = threshold
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if unexpected:
            print(f"  [Checkpoint] Ignoring {len(unexpected)} unexpected keys (rotary embedding buffers)")
        # Read the optimal threshold saved during training
        if "metrics" in state and "optimal_threshold" in state["metrics"]:
            optimal_threshold = state["metrics"]["optimal_threshold"]
            print(f"  [Checkpoint] Training-time optimal threshold: {optimal_threshold:.4f}")
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("[WARN] No checkpoint specified, using random weights")

    # ── evaluation ──
    engine = TrainingEngine(model, loss_fn, training_config)
    aggregate_contig = use_sliding_window

    test_metrics = engine.eval_epoch(
        test_loader,
        aggregate_contig=aggregate_contig,
        threshold=optimal_threshold,
        find_optimal_threshold=False,  # Use the training-time validation-set optimal threshold; do not re-search on the test set
        early_exit=not getattr(args, 'no_early_exit', False),  # F6: --no_early_exit reports the full forward-pass baseline (M-05)
    )

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"  Threshold   : {optimal_threshold:.4f}")
    print(f"  Optimal Thr : {optimal_threshold:.4f} (from checkpoint)")
    print(f"  Accuracy    : {test_metrics.get('accuracy', 0):.4f}")
    print(f"  Precision   : {test_metrics.get('precision', 0):.4f}")
    print(f"  Recall      : {test_metrics.get('recall', 0):.4f}")
    print(f"  F1-score    : {test_metrics.get('f1', 0):.4f}")
    print(f"  AUROC       : {test_metrics.get('auc', 0):.4f}")
    print(f"  AUPRC       : {test_metrics.get('auprc', 0):.4f}")
    print(f"  Specificity : {test_metrics.get('specificity', 0):.4f}")

    # stratifiedevaluation
    hard_auc = test_metrics.get("strat_hard_neg_auc")
    if hard_auc is not None:
        print(f"\n  --- Stratified ---")
        print(f"  hard_neg_auc: {hard_auc:.4f}")
        print(f"  easy_neg_auc: {test_metrics.get('strat_easy_neg_auc', 0):.4f}")
        print(f"  pos_recall  : {test_metrics.get('strat_pos_recall', 0):.4f}")

    # Calibration
    ece = test_metrics.get("cal_ece")
    if ece is not None:
        print(f"\n  --- Calibration ---")
        print(f"  ECE         : {ece:.4f}")
        print(f"  Optimal T   : {test_metrics.get('cal_optimal_temperature', 1.0):.2f}")
        print(f"  ECE_cal     : {test_metrics.get('cal_ece_calibrated', 0):.4f}")

    # Direction accuracy
    dir_acc = test_metrics.get("direction_accuracy")
    if dir_acc is not None:
        print(f"\n  Direction acc : {dir_acc:.4f}")

    print("=" * 50)

    # ── Visualization: ROC / PR curves ──
    all_probs = test_metrics.get("_all_probs")
    all_labels = test_metrics.get("_all_labels")
    if all_probs is not None and all_labels is not None:
        try:
            from evaluation.impl.visualization import plot_roc_curve, plot_pr_curve
            eval_out = args.output_dir or eval_cfg.get("output_dir", "./eval_output")
            os.makedirs(eval_out, exist_ok=True)

            roc_path = os.path.join(eval_out, "roc_curve.png")
            pr_path = os.path.join(eval_out, "pr_curve.png")

            plot_roc_curve(all_labels, all_probs, save_path=roc_path,
                           title="DeepFluHGT ROC (Test)")
            plot_pr_curve(all_labels, all_probs, save_path=pr_path,
                          title="DeepFluHGT PR (Test)")

            print(f"\n  ROC curve: {roc_path}")
            print(f"  PR  curve: {pr_path}")
        except Exception as e:
            print(f"\n  [WARN] Plot generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Checkpoint loading utilities
# ═══════════════════════════════════════════════════════════════════════════════ #

def _load_checkpoint_flexible(model, state_dict):
    """
    Flexibly load a checkpoint, handling pos_embedding size mismatches.

    Scenario: trained with max_contig_length=10000 (pos_emb=10064),
          predicting in SW mode with max_contig_length=1500 (pos_emb=2048), or vice versa.

    Strategy: for size-mismatched tensors, truncate or partially copy.
    """
    import torch

    model_sd = model.state_dict()
    mismatches = []

    for key, value in state_dict.items():
        if key not in model_sd:
            continue
        if value.shape == model_sd[key].shape:
            model_sd[key] = value
        else:
            # Size mismatch  —  truncate or partially copy
            model_shape = model_sd[key].shape
            if len(value.shape) == 2 and len(model_shape) == 2:
                # 2D: take the min number of rows
                n = min(value.shape[0], model_shape[0])
                model_sd[key][:n] = value[:n]
                mismatches.append((key, value.shape, model_shape))
            else:
                # Other dimension mismatch, skip
                mismatches.append((key, value.shape, model_shape))

    model.load_state_dict(model_sd, strict=False)

    if mismatches:
        for key, src, dst in mismatches:
            print(f"  [INFO] {key}: {list(src)} → {list(dst)} (adapted)")


# ═══════════════════════════════════════════════════════════════════════════════ #
#  Sliding-window prediction aggregation
# ═══════════════════════════════════════════════════════════════════════════════ #

def _aggregate_window_predictions(window_results, threshold=0.9):
    """
    Aggregate window-level predictions into contig-level predictions.

    Strategy:
      - Classification: max(cls_prob)  —  a contig is positive if any window detects HGT
      - Breakpoints: NMS merge (within 50 bp, keep the highest-confidence peak)
      - Output the aggregated result per contig (original sequence coordinates)
    """
    from collections import defaultdict

    # Group by original_seq_id (window seq_id is already the original_seq_id)
    contig_windows = defaultdict(list)
    for r in window_results:
        key = r.get("original_seq_id") or r["seq_id"]
        contig_windows[key].append(r)

    aggregated = []
    for contig_id, windows in contig_windows.items():
        # Classification: max pooling
        max_prob = max(r["hgt_probability"] for r in windows)
        pred_label = 1 if max_prob >= threshold else 0

        # Breakpoints: collect breakpoints from all windows, map to original coordinates
        all_peaks = []
        for r in windows:
            if r["breakpoint_start"] >= 0:
                all_peaks.append({
                    "start": r["breakpoint_start"],
                    "end": r["breakpoint_end"],
                    "score": r["bp_max_score"],
                })

        # NMS: merge within 50 bp
        merged_peaks = _nms_peaks(all_peaks, distance=50)

        # Build contig-level result
        contig_len = windows[0].get("contig_len", 0)
        result = {
            "seq_id": contig_id,
            "contig_len": contig_len,
            "predicted_label": pred_label,
            "hgt_probability": round(max_prob, 6),
            "n_windows": len(windows),
            "n_breakpoints": len(merged_peaks),
        }

        if merged_peaks:
            result["breakpoint_start"] = merged_peaks[0]["start"]
            result["breakpoint_end"] = merged_peaks[0]["end"]
            result["bp_max_score"] = merged_peaks[0]["score"]
            if len(merged_peaks) > 1:
                result["additional_breakpoints"] = merged_peaks[1:]
        else:
            result["breakpoint_start"] = -1
            result["breakpoint_end"] = -1
            result["bp_max_score"] = 0.0

        aggregated.append(result)

    return aggregated


def _nms_peaks(peaks, distance=50):
    """NMS merge of nearby breakpoint peaks."""
    if not peaks:
        return []

    # Sort by confidence, descending
    peaks = sorted(peaks, key=lambda p: p["score"], reverse=True)
    kept = []

    for peak in peaks:
        center = (peak["start"] + peak["end"]) // 2
        dominated = False
        for k in kept:
            k_center = (k["start"] + k["end"]) // 2
            if abs(center - k_center) < distance:
                dominated = True
                break
        if not dominated:
            kept.append(peak)

    return kept


# ═══════════════════════════════════════════════════════════════════════════════ #
#  predict
# ═══════════════════════════════════════════════════════════════════════════════ #

def cmd_predict(args):
    """
    Prediction entry point  —  run HGT prediction on unlabeled data.

    Supported input formats:
      - PE FASTQ compressed:     R1.fastq.gz + R2.fastq.gz
      - PE FASTQ uncompressed:   R1.fastq + R2.fastq
      - PE FASTA compressed:     R1.fasta.gz + R2.fasta.gz
      - PE FASTA uncompressed:   R1.fasta + R2.fasta
      - Contig FASTA:     .fasta / .fa / .fasta.gz / .fa.gz
      - Contig FASTQ:     .fastq / .fq / .fastq.gz / .fq.gz
    """
    import json
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from model.contracts import HGTConfig
    from model.api import build_model
    from data_pipeline.impl.input_reader import read_sequences
    from data_pipeline.impl.dataset import HGTDataset
    from data_pipeline.impl.collator import HGTCollator

    yaml_cfg = load_config(args.config)
    model_cfg = yaml_cfg.get("model", {})
    data_cfg = yaml_cfg.get("data", {})

    data_mode = args.mode or data_cfg.get("mode", "paired_end")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or 16
    threshold = args.threshold or 0.5
    n_mc = args.mc_forward or 0

    # ── Global seed control (BUG-026 fix) ──
    from utils.impl.seed import seed_all
    seed_all(42)

    print("=" * 70)
    print("DeepFluHGT  —  Prediction")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"Input: {args.input}")
    print(f"Mode: {data_mode}  Batch: {batch_size}  Threshold: {threshold}")
    if n_mc > 0:
        print(f"MC Dropout: {n_mc} forward passes")
    print("=" * 70)

    # ── Build model ──
    model_config = HGTConfig(
        data_mode=data_mode,
        pretrained_nt_path=model_cfg.get("pretrained_nt_path", ""),
        pretrained_caduceus_path=model_cfg.get("pretrained_caduceus_path", ""),
        use_local_encoder=model_cfg.get("use_local_encoder", True),
        use_evo=model_cfg.get("use_evo", True),
        pretrained_evo_dim=model_cfg.get("pretrained_evo_dim", 4096),
        evo_hidden=model_cfg.get("evo_hidden", 512),
    )
    model = build_model(model_config).to(device)

    # Load checkpoint (supports pos_embedding size mismatch)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        _load_checkpoint_flexible(model, state["model"])
        print(f"Loaded checkpoint: {args.checkpoint}")
        # Automatically use the optimal threshold found during training
        ckpt_metrics = state.get("metrics", {})
        ckpt_opt_thr = ckpt_metrics.get("optimal_threshold")
        if ckpt_opt_thr is not None and args.threshold is None:
            threshold = ckpt_opt_thr
            print(f"  Using training-time optimal threshold: {threshold:.2f} (from Val-set F1 maximization search)")
    else:
        print("[WARN] No checkpoint loaded  —  using random weights!")

    # ── readinput ──
    samples, detected_mode = read_sequences(
        args.input, mode=data_mode, read_len=args.read_len or 0)
    actual_mode = detected_mode if data_mode == "auto" else data_mode

    # ── Contig sliding-window mode ──
    use_sw = (actual_mode == "contig" and
              data_cfg.get("use_sliding_window", False))
    sw_window_size = data_cfg.get("sw_window_size", 1500)
    sw_overlap = data_cfg.get("sw_overlap", 300)
    sw_max_eval = data_cfg.get("sw_max_eval_windows", 10)

    if use_sw:
        from data_pipeline.impl.dataset import SlidingWindowContigDataset

        # No model rebuild needed — reuse the model from the loaded checkpoint
        # The SW dataset already truncates sequences to sw_window_size bp;
        # the model's pos_embedding table can be longer than the actual input, inference is unaffected
        print(f"  [Sliding Window] Using loaded model (no rebuild needed)")

        dataset = SlidingWindowContigDataset(
            samples=samples,
            window_size=sw_window_size,
            overlap=sw_overlap,
            is_training=False,
            max_neg_windows=9999,
            max_eval_windows=sw_max_eval,
            extract_bio=True,
        )
        print(f"  [Sliding Window] window={sw_window_size}bp, "
              f"overlap={sw_overlap}bp, windows={len(dataset)}")
    else:
        # Create Dataset (label=-1, no events)
        dataset = HGTDataset(
            samples=samples,
            mode=actual_mode,
            read_len=args.read_len or 150,
        )

    # create DataLoader
    collator = HGTCollator(pad_id=0)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collator,
                        num_workers=0)

    # ── inference ──
    model.eval()
    all_results = []

    with torch.no_grad():
        for batch in loader:
            # Move to device
            for fname in batch.__dataclass_fields__:
                val = getattr(batch, fname)
                if isinstance(val, torch.Tensor) and val.numel() > 0:
                    setattr(batch, fname, val.to(device))
                elif isinstance(val, torch.Tensor):
                    setattr(batch, fname, val.to(device))

            output = model(batch, early_exit=True)  # BUG-003 fix: enable early exit for inference

            # probability
            cls_probs = torch.sigmoid(output.cls_logits.squeeze(-1)).cpu().numpy()
            bp_heatmaps = output.bp_heatmaps.cpu().numpy()

            # optional: direction
            dir_probs = None
            if output.direction_logits is not None:
                dir_probs = torch.sigmoid(output.direction_logits.squeeze(-1)).cpu().numpy()

            # optional: role
            role_probs = None
            if output.role_logits is not None:
                role_probs = torch.sigmoid(output.role_logits.squeeze(-1)).cpu().numpy()

            # Parse each sample
            for i in range(len(cls_probs)):
                prob = float(cls_probs[i])
                pred_label = 1 if prob >= threshold else 0

                # Breakpoint localization
                heatmap = bp_heatmaps[i]
                attention_mask = batch.attention_mask[i].cpu().numpy()
                valid_len = int(attention_mask.sum())
                valid_heatmap = heatmap[:valid_len]

                bp_start, bp_end = -1, -1
                if pred_label == 1 and valid_heatmap.max() > 0:
                    # Find the peak-value region
                    bp_mask = (valid_heatmap > threshold * valid_heatmap.max()).astype(int)
                    bp_positions = np.where(bp_mask)[0]
                    if len(bp_positions) > 0:
                        # Subtract the CLS offset (position 0 = CLS)
                        bp_start = int(bp_positions[0]) - 1
                        bp_end = int(bp_positions[-1]) - 1

                # ── Sliding-window coordinate mapping ──
                if use_sw and i < len(batch.window_starts):
                    win_start = batch.window_starts[i]
                    win_end = batch.window_ends[i]
                    orig_len = batch.original_seq_lens[i]
                    orig_id = batch.original_seq_ids[i]

                    # Map in-window coordinates to original sequence coordinates
                    if bp_start >= 0:
                        bp_start = win_start + bp_start
                        bp_end = win_start + bp_end
                else:
                    win_start = 0
                    win_end = 0
                    orig_len = 0
                    orig_id = ""

                result = {
                    "seq_id": batch.seq_ids[i] if batch.seq_ids else f"sample_{len(all_results)}",
                    "predicted_label": pred_label,
                    "hgt_probability": round(prob, 6),
                    "breakpoint_start": bp_start,
                    "breakpoint_end": bp_end,
                    "bp_max_score": round(float(valid_heatmap.max()), 4) if valid_heatmap.max() > 0 else 0.0,
                }

                # Sliding-window extra information
                if use_sw:
                    result["contig_len"] = orig_len
                    result["window_start"] = win_start
                    result["window_end"] = win_end
                    if orig_id:
                        result["original_seq_id"] = orig_id

                if dir_probs is not None:
                    result["direction_prob"] = round(float(dir_probs[i]), 4)
                    result["direction"] = "donor→recipient" if dir_probs[i] >= 0.5 else "recipient→donor"

                if role_probs is not None:
                    result["role_prob"] = round(float(role_probs[i]), 4)

                all_results.append(result)

    # ── MC Dropout uncertainty (optional) ──
    if n_mc > 0:
        print(f"\n[MC Dropout] Running {n_mc} forward passes...")
        from utils.impl.uncertainty import mc_dropout_predict

        model.train()  # enable dropout
        uncertainties = []
        with torch.no_grad():
            for batch in loader:
                for fname in batch.__dataclass_fields__:
                    val = getattr(batch, fname)
                    if isinstance(val, torch.Tensor):
                        setattr(batch, fname, val.to(device))

                mc_result = mc_dropout_predict(model, batch, n_forward=n_mc, device=device)
                for j in range(len(mc_result["mean_probs"])):
                    uncertainties.append({
                        "mean_prob": round(float(mc_result["mean_probs"][j]), 6),
                        "std_prob": round(float(mc_result["std_probs"][j]), 6),
                        "entropy": round(float(mc_result["entropy"][j]), 4),
                    })

        # Merge uncertainty into results
        for i, unc in enumerate(uncertainties):
            if i < len(all_results):
                all_results[i]["mc_mean_prob"] = unc["mean_prob"]
                all_results[i]["mc_std_prob"] = unc["std_prob"]
                all_results[i]["mc_entropy"] = unc["entropy"]

    # ── Sliding-window aggregation: window-level → contig-level ──
    if use_sw:
        all_results = _aggregate_window_predictions(all_results, threshold=threshold)

    # ── Output results ──
    output_dir = args.output_dir or "./predict_output"
    os.makedirs(output_dir, exist_ok=True)

    # TSV output (BED-like format)
    tsv_path = os.path.join(output_dir, "predictions.tsv")
    with open(tsv_path, "w") as f:
        headers = ["seq_id", "predicted_label", "hgt_probability",
                   "breakpoint_start", "breakpoint_end", "bp_max_score"]
        if any("direction_prob" in r for r in all_results):
            headers += ["direction_prob", "direction"]
        if any("role_prob" in r for r in all_results):
            headers += ["role_prob"]
        if any("mc_entropy" in r for r in all_results):
            headers += ["mc_mean_prob", "mc_std_prob", "mc_entropy"]

        f.write("\t".join(headers) + "\n")
        for r in all_results:
            values = [str(r.get(h, "")) for h in headers]
            f.write("\t".join(values) + "\n")

    # JSON output
    json_path = os.path.join(output_dir, "predictions.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary statistics ──
    n_total = len(all_results)
    n_pos = sum(1 for r in all_results if r["predicted_label"] == 1)
    n_neg = n_total - n_pos
    avg_prob = np.mean([r["hgt_probability"] for r in all_results])

    print(f"\n{'='*70}")
    print("Prediction Summary")
    print(f"{'='*70}")
    print(f"  Total samples: {n_total}")
    print(f"  Predicted HGT+: {n_pos} ({n_pos/max(1,n_total)*100:.1f}%)")
    print(f"  Predicted HGT-: {n_neg} ({n_neg/max(1,n_total)*100:.1f}%)")
    print(f"  Avg probability: {avg_prob:.4f}")

    if n_pos > 0:
        pos_probs = [r["hgt_probability"] for r in all_results if r["predicted_label"] == 1]
        print(f"  HGT+ prob range: [{min(pos_probs):.4f}, {max(pos_probs):.4f}]")

        # Number of samples with predicted breakpoints
        n_bp = sum(1 for r in all_results if r["predicted_label"] == 1 and r["breakpoint_start"] >= 0)
        print(f"  With breakpoints: {n_bp}/{n_pos}")

    print(f"\n  Output: {output_dir}/")
    print(f"    predictions.tsv")
    print(f"    predictions.json")


# ═══════════════════════════════════════════════════════════════════════════════ #
#  test
# ═══════════════════════════════════════════════════════════════════════════════ #

def cmd_test(args):
    """Run module self-tests."""
    module = args.module
    if module == 'simulator':
        from simulator.api import test
        ok = test()
        sys.exit(0 if ok else 1)
    elif module == 'data_pipeline':
        from data_pipeline.api import test as dp_test
        ok = dp_test()
        sys.exit(0 if ok else 1)
    elif module == 'model':
        from model.api import test as model_test
        ok = model_test()
        sys.exit(0 if ok else 1)
    elif module == 'loss':
        from loss.api import test as loss_test
        ok = loss_test()
        sys.exit(0 if ok else 1)
    elif module == 'training_engine':
        from training_engine.api import test as te_test
        ok = te_test()
        sys.exit(0 if ok else 1)
    elif module == 'evaluation':
        from evaluation.api import test as eval_test
        ok = eval_test()
        sys.exit(0 if ok else 1)
    elif module == 'utils':
        from utils.api import test as utils_test
        ok = utils_test()
        sys.exit(0 if ok else 1)
    elif module == 'all':
        results = {}
        test_modules = [
            ('simulator', 'simulator.api'),
            ('data_pipeline', 'data_pipeline.api'),
            ('model', 'model.api'),
            ('loss', 'loss.api'),
            ('training_engine', 'training_engine.api'),
            ('evaluation', 'evaluation.api'),
            ('utils', 'utils.api'),
        ]
        for name, mod_path in test_modules:
            try:
                mod = __import__(mod_path, fromlist=['test'])
                results[name] = mod.test()
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                results[name] = False

        all_ok = all(results.values())
        for name, ok in results.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        sys.exit(0 if all_ok else 1)
    else:
        print(f"Module '{module}' not yet implemented.")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════ #
#  CLI Parser
# ═══════════════════════════════════════════════════════════════════════════════ #

def build_parser():
    parser = argparse.ArgumentParser(
        description="DeepFluHGT  —  HGT Detection Framework")
    sub = parser.add_subparsers(dest='command')

    # ── simulate ──
    sim = sub.add_parser('simulate', help='Generate simulated dataset')
    sim.add_argument('--mode', choices=['paired_end', 'contig'], required=True)
    sim.add_argument('--virus_fasta', required=True)
    sim.add_argument('--microbe_dir', required=True)
    sim.add_argument('--output_dir', default=None)
    sim.add_argument('--seed', type=int, default=None)
    sim.add_argument('--n_pos', type=int, default=None)
    sim.add_argument('--n_neg', type=int, default=None)
    sim.add_argument('--read_len', type=int, default=None)
    sim.add_argument('--contig_min', type=int, default=None)
    sim.add_argument('--contig_max', type=int, default=None)
    sim.add_argument('--error_rate', type=float, default=None)
    sim.add_argument('--gc_bias', action='store_true', default=None)
    sim.add_argument('--use_coverage', action='store_true', default=None)
    sim.add_argument('--processes', type=int, default=None)
    sim.add_argument('--no_split', action='store_true', default=None)
    sim.add_argument('--train_ratio', type=float, default=None)
    sim.add_argument('--val_ratio', type=float, default=None)
    sim.add_argument('--test_ratio', type=float, default=None)
    sim.add_argument('--insertion_prob', type=float, default=None)
    sim.add_argument('--replacement_prob', type=float, default=None)
    sim.add_argument('--deletion_prob', type=float, default=None)
    sim.add_argument('--multi_event', action='store_true', default=None)
    sim.add_argument('--multi_event_prob', type=float, default=None)
    sim.add_argument('--multi_event_max', type=int, default=None)
    sim.add_argument('--multi_event_min_gap', type=int, default=None)
    sim.add_argument('--sandwich', action='store_true', default=None)
    sim.add_argument('--sandwich_prob', type=float, default=None)
    sim.add_argument('--sandwich_min_gap', type=int, default=None)
    sim.add_argument('--virus_frag_min', type=int, default=None)
    sim.add_argument('--virus_frag_max', type=int, default=None)
    sim.add_argument('--virus_frag_per_genome', type=int, default=None)
    sim.add_argument('--multi_host', action='store_true', default=None)
    sim.add_argument('--multi_host_prob', type=float, default=None)
    sim.add_argument('--multi_host_max', type=int, default=None)
    sim.add_argument('--config', default='configs/default.yaml')
    # Biological realism enhancement parameters
    sim.add_argument('--seq_platform', choices=['illumina', 'nanopore', 'pacbio'], default=None)
    sim.add_argument('--coverage_distribution', choices=['lognormal', 'zipf'], default=None)
    sim.add_argument('--coverage_zipf_alpha', type=float, default=None)
    sim.add_argument('--batch_effect_sigma', type=float, default=None)
    sim.add_argument('--contamination_rate', type=float, default=None)
    sim.add_argument('--contaminant_dir', default=None)
    sim.add_argument('--assembly_artifact_rate', type=float, default=None)
    sim.add_argument('--label_noise', type=float, default=None)
    sim.add_argument('--boundary_shift_max', type=int, default=None)
    sim.add_argument('--evolutionary_distance', type=float, default=None)
    sim.add_argument('--host_mutation_rate', type=float, default=None)
    sim.add_argument('--open_set_virus', action='store_true', default=None)
    sim.add_argument('--gc_amelioration', action='store_true', default=None)
    sim.add_argument('--evo_distance_distribution', choices=['uniform', 'gamma'], default=None)
    sim.add_argument('--evo_distance_gamma_shape', type=float, default=None)
    sim.add_argument('--microhomology_len', type=int, default=None)
    sim.add_argument('--genome_isolation', action='store_true', default=None)

    # ── train ──
    train = sub.add_parser('train', help='Train model')
    train.add_argument('--mode', choices=['paired_end', 'contig'], default=None)
    train.add_argument('--data_dir', default=None,
                       help='Data directory (auto-detects train/val/test subdirs)')
    train.add_argument('--lr', type=float, default=None)
    train.add_argument('--epochs', type=int, default=None)
    train.add_argument('--batch_size', type=int, default=None)
    train.add_argument('--max_length', type=int, default=None,
                       help='Max sequence length for contig mode (default: from config)')
    train.add_argument('--ckpt_dir', default=None,
                       help='Checkpoint output directory (default: ./checkpoints)')
    train.add_argument('--config', default='configs/default.yaml')
    train.add_argument('--preflight', action='store_true',
                       help='Deployment preflight: build the model on a real data path and exit after '
                            'the contract assertion passes, no training (unlike --dry_run, which is a synthetic-data demo)')
    train.add_argument('--resume', default=None,
                       help='Resume from checkpoint: file path, directory (auto-detect), or None')
    train.add_argument('--dry_run', action='store_true',
                       help='Use synthetic data, no checkpoint save')

    # ── eval ──
    eval_parser = sub.add_parser('eval', help='Evaluate model on test data')
    eval_parser.add_argument('--mode', choices=['paired_end', 'contig'], default=None)
    eval_parser.add_argument('--data_dir', default=None,
                             help='Data directory with test split')
    eval_parser.add_argument('--checkpoint', default=None,
                             help='Path to model checkpoint')
    eval_parser.add_argument('--output_dir', default=None)
    eval_parser.add_argument('--config', default='configs/default.yaml')
    eval_parser.add_argument('--batch_size', type=int, default=None)
    eval_parser.add_argument('--dry_run', action='store_true')
    eval_parser.add_argument('--no_early_exit', action='store_true',
                             help='Disable early-exit inference short-circuit (M-05 baseline: report full forward-pass metrics, not the early-exit hybrid decision)')

    # ── test ──
    tst = sub.add_parser('test', help='Run module self-tests')
    tst.add_argument('module', choices=MODULES + ['all'],
                     default='all', nargs='?')

    # ── predict ──
    pred = sub.add_parser('predict', help='Predict HGT on new sequences')
    pred.add_argument('input',
                      help='Input path: directory, file, or R1,R2 (comma-separated)')
    pred.add_argument('--checkpoint', required=True,
                      help='Path to model checkpoint')
    pred.add_argument('--mode', choices=['paired_end', 'contig', 'auto'],
                      default='auto')
    pred.add_argument('--read_len', type=int, default=0,
                      help='PE read length (0=auto-detect)')
    pred.add_argument('--batch_size', type=int, default=None)
    pred.add_argument('--threshold', type=float, default=None,
                      help='Classification threshold (max-pooling contig: 0.9)')
    pred.add_argument('--mc_forward', type=int, default=0,
                      help='MC Dropout forward passes (0=disabled)')
    pred.add_argument('--output_dir', default=None)
    pred.add_argument('--config', default='configs/default.yaml')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'simulate':
        cmd_simulate(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'eval':
        cmd_eval(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'test':
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
