import argparse
import logging
from pathlib import Path
import time
import json
import torch
import numpy as np

from gog_fraud.pipelines.run_fraud_benchmark import (
    _load_config, _cfg_get, _nested_get, 
    _build_level1_trainer, _call_level1_trainer_fit,
    _build_level2_trainer, _call_level2_trainer_fit, _build_l2_dynamic_loader_builder,
    _best_effort_save_table
)
from gog_fraud.evaluation.benchmark import BenchmarkTable, evaluate_benchmark

from gog_fraud.data.io.streaming_dataset import StreamingDataset
from gog_fraud.models.extensions.mc.config import MCDropoutConfig
from gog_fraud.models.extensions.mc.mc_dropout import MCDropoutEstimator
from gog_fraud.evaluation.mc_metrics import (
    calc_calibration_ece, calc_uncertainty_correlation, run_selective_prediction
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)

def evaluate_streaming(model, dataset, cfg, setting, train_g, stream_g, stage="l1", l1_model=None):
    if stage == "l1":
        from gog_fraud.training.loops.level1 import _prepare_level1_loader
        loader = _prepare_level1_loader(stream_g, split_name="test", batch_size=1, shuffle=False, label_dict=dataset.labels)
    else:
        from gog_fraud.training.loops.level2 import _prepare_level2_loader
        loader_builder = _build_l2_dynamic_loader_builder(l1_model, cfg)
        loader = _prepare_level2_loader(
            stream_g, 
            split="test", 
            batch_size=1, 
            shuffle=False, 
            label_dict=dataset.labels, 
            global_graph=dataset.global_graph,
            loader_builder=loader_builder
        )
        
    mode = cfg.get("streaming", {}).get("mode", "virtual")
    duration = cfg.get("streaming", {}).get("compressed_duration_sec", 3600)
    
    mc_cfg = MCDropoutConfig(mc_samples=cfg.get("mc_samples", 8), dropout_p=cfg.get("dropout_p", 0.1), execution_mode="sequential")
    estimator = MCDropoutEstimator(mc_cfg)
    
    device = next(model.parameters()).device
    model.eval()
    
    all_y = []
    all_scores = []
    all_unc = []
    latencies = []
    
    start_sim_time = time.time()
    total_graphs = len(stream_g)
    
    # Simple tick distribution
    tick_delay = duration / max(total_graphs, 1)
    if mode == "virtual":
        log.info(f"[Streaming Replay] Starting Virtual mode - 1 {stage} Graph per tick ~ {tick_delay:.2f}s virtual gap.")
    else:
        log.info(f"[Streaming Replay] Starting Wallclock mode - Waiting {tick_delay:.2f}s per graph.")
        
    for i, batch in enumerate(loader):
        processing_start = time.time()
        
        if stage == "l1":
            try: batch = batch.to(device)
            except: pass
            y = batch.y
        else:
            try: batch = batch.to(device)
            except: pass
            y = getattr(batch, "level1_label", getattr(batch, "y", None))
            if y is not None and y.size(0) == 1 and batch.x.size(0) > 1:
                y = y.expand(batch.x.size(0), 1)
                
        if y is None: 
            processing_time = time.time() - processing_start
            continue
            
        mc_out = estimator.estimate(model, batch)
        
        all_y.append(y.detach().cpu().view(-1))
        all_scores.append(mc_out.mean_score.detach().cpu().view(-1))
        all_unc.append(mc_out.uncertainty.detach().cpu().view(-1))
        
        processing_time = time.time() - processing_start
        latencies.append(processing_time)
        
        if mode == "wallclock":
            time.sleep(max(0, tick_delay - processing_time))
            
        if (i + 1) % 50 == 0:
            avg_lat = sum(latencies[-50:]) / 50 * 1000
            log.info(f"[{i+1}/{total_graphs}] Latency: {avg_lat:.2f}ms. Uncertainty avg: {float(all_unc[-1].mean()):.3f}")
            
    if not all_y:
        return None, None, None, None

    yt = torch.cat(all_y, dim=0).numpy()
    ys = torch.cat(all_scores, dim=0).numpy()
    unc = torch.cat(all_unc, dim=0).numpy()
    
    return yt, ys, unc, latencies

def augment_streaming_dataset(cfg, train_g, stream_g):
    from gog_fraud.adapters.legacy_adapter import LegacyAdapterConfig, LegacyBatchRunner
    import torch
    import logging
    log = logging.getLogger(__name__)
    
    legacy_cfg = _cfg_get(cfg, "legacy", {}) or {}
    model_names = _cfg_get(legacy_cfg, "models", ["DOMINANT", "DONE", "GAE", "AnomalyDAE", "CoLA"])
    
    dataset_cfg = _cfg_get(cfg, "dataset", {}) or {}
    chain_name = dataset_cfg.get("chain", "polygon").lower()

    base_adapter_cfg = LegacyAdapterConfig(
        agg_method      = _cfg_get(legacy_cfg, "agg_method", "max"),
        topk            = int(_cfg_get(legacy_cfg, "topk", 3)),
        normalize_score = bool(_cfg_get(legacy_cfg, "normalize_score", True)),
        gpu             = int(_cfg_get(legacy_cfg, "gpu", 0)),
        hid_dim         = int(_cfg_get(legacy_cfg, "hid_dim", 16)),
        num_layers      = int(_cfg_get(legacy_cfg, "num_layers", 2)),
        epoch           = int(_cfg_get(legacy_cfg, "epoch", 50)),
        lr              = float(_cfg_get(legacy_cfg, "lr", 0.003)),
        use_best_params = True,
        chain           = chain_name
    )
    
    all_graphs = train_g + stream_g
    if not all_graphs:
        return train_g, stream_g
        
    log.info(f"[Augment] Running legacy batch runner on {len(all_graphs)} graphs for models {model_names}")
    batch = LegacyBatchRunner(
        config=base_adapter_cfg,
        detector_overrides=base_adapter_cfg.detector_overrides,
        score_reduce=base_adapter_cfg.score_reduce,
        progress_every=base_adapter_cfg.progress_every
    )
    
    all_scores = batch.run_many(model_names=model_names, graphs=all_graphs)
    
    contract_to_scores = {}
    for g in all_graphs:
        cid = getattr(g, "contract_id", None)
        if cid is not None:
            contract_to_scores[cid] = [0.0] * len(model_names)
            
    for i, model_name in enumerate(model_names):
        if model_name in all_scores:
            for r in all_scores[model_name].records:
                if r.contract_id in contract_to_scores:
                    contract_to_scores[r.contract_id][i] = float(r.score)
                    
    for g in all_graphs:
        cid = getattr(g, "contract_id", None)
        if cid in contract_to_scores:
            scores = contract_to_scores[cid]
            data = getattr(g, "graph", g)
            if hasattr(data, "x") and data.x is not None:
                score_tensor = torch.tensor(scores, dtype=torch.float, device=data.x.device)
                score_tensor = score_tensor.expand(data.x.size(0), -1)
                data.x = torch.cat([data.x, score_tensor], dim=-1)
                
    log.info(f"[Augment] Appended {len(model_names)} legacy features to node features.")
    return train_g, stream_g

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output", required=False, type=str, default=None)
    parser.add_argument("--stages", required=False, type=str, default="l1,l1_l2")
    parser.add_argument("--chain", required=False, type=str, default=None)
    parser.add_argument("--max_samples", required=False, type=int, default=None)
    args = parser.parse_args()

    active_stages = [s.strip().lower() for s in args.stages.split(",")]

    cfg = _load_config(args.config)
    
    # Chain override
    if args.chain:
        if "dataset" not in cfg: cfg["dataset"] = {}
        cfg["dataset"]["chain"] = args.chain
        log.info(f"[Streaming Replay] Chain override: {args.chain}")

    setting = str(_cfg_get(cfg, "setting", "strict"))
    output_dir = Path(args.output or _cfg_get(cfg, "output_dir", "results/streaming_replay"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chain = cfg.get("dataset", {}).get("chain", 'polygon')
    
    log.info("=" * 50)
    log.info("[Streaming Dataset] Initialization")
    dataset = StreamingDataset.from_config(cfg)
    tx_root = cfg.get("dataset", {}).get("transactions_root", "../_data/dataset/transactions")
    
    # (0) Dynamic in_dim inference from dataset
    in_dim_inferred = 32
    if hasattr(dataset, "train_graphs") and len(dataset.train_graphs) > 0:
        first_item = dataset.train_graphs[0]
        data_obj = getattr(first_item, "graph", first_item)
        if hasattr(data_obj, "x") and data_obj.x is not None:
            in_dim_inferred = data_obj.x.size(-1)
            log.info(f"[Streaming Replay] Inferred dynamic in_dim: {in_dim_inferred} from dataset")

    # Actually perform the streaming split
    train_g, stream_g = dataset.prepare_streaming_splits(tx_root, train_ratio=0.8)

    # Optional subsetting
    if args.max_samples and len(stream_g) > args.max_samples:
        log.info(f"[Streaming Replay] Subsetting stream_g to {args.max_samples} samples.")
        stream_g = stream_g[:args.max_samples]

    aug_stages = ["l1_legacy_aug", "l1_l2_legacy_aug"]
    is_aug = any(s in active_stages for s in aug_stages)
    if is_aug:
        train_g, stream_g = augment_streaming_dataset(cfg, train_g, stream_g)
        legacy_models = _cfg_get(cfg.get("legacy", {}), "models", ["DOMINANT", "DONE", "GAE", "AnomalyDAE", "CoLA"])
        in_dim_inferred += len(legacy_models)
        log.info(f"[Streaming Replay] Updated in_dim to {in_dim_inferred} after legacy augmentation.")

    if "level1" not in cfg: cfg["level1"] = {}
    cfg["level1"]["in_dim"] = in_dim_inferred if in_dim_inferred != 32 else cfg["level1"].get("in_dim", 32)
    log.info(f"[Streaming Replay] Final Level1 Input Dimension: {cfg['level1']['in_dim']}")
    
    if not stream_g:
        log.error("[Streaming Replay] No samples found in stream_g. Check dataset or --chain.")
        return

    # Document Subset Range
    sample_ids = [getattr(g, 'contract_id', str(i)) for i, g in enumerate(stream_g)]
    log.info(f"[Streaming Replay] Replay Subset: {len(stream_g)} contracts.")
    if hasattr(dataset, 'contract_timestamps'):
        sub_ts = [dataset.contract_timestamps[cid] for cid in dataset.labels.keys() if cid in sample_ids]
        if sub_ts:
            log.info(f"[Streaming Replay] Time Range: {min(sub_ts)} - {max(sub_ts)}")

    l1_cache_path = output_dir / f"l1_model_weights_{chain}{'_aug' if is_aug else ''}.pt"
    l1_model = None
    
    if "l1" in active_stages or "l1_legacy_aug" in active_stages:
        is_l1_aug = "l1_legacy_aug" in active_stages
        stage_name = "Stage 1: Level 1 + StreamMC (Augmented)" if is_l1_aug else "Stage 1: Level 1 + StreamMC"
        log.info("=" * 50)
        log.info(f"(A) {stage_name} - Warmup on Historical Context")
    trainer = _build_level1_trainer(cfg)
    if l1_cache_path.exists():
        trainer.model.load_state_dict(torch.load(l1_cache_path))
        log.info("L1 Historical Warmup weights loaded from cache.")
    else:
        _call_level1_trainer_fit(trainer, train_g, train_g[:100] if train_g else [], dataset.labels, cfg)
        torch.save(trainer.model.state_dict(), l1_cache_path)
    
    l1_model = trainer.model

    table = BenchmarkTable()
    
    if "l1" in active_stages or "l1_legacy_aug" in active_stages:
        import time
        _t0_l1 = time.perf_counter()
        log.info(f"(B) Streaming Replay Simulation Phase - {stage_name}")
        
        import psutil
        process_stream = psutil.Process()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        max_nodes_stream = max(
            (g.graph.num_nodes if hasattr(g, "graph") else g.num_nodes)
            for g in stream_g
        ) if stream_g else 0
        
        yt, ys, unc, latencies = evaluate_streaming(l1_model, dataset, cfg, setting, train_g, stream_g, stage="l1")
        
        if yt is not None:
            peak_ram_stream = process_stream.memory_info().rss / (1024 * 1024)
            peak_gpu_stream = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            # Triage Analysis
            from gog_fraud.evaluation.mc_metrics import calc_fixed_budget_utility
            budget_50 = calc_fixed_budget_utility(yt, ys, unc, budget=min(50, len(yt)))
            budget_1pct = calc_fixed_budget_utility(yt, ys, unc, budget=0.01)
            budget_5pct = calc_fixed_budget_utility(yt, ys, unc, budget=0.05)
            
            log.info(f"Triage Utility (Top 50) -> Gain: {budget_50['precision_gain']:.4f} (Cov: {budget_50['coverage']:.2%})")
            log.info(f"Triage Utility (Top 1%) -> Gain: {budget_1pct['precision_gain']:.4f} (Cov: {budget_1pct['coverage']:.2%})")
            log.info(f"Triage Utility (Top 5%) -> Gain: {budget_5pct['precision_gain']:.4f} (Cov: {budget_5pct['coverage']:.2%})")
            
            res = evaluate_benchmark(
                y_true=yt, y_score=ys, model_name="L1-StreamMC-Aug" if is_l1_aug else "L1-StreamMC", setting=setting,
                max_nodes_processed=max_nodes_stream, peak_ram_mb=peak_ram_stream, peak_gpu_mb=peak_gpu_stream,
                elapsed_sec=time.perf_counter() - _t0_l1,
            )
            # Avoid dict assignment to dataclass. Just log the gain.
            log.info(f"Streaming Result for {chain}: ROC-AUC={res.roc_auc:.4f}, PR-AUC={res.pr_auc:.4f}")
            
            table.add(res)
            table.save_csv(output_dir / f"streaming_results_{chain}.csv")
            
            if latencies:
                avg_lat = np.mean(latencies) * 1000
                p95 = np.percentile(latencies, 95) * 1000
                p99 = np.percentile(latencies, 99) * 1000
                throughput = 1.0 / np.mean(latencies)
                
                vram_mb = 0
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                    
                log.info(f"--- Latency (ms) | Avg: {avg_lat:.2f} | P95: {p95:.2f} | P99: {p99:.2f}")
                log.info(f"--- Throughput: {throughput:.2f} GPS | Peak VRAM: {vram_mb:.1f} MB")

    if ("l1_l2" in active_stages or "l1_l2_legacy_aug" in active_stages) and l1_model is not None:
        import time
        _t0_l1l2 = time.perf_counter()
        is_l2_aug = "l1_l2_legacy_aug" in active_stages
        stage_name = "Level 1 + Level 2 + StreamMC (Augmented)" if is_l2_aug else "Level 1 + Level 2 + StreamMC"
        log.info(f"(C) Streaming Replay Simulation Phase - {stage_name}")
        
        l2_cache_path = output_dir / f"l2_model_weights_{chain}{'_aug' if is_l2_aug else ''}.pt"
        l2_trainer = _build_level2_trainer(cfg, l1_model)
        
        if l2_cache_path.exists():
            l2_trainer.model.load_state_dict(torch.load(l2_cache_path))
            log.info("L2 Historical Warmup weights loaded from cache.")
        else:
            log.info("Training L2 on Historical Context...")
            _call_level2_trainer_fit(
                trainer=l2_trainer, l1_model=l1_model, cfg=cfg,
                train_ids=train_g, valid_ids=train_g[:100] if train_g else [], labels=dataset.labels,
                global_graph=dataset.global_graph,
                loader_builder=_build_l2_dynamic_loader_builder(l1_model, cfg)
            )
            torch.save(l2_trainer.model.state_dict(), l2_cache_path)

        import psutil
        process_stream = psutil.Process()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        max_nodes_stream = max(
            (g.num_nodes if hasattr(g, "num_nodes") else g.graph.num_nodes)
            for g in stream_g
        ) if stream_g else 0
        
        yt, ys, unc, latencies = evaluate_streaming(l2_trainer.model, dataset, cfg, setting, train_g, stream_g, stage="l2", l1_model=l1_model)
        
        if yt is not None:
            peak_ram_stream = process_stream.memory_info().rss / (1024 * 1024)
            peak_gpu_stream = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            from gog_fraud.evaluation.mc_metrics import calc_fixed_budget_utility
            budget_50 = calc_fixed_budget_utility(yt, ys, unc, budget=min(50, len(yt)))
            budget_1pct = calc_fixed_budget_utility(yt, ys, unc, budget=0.01)
            budget_5pct = calc_fixed_budget_utility(yt, ys, unc, budget=0.05)
            
            log.info(f"Triage Utility (Top 50) -> Gain: {budget_50['precision_gain']:.4f} (Cov: {budget_50['coverage']:.2%})")
            log.info(f"Triage Utility (Top 1%) -> Gain: {budget_1pct['precision_gain']:.4f} (Cov: {budget_1pct['coverage']:.2%})")
            log.info(f"Triage Utility (Top 5%) -> Gain: {budget_5pct['precision_gain']:.4f} (Cov: {budget_5pct['coverage']:.2%})")
            
            res = evaluate_benchmark(
                y_true=yt, y_score=ys, model_name="L1+L2-StreamMC-Aug" if is_l2_aug else "L1+L2-StreamMC", setting=setting,
                max_nodes_processed=max_nodes_stream, peak_ram_mb=peak_ram_stream, peak_gpu_mb=peak_gpu_stream,
                elapsed_sec=time.perf_counter() - _t0_l1l2,
            )
            log.info(f"Streaming Result for {chain} (L1+L2): ROC-AUC={res.roc_auc:.4f}, PR-AUC={res.pr_auc:.4f}")
            
            table.add(res)
            table.save_csv(output_dir / f"streaming_results_{chain}.csv")
            
            if latencies:
                avg_lat = np.mean(latencies) * 1000
                p95 = np.percentile(latencies, 95) * 1000
                p99 = np.percentile(latencies, 99) * 1000
                throughput = 1.0 / np.mean(latencies)
                
                vram_mb = 0
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                    
                log.info(f"--- Latency (ms) | Avg: {avg_lat:.2f} | P95: {p95:.2f} | P99: {p99:.2f}")
                log.info(f"--- Throughput: {throughput:.2f} GPS | Peak VRAM: {vram_mb:.1f} MB")

        table.print_summary()

if __name__ == "__main__":
    main()
