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
            log.info(f"[{i+1}/{total_graphs}] Latency: {avg_lat:.2f}ms. Uncertainty avg: {float(all_unc[-1]):.3f}")
            
    if not all_y:
        return None, None, None, None

    yt = torch.cat(all_y, dim=0).numpy()
    ys = torch.cat(all_scores, dim=0).numpy()
    unc = torch.cat(all_unc, dim=0).numpy()
    
    return yt, ys, unc, latencies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output", required=False, type=str, default=None)
    parser.add_argument("--chain", required=False, type=str, default=None)
    parser.add_argument("--max_samples", required=False, type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    
    # Chain override
    if args.chain:
        if "dataset" not in cfg: cfg["dataset"] = {}
        cfg["dataset"]["chain"] = args.chain
        log.info(f"[Streaming Replay] Chain override: {args.chain}")

    setting = str(_cfg_get(cfg, "setting", "strict"))
    output_dir = Path(args.output or _cfg_get(cfg, "output", "results/streaming_replay"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chain = cfg.get("dataset", {}).get("chain", 'polygon')
    
    log.info("=" * 50)
    log.info("[Streaming Dataset] Initialization")
    dataset = StreamingDataset.from_config(cfg)
    tx_root = cfg.get("dataset", {}).get("transactions_root", "../_data/dataset/transactions")
    
    # Actually perform the read
    train_g, stream_g = dataset.prepare_streaming_splits(tx_root, train_ratio=0.8)
    
    # Optional subsetting
    if args.max_samples and len(stream_g) > args.max_samples:
        log.info(f"[Streaming Replay] Subsetting stream_g to {args.max_samples} samples.")
        stream_g = stream_g[:args.max_samples]
    
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

    l1_cache_path = output_dir / f"l1_model_weights_{chain}.pt"
    
    log.info("(A) Level 1 Warmup on Historical Context")
    trainer = _build_level1_trainer(cfg)
    if l1_cache_path.exists():
        trainer.model.load_state_dict(torch.load(l1_cache_path))
        log.info("L1 Historical Warmup weights loaded from cache.")
    else:
        _call_level1_trainer_fit(trainer, train_g, train_g[:100] if train_g else [], dataset.labels, cfg)
        torch.save(trainer.model.state_dict(), l1_cache_path)
    
    l1_model = trainer.model

    log.info("(B) Streaming Replay Simulation Phase")
    table = BenchmarkTable()
    
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
            y_true=yt, y_score=ys, model_name="L1-StreamMC", setting=setting,
            max_nodes_processed=max_nodes_stream, peak_ram_mb=peak_ram_stream, peak_gpu_mb=peak_gpu_stream
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

        table.print_summary()

if __name__ == "__main__":
    main()
