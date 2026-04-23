import argparse
import logging
from pathlib import Path
import json
import torch
import numpy as np

from gog_fraud.pipelines.run_fraud_benchmark import (
    _load_config, _cfg_get, _nested_get, _build_dataset_from_cfg, 
    _get_split_graphs, _build_level1_trainer, _call_level1_trainer_fit,
    _build_level2_trainer, _call_level2_trainer_fit, _build_l2_dynamic_loader_builder,
    run_legacy_baselines, _best_effort_save_table
)
from gog_fraud.evaluation.benchmark import BenchmarkTable, evaluate_benchmark

from gog_fraud.models.extensions.mc.config import MCDropoutConfig
from gog_fraud.models.extensions.mc.mc_dropout import MCDropoutEstimator
from gog_fraud.evaluation.mc_metrics import (
    calc_calibration_ece, calc_uncertainty_correlation, 
    run_selective_prediction, calc_bootstrap_ci, calc_fixed_budget_utility
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)

def evaluate_with_mc(model, dataset, cfg, setting, stage="l1", l1_model=None):
    # Prepare data loaders and targets similarly
    if stage == "l1":
        from gog_fraud.training.loops.level1 import _prepare_level1_loader
        train_g, valid_g, test_g = _get_split_graphs(dataset, cfg, setting)
        loader = _prepare_level1_loader(test_g, split_name="test", batch_size=128, shuffle=False, label_dict=dataset.labels)
    else:
        from gog_fraud.training.loops.level2 import _prepare_level2_loader
        train_g, valid_g, test_g = _get_split_graphs(dataset, cfg, setting)
        loader_builder = _build_l2_dynamic_loader_builder(l1_model, cfg)
        loader = _prepare_level2_loader(
            test_g, 
            split="test", 
            batch_size=128, 
            shuffle=False, 
            label_dict=dataset.labels, 
            global_graph=dataset.global_graph,
            loader_builder=loader_builder
        )
        
    if loader is None:
        log.warning(f"[MC Benchmark] Empty loader for {stage}.")
        return None, None, None, None
        
    mc_cfg = MCDropoutConfig(mc_samples=cfg.get("mc_samples", 8), dropout_p=cfg.get("dropout_p", 0.1), execution_mode="sequential")
    estimator = MCDropoutEstimator(mc_cfg)
    
    device = next(model.parameters()).device
    model.eval()
    
    all_y = []
    all_scores = []
    all_unc = []
    
    for batch in loader:
        if stage == "l1":
            try: batch = batch.to(device)
            except: pass
            y = batch.y
        else:
            try: batch = batch.to(device)
            except: pass
            y = getattr(batch, "level1_label", getattr(batch, "y", None))
            if y.size(0) == 1 and batch.x.size(0) > 1:
                y = y.expand(batch.x.size(0), 1)
                
        if y is None: continue
            
        mc_out = estimator.estimate(model, batch)
        
        all_y.append(y.detach().cpu().view(-1))
        all_scores.append(mc_out.mean_score.detach().cpu().view(-1))
        all_unc.append(mc_out.uncertainty.detach().cpu().view(-1))
        
    if not all_y:
        return None, None, None, None
        
    yt = torch.cat(all_y, dim=0).numpy()
    ys = torch.cat(all_scores, dim=0).numpy()
    unc = torch.cat(all_unc, dim=0).numpy()
    
    # Validation against dimension parity
    min_size = min(len(yt), len(ys))
    
    return yt[:min_size], ys[:min_size], unc[:min_size], mc_cfg


def augment_dataset_with_legacy_scores(dataset, cfg, setting):
    """Run 5 legacy anomaly detectors on all graphs and append their scores to node features."""
    from gog_fraud.adapters.legacy_adapter import LegacyAdapterConfig, LegacyBatchRunner
    import torch

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

    all_graphs = dataset.train_graphs + dataset.valid_graphs + dataset.test_graphs
    if not all_graphs:
        return dataset

    log.info(f"[Augment] Running legacy batch runner on {len(all_graphs)} graphs for models {model_names}")
    batch = LegacyBatchRunner(
        config=base_adapter_cfg,
        detector_overrides=base_adapter_cfg.detector_overrides,
        score_reduce=base_adapter_cfg.score_reduce,
        progress_every=base_adapter_cfg.progress_every
    )

    all_scores = batch.run_many(model_names=model_names, graphs=all_graphs)

    # Build contract_id -> [score_model0, score_model1, ...] mapping
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

    # Concatenate scores to node features for every graph
    for g in all_graphs:
        cid = getattr(g, "contract_id", None)
        if cid in contract_to_scores:
            scores = contract_to_scores[cid]
            data = getattr(g, "graph", g)
            if hasattr(data, "x") and data.x is not None:
                score_tensor = torch.tensor(scores, dtype=torch.float, device=data.x.device)
                score_tensor = score_tensor.expand(data.x.size(0), -1)
                data.x = torch.cat([data.x, score_tensor], dim=-1)

    log.info(f"[Augment] Appended {len(model_names)} legacy score features to node features.")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output", required=False, type=str, default=None)
    parser.add_argument("--stages", required=False, type=str, default="l1,l1_l2")
    parser.add_argument("--chain", required=False, type=str, default=None)
    parser.add_argument("--bootstrap", action="store_true", help="Enable bootstrapping for CI")
    parser.add_argument("--max_samples", required=False, type=int, default=None)
    args = parser.parse_args()

    active_stages = [s.strip().lower() for s in args.stages.split(",")]
    cfg = _load_config(args.config)
    
    # Chain override
    if args.chain:
        if "dataset" not in cfg: cfg["dataset"] = {}
        cfg["dataset"]["chain"] = args.chain
        log.info(f"[MC Benchmark] Chain override: {args.chain}")

    setting = str(_cfg_get(cfg, "setting", "strict"))
    output_dir = Path(args.output or _cfg_get(cfg, "output_dir", "results/benchmark_mc"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = _build_dataset_from_cfg(cfg)
    chain = cfg.get("dataset", {}).get("chain", 'polygon')
    
    table = BenchmarkTable()
    
    # (0) Dynamic in_dim inference from dataset
    in_dim_inferred = 32
    if hasattr(dataset, "train_graphs") and len(dataset.train_graphs) > 0:
        first_item = dataset.train_graphs[0]
        data_obj = getattr(first_item, "graph", first_item)
        if hasattr(data_obj, "x") and data_obj.x is not None:
            in_dim_inferred = data_obj.x.size(-1)
            log.info(f"[MC Benchmark] Inferred dynamic in_dim: {in_dim_inferred} from dataset")

    if "level1" not in cfg: cfg["level1"] = {}

    # Detect if augmentation stages are requested and run Legacy Feature Augmentation
    aug_stages = ["l1_legacy_aug", "l1_l2_legacy_aug"]
    is_aug = any(s in active_stages for s in aug_stages)
    if is_aug:
        log.info("[MC Benchmark] Legacy Feature Augmentation requested. Running legacy models on all graphs...")
        dataset = augment_dataset_with_legacy_scores(dataset, cfg, setting)
        legacy_models = (_cfg_get(cfg.get("legacy", {}), "models", ["DOMINANT", "DONE", "GAE", "AnomalyDAE", "CoLA"]) or [])
        in_dim_inferred += len(legacy_models)
        log.info(f"[MC Benchmark] Updated in_dim to {in_dim_inferred} after legacy augmentation.")

    cfg["level1"]["in_dim"] = in_dim_inferred if in_dim_inferred != 32 else cfg["level1"].get("in_dim", 32)
    log.info(f"[MC Benchmark] Final Level1 Input Dimension: {cfg['level1']['in_dim']}")

    l1_cache_path = output_dir / f"l1_model_weights_{chain}{'_aug' if is_aug else ''}.pt"
    l1_model = None

    if "l1" in active_stages or "l1_legacy_aug" in active_stages:
        import time
        _t0_l1 = time.perf_counter()
        is_l1_aug = "l1_legacy_aug" in active_stages
        stage_label = "Stage 1: Level 1 + MC (Legacy-Aug)" if is_l1_aug else "Stage 1: Level 1 + MC"
        log.info("=" * 50)
        log.info(f"(A) Running {stage_label} ({chain}) ...")
        train_g, valid_g, test_g = _get_split_graphs(dataset, cfg, setting)
        
        if args.max_samples and len(test_g) > args.max_samples:
            log.info(f"[MC Benchmark] Subsetting evaluation test_g to {args.max_samples}")
            test_g = test_g[:args.max_samples]

        trainer = _build_level1_trainer(cfg)
        
        # Load or train
        if l1_cache_path.exists():
            trainer.model.load_state_dict(torch.load(l1_cache_path))
            log.info("L1 Model loaded from cache.")
        else:
            _call_level1_trainer_fit(trainer, train_g, valid_g, dataset.labels, cfg)
            torch.save(trainer.model.state_dict(), l1_cache_path)
        
        l1_model = trainer.model
        
        import psutil
        process_l1 = psutil.Process()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        max_nodes_l1 = max(
            (g.graph.num_nodes if hasattr(g, "graph") else g.num_nodes)
            for g in test_g
        ) if test_g else 0
        
        # Original evaluation
        _, yt_orig, ys_orig = trainer.evaluate(test_g, label_dict=dataset.labels, return_preds=True)
        
        peak_ram_l1_orig = process_l1.memory_info().rss / (1024 * 1024)
        peak_gpu_l1_orig = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        
        res_orig = evaluate_benchmark(
            y_true=yt_orig, y_score=ys_orig,
            model_name="L1-Base-Aug" if is_l1_aug else "L1-Base",
            setting=setting,
            max_nodes_processed=max_nodes_l1, peak_ram_mb=peak_ram_l1_orig, peak_gpu_mb=peak_gpu_l1_orig,
            elapsed_sec=time.perf_counter() - _t0_l1,
        )
        table.add(res_orig)
        
        # Reset memory for MC if we want independent peaks, or just continue. 
        # Continuing is fine since we want the peak of the whole process.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # MC Evaluation
        yt_mc, ys_mc, unc_mc, mc_cfg = evaluate_with_mc(trainer.model, dataset, cfg, setting, stage="l1")
        if yt_mc is not None:
            peak_ram_l1_mc = process_l1.memory_info().rss / (1024 * 1024)
            peak_gpu_l1_mc = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            res_mc = evaluate_benchmark(
                y_true=yt_mc, y_score=ys_mc,
                model_name="L1-MC-Aug" if is_l1_aug else "L1-MC",
                setting=setting,
                max_nodes_processed=max_nodes_l1, peak_ram_mb=peak_ram_l1_mc, peak_gpu_mb=peak_gpu_l1_mc,
                elapsed_sec=time.perf_counter() - _t0_l1,
            )
            
            if args.bootstrap:
                from sklearn.metrics import roc_auc_score, average_precision_score
                log.info("[MC Benchmark] Calculating CIs via bootstrapping...")
                m_auc, l_auc, u_auc = calc_bootstrap_ci(yt_mc, ys_mc, roc_auc_score)
                m_pr, l_pr, u_pr = calc_bootstrap_ci(yt_mc, ys_mc, average_precision_score)
                res_mc.ci_roc_auc = (l_auc, u_auc)
                res_mc.ci_pr_auc = (l_pr, u_pr)
                log.info(f"L1-MC ROC-AUC CI: [{l_auc:.4f}, {u_auc:.4f}]")

            # Additional MC Specific Metrics
            ece = calc_calibration_ece(yt_mc, ys_mc)
            corr = calc_uncertainty_correlation(yt_mc, ys_mc, unc_mc)
            sel_res = run_selective_prediction(yt_mc, ys_mc, unc_mc, coverage_ratio=0.8)

            table.add(res_mc)
            
            # Triage Utility Reporting
            budget_50 = calc_fixed_budget_utility(yt_mc, ys_mc, unc_mc, budget=50)
            budget_1pct = calc_fixed_budget_utility(yt_mc, ys_mc, unc_mc, budget=0.01)
            budget_5pct = calc_fixed_budget_utility(yt_mc, ys_mc, unc_mc, budget=0.05)
            
            log.info(f"MC Utility -> ECE: {ece:.4f}, Err-Unc Corr: {corr:.4f}")
            log.info(f"Selective Prediction (top 80% coverage) -> ROC-AUC: {sel_res.get('roc_auc', 0):.4f}, F1: {sel_res.get('f1', 0):.4f}")
            log.info(f"Triage Utility (Top 50) -> Gain: {budget_50['precision_gain']:.4f} (Cov: {budget_50['coverage']:.2%})")
            log.info(f"Triage Utility (Top 1%) -> Gain: {budget_1pct['precision_gain']:.4f} (Cov: {budget_1pct['coverage']:.2%})")
            log.info(f"Triage Utility (Top 5%) -> Gain: {budget_5pct['precision_gain']:.4f} (Cov: {budget_5pct['coverage']:.2%})")

    if ("l1_l2" in active_stages or "l1_l2_legacy_aug" in active_stages) and l1_model is not None:
        import time
        _t0_l1l2 = time.perf_counter()
        is_l2_aug = "l1_l2_legacy_aug" in active_stages
        stage_label = "Stage 2: Level 1+L2+MC (Legacy-Aug)" if is_l2_aug else "Stage 2: Level 1 + Level 2 + MC"
        log.info("=" * 50)
        log.info(f"(B) Running {stage_label} ({chain}) ...")
        train_g, valid_g, test_g = _get_split_graphs(dataset, cfg, setting)
        l2_trainer = _build_level2_trainer(cfg, l1_model)
        
        _call_level2_trainer_fit(
            trainer=l2_trainer, l1_model=l1_model, cfg=cfg,
            train_ids=train_g, valid_ids=valid_g, labels=dataset.labels,
            global_graph=dataset.global_graph,
            loader_builder=_build_l2_dynamic_loader_builder(l1_model, cfg)
        )
        
        import psutil
        process_l1l2 = psutil.Process()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        max_nodes_l1l2 = max(
            (g.num_nodes if hasattr(g, "num_nodes") else g.graph.num_nodes)
            for g in test_g
        ) if test_g else 0
        
        _, yt_orig, ys_orig = l2_trainer.evaluate(
            test_g, label_dict=dataset.labels, global_graph=dataset.global_graph,
            loader_builder=_build_l2_dynamic_loader_builder(l1_model, cfg), return_preds=True
        )
        
        peak_ram_l1l2_orig = process_l1l2.memory_info().rss / (1024 * 1024)
        peak_gpu_l1l2_orig = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        
        res_l1l2_orig = evaluate_benchmark(
            y_true=yt_orig, y_score=ys_orig,
            model_name="L1+L2-Base-Aug" if is_l2_aug else "L1+L2-Base",
            setting=setting,
            max_nodes_processed=max_nodes_l1l2, peak_ram_mb=peak_ram_l1l2_orig, peak_gpu_mb=peak_gpu_l1l2_orig,
            elapsed_sec=time.perf_counter() - _t0_l1l2,
        )
        table.add(res_l1l2_orig)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        yt_mc, ys_mc, unc_mc, mc_cfg = evaluate_with_mc(l2_trainer.model, dataset, cfg, setting, stage="l2", l1_model=l1_model)
        if yt_mc is not None:
            peak_ram_l1l2_mc = process_l1l2.memory_info().rss / (1024 * 1024)
            peak_gpu_l1l2_mc = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
            
            res_mc = evaluate_benchmark(
                y_true=yt_mc, y_score=ys_mc,
                model_name="L1+L2-MC-Aug" if is_l2_aug else "L1+L2-MC",
                setting=setting,
                max_nodes_processed=max_nodes_l1l2, peak_ram_mb=peak_ram_l1l2_mc, peak_gpu_mb=peak_gpu_l1l2_mc,
                elapsed_sec=time.perf_counter() - _t0_l1l2,
            )
            
            if args.bootstrap:
                from sklearn.metrics import roc_auc_score, average_precision_score
                log.info("[MC Benchmark] Calculating CIs via bootstrapping...")
                m_auc, l_auc, u_auc = calc_bootstrap_ci(yt_mc, ys_mc, roc_auc_score)
                m_pr, l_pr, u_pr = calc_bootstrap_ci(yt_mc, ys_mc, average_precision_score)
                res_mc.ci_roc_auc = (l_auc, u_auc)
                res_mc.ci_pr_auc = (l_pr, u_pr)
                log.info(f"L1+L2-MC ROC-AUC CI: [{l_auc:.4f}, {u_auc:.4f}]")

            # Additional MC Specific Metrics
            ece = calc_calibration_ece(yt_mc, ys_mc)
            corr = calc_uncertainty_correlation(yt_mc, ys_mc, unc_mc)
            sel_res = run_selective_prediction(yt_mc, ys_mc, unc_mc, coverage_ratio=0.8)

            table.add(res_mc)
            
            # Triage Utility Reporting
            budget_50 = calc_fixed_budget_utility(yt_mc, ys_mc, unc_mc, budget=50)
            budget_1pct = calc_fixed_budget_utility(yt_mc, ys_mc, unc_mc, budget=0.01)
            budget_5pct = calc_fixed_budget_utility(yt_mc, ys_mc, unc_mc, budget=0.05)
            
            log.info(f"MC Utility -> ECE: {ece:.4f}, Err-Unc Corr: {corr:.4f}")
            log.info(f"Selective Prediction (top 80% coverage) -> ROC-AUC: {sel_res.get('roc_auc', 0):.4f}, F1: {sel_res.get('f1', 0):.4f}")
            log.info(f"Triage Utility (Top 50) -> Gain: {budget_50['precision_gain']:.4f} (Cov: {budget_50['coverage']:.2%})")
            log.info(f"Triage Utility (Top 1%) -> Gain: {budget_1pct['precision_gain']:.4f} (Cov: {budget_1pct['coverage']:.2%})")
            log.info(f"Triage Utility (Top 5%) -> Gain: {budget_5pct['precision_gain']:.4f} (Cov: {budget_5pct['coverage']:.2%})")

    table.save_csv(output_dir / f"mc_benchmark_{chain}.csv")
    table.print_summary()

if __name__ == "__main__":
    main()
