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
    
    # Validation against dimension parity just in case (e.g. graph vs node resolution)
    min_size = min(len(yt), len(ys))
    
    return yt[:min_size], ys[:min_size], unc[:min_size], mc_cfg

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
    output_dir = Path(args.output or _cfg_get(cfg, "output", "results/benchmark_mc"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = _build_dataset_from_cfg(cfg)
    chain = cfg.get("dataset", {}).get("chain", 'polygon')
    
    table = BenchmarkTable()
    l1_cache_path = output_dir / f"l1_model_weights_{chain}.pt"
    l1_model = None

    if "l1" in active_stages:
        log.info("=" * 50)
        log.info(f"(A) Running Stage 1: Level 1 + MC ({chain}) ...")
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
        
        # Original evaluation
        _, yt_orig, ys_orig = trainer.evaluate(test_g, label_dict=dataset.labels, return_preds=True)
        res_orig = evaluate_benchmark(y_true=yt_orig, y_score=ys_orig, model_name="L1-Base", setting=setting)
        table.add(res_orig)
        
        # MC Evaluation
        yt_mc, ys_mc, unc_mc, mc_cfg = evaluate_with_mc(trainer.model, dataset, cfg, setting, stage="l1")
        if yt_mc is not None:
            res_mc = evaluate_benchmark(y_true=yt_mc, y_score=ys_mc, model_name="L1-MC", setting=setting)
            
            if args.bootstrap:
                from sklearn.metrics import roc_auc_score, average_precision_score
                log.info("[MC Benchmark] Calculating CIs via bootstrapping...")
                m_auc, l_auc, u_auc = calc_bootstrap_ci(yt_mc, ys_mc, roc_auc_score)
                m_pr, l_pr, u_pr = calc_bootstrap_ci(yt_mc, ys_mc, average_precision_score)
                res_mc["roc_auc_ci"] = (l_auc, u_auc)
                res_mc["pr_auc_ci"] = (l_pr, u_pr)
                log.info(f"L1-MC ROC-AUC CI: [{l_auc:.4f}, {u_auc:.4f}]")

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

    if "l1_l2" in active_stages and l1_model is not None:
        log.info("=" * 50)
        log.info("(B) Running Stage 2: Level 1 + Level 2 + MC ...")
        train_g, valid_g, test_g = _get_split_graphs(dataset, cfg, setting)
        l2_trainer = _build_level2_trainer(cfg, l1_model)
        
        _call_level2_trainer_fit(
            trainer=l2_trainer, l1_model=l1_model, cfg=cfg,
            train_ids=train_g, valid_ids=valid_g, labels=dataset.labels,
            global_graph=dataset.global_graph,
            loader_builder=_build_l2_dynamic_loader_builder(l1_model, cfg)
        )
        
        _, yt_orig, ys_orig = l2_trainer.evaluate(
            test_g, label_dict=dataset.labels, global_graph=dataset.global_graph,
            loader_builder=_build_l2_dynamic_loader_builder(l1_model, cfg), return_preds=True
        )
        res_orig = evaluate_benchmark(y_true=yt_orig, y_score=ys_orig, model_name="L1+L2-Base", setting=setting)
        table.add(res_orig)
        
        yt_mc, ys_mc, unc_mc, mc_cfg = evaluate_with_mc(l2_trainer.model, dataset, cfg, setting, stage="l2", l1_model=l1_model)
        if yt_mc is not None:
            res_mc = evaluate_benchmark(y_true=yt_mc, y_score=ys_mc, model_name="L1+L2-MC", setting=setting)
            
            if args.bootstrap:
                from sklearn.metrics import roc_auc_score, average_precision_score
                log.info("[MC Benchmark] Calculating CIs via bootstrapping...")
                m_auc, l_auc, u_auc = calc_bootstrap_ci(yt_mc, ys_mc, roc_auc_score)
                m_pr, l_pr, u_pr = calc_bootstrap_ci(yt_mc, ys_mc, average_precision_score)
                res_mc["roc_auc_ci"] = (l_auc, u_auc)
                res_mc["pr_auc_ci"] = (l_pr, u_pr)
                log.info(f"L1+L2-MC ROC-AUC CI: [{l_auc:.4f}, {u_auc:.4f}]")

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
