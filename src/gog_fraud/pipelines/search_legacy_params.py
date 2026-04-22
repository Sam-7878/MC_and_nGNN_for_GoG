# src/gog_fraud/pipelines/search_legacy_params.py
import os
import json
import logging
import gc
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import pandas as pd
from tqdm import tqdm

from gog_fraud.adapters.legacy_adapter import (
    LegacyAdapterConfig, 
    LegacyBatchRunner, 
    LegacyEvalItem,
    _unwrap_data, 
    _extract_contract_id, 
    _extract_label, 
    _prepare_graph_for_detector,
    _get_partition_cache_path,
    _partition_graph
)
from gog_fraud.data.io.dataset import FraudDataset, DatasetConfig
from gog_fraud.evaluation.benchmark import evaluate_benchmark

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LegacySearch")

# 전역 리소스 제한 (GPU fit 제한용)
gpu_semaphore = None

# 전역 데이터 (Fork 시 복사 방지용)
global_valid_graphs = None
global_labels_dict = None

def init_worker(sema, eval_items, labels_dict):
    global gpu_semaphore, global_valid_graphs, global_labels_dict
    gpu_semaphore = sema
    global_valid_graphs = eval_items
    global_labels_dict = labels_dict
    
    # Delayed internal imports to avoid parent state contamination
    import torch
    import pygod
    torch.set_num_threads(1)  # Limit CPU thread bloat per worker

def _run_single_config(
    chain: str,
    model_name: str,
    params: Dict[str, Any],
    adapter_base_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    단일 하이퍼파라미터 조합에 대해 검증 세트 평가를 수행한다.
    """
    try:
        # GPU 점유 제한 시작
        if gpu_semaphore:
            gpu_semaphore.acquire()
            
        import torch
        # 전역 GPU 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        config = LegacyAdapterConfig(
            chain=chain,
            detector_overrides={model_name: params},
            **adapter_base_cfg
        )
        runner = LegacyBatchRunner(config=config)
        
        # 1. 모델 실행 (이미 전처리된 아이템 사용)
        output = runner.run_on_items(model_name, global_valid_graphs)
        
        if not output.records:
            return {"error": "no_records"}
            
        # 2. 결과 추출 및 평가
        partitioned_count = getattr(output, "partitioned_graph_count", 0)
        y_score = {r.contract_id: r.score for r in output.records}
        y_true = global_labels_dict
        
        result = evaluate_benchmark(
            y_true=y_true,
            y_score=y_score,
            model_name=model_name,
            setting=f"search_{chain}",
            bootstrap=False  # 검색 속도를 위해 bootstrap은 끔
        )
        
        # 결과 정리
        return {
            "chain": chain,
            "model": model_name,
            "hid_dim": params["hid_dim"],
            "lr": params["lr"],
            "epoch": params["epoch"],
            "pr_auc": float(result.pr_auc),
            "roc_auc": float(result.roc_auc),
            "best_f1": float(result.best_f1),
            "num_samples": result.num_samples,
            "partitioned_count": int(partitioned_count),
            "error": None
        }
        
    except Exception as exc:
        logger.error(f"[Worker] Error searching {model_name} {params}: {exc}")
        return {"error": str(exc)}
    finally:
        if gpu_semaphore:
            gpu_semaphore.release()
        gc.collect()

def _pre_partition_graphs(graphs, labels_dict, adapter_cfg: Dict[str, Any], chain: str) -> List[LegacyEvalItem]:
    import torch
    logger.info("Pre-partitioning %d graphs for chain %s...", len(graphs), chain)
    eval_items = []
    
    max_nodes = adapter_cfg.get("max_nodes", 4096)
    p_size = adapter_cfg.get("partition_size", 4096)
    p_overlap = adapter_cfg.get("partition_overlap", 0.0)
    cache_dir = adapter_cfg.get("partition_cache_dir", "../_data/dataset/.cache/partitioned_graphs/")

    for idx, item in enumerate(tqdm(graphs, desc="Pre-partitioning")):
        data = _unwrap_data(item)
        contract_id = _extract_contract_id(item, data, idx)
        label = labels_dict.get(contract_id)
        
        if data is None:
            continue
            
        num_nodes = getattr(data, 'num_nodes', None)
        if num_nodes is None and getattr(data, 'x', None) is not None:
            num_nodes = data.x.size(0)
            
        is_large = (num_nodes is not None and num_nodes > max_nodes)
        
        eval_item = LegacyEvalItem(contract_id=contract_id, label=label, is_large=is_large)
        
        if is_large:
            # Check cache
            cp = _get_partition_cache_path(cache_dir, chain, contract_id, p_size, p_overlap)
            if cp.exists():
                try:
                    eval_item.subgraphs = torch.load(cp, weights_only=False)
                except:
                    eval_item.subgraphs = _partition_graph(data, p_size, p_overlap)
            else:
                eval_item.subgraphs = _partition_graph(data, p_size, p_overlap)
                try:
                    torch.save(eval_item.subgraphs, cp)
                except:
                    pass
        else:
            eval_item.data = _prepare_graph_for_detector(data)
            # Ensure num_nodes is explicitly set even for small graphs
            eval_item.data.num_nodes = eval_item.data.x.size(0)
            
        eval_items.append(eval_item)
        
    return eval_items

def perform_search():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=str, default="polygon,bsc,ethereum")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu_limit", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="docs/work_reports/legacy_param_search/")
    parser.add_argument("--coarse", action="store_true", help="Run a coarse grid first")
    parser.add_argument("--refine_from", type=str, default=None, help="Refine search around best params from this JSON")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    chains = [c.strip() for c in args.chains.split(",")]
    
    # 전체 검색 공간 (Full Grid)
    FULL_HID_DIMS = [4, 8, 12, 16, 20]
    FULL_LRS = [0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03]
    FULL_EPOCHS = [20, 30, 40, 50, 80, 100, 120]

    def get_neighbors(best_p: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find neighbors in the full grid for refinement.
        """
        def neighbors(val, space):
            try:
                idx = space.index(val)
                return space[max(0, idx-1) : min(len(space), idx+2)]
            except ValueError:
                return [val]

        nh = neighbors(best_p["hid_dim"], FULL_HID_DIMS)
        nl = neighbors(best_p["lr"], FULL_LRS)
        ne = neighbors(best_p["epoch"], FULL_EPOCHS)
        
        nbhd = []
        for h in nh:
            for l in nl:
                for e in ne:
                    nbhd.append({"hid_dim": h, "lr": l, "epoch": e})
        return nbhd

    models = ["DOMINANT", "DONE", "GAE", "AnomalyDAE", "CoLA"]

    # 1. 검색 공간 결정
    if args.refine_from:
        # Phase 2: Refinement mode
        with open(args.refine_from, 'r') as f:
            phase1_best = json.load(f)
        # Note: Grid will be per-model in the loop below
    elif args.coarse:
        # Phase 1: Mini-grid
        hid_dims = [4, 12, 20]
        lrs = [0.003, 0.015, 0.03]
        epochs = [20, 50, 120]
        grid = [{"hid_dim": h, "lr": l, "epoch": e} for h in hid_dims for l in lrs for e in epochs]
    else:
        # Standard Full Sweep
        grid = [{"hid_dim": h, "lr": l, "epoch": e} for h in FULL_HID_DIMS for l in FULL_LRS for e in FULL_EPOCHS]
                
    logger.info(f"Grid size: {len(grid)} combinations per model/chain.")
    
    # 공통 데이터 로더 베이스 경로 (configs/benchmark/full_system.yaml 참고)
    base_data_cfg = {
        "transactions_root": "../_data/dataset/transactions",
        "labels_path": "../_data/dataset/labels.csv",
        "global_graph_root": "../_data/dataset/global_graph"
    }
    
    # Adapter 공통 설정 (Partition 등)
    adapter_base_cfg = {
        "max_nodes": 4096,
        "partition_size": 4096,
        "large_graph_mode": "partition",
        "aggregation_method": "max"
    }
    
    sema = multiprocessing.Manager().Semaphore(args.gpu_limit)
    
    all_best_params = {}
    
    for chain in chains:
        logger.info(f"=== Starting Search for Chain: {chain} ===")
        
        # 데이터 로드
        try:
            ds_cfg = DatasetConfig(
                chain=chain,
                **base_data_cfg
            )
            ds = FraudDataset(ds_cfg).load()
            valid_graphs = ds.valid_graphs
            labels_dict = ds.labels
            
            if not valid_graphs:
                logger.warning(f"No validation graphs for {chain}. Skipping.")
                continue
                
            # 최적화: 모델 루프 실행 전 전체 데이터를 한 번만 전처리(파티셔닝 등) 함
            eval_items = _pre_partition_graphs(valid_graphs, labels_dict, adapter_base_cfg, chain)
            
        except Exception as e:
            logger.error(f"Failed to load dataset for {chain}: {e}")
            continue
            
        chain_results = []
        
        for model_name in models:
            logger.info(f"Searching {model_name} on {chain}...")
            
            # 1.1 검색 공간 결정 (Refinement 대응)
            if args.refine_from:
                if chain in phase1_best and model_name in phase1_best[chain]:
                    model_grid = get_neighbors(phase1_best[chain][model_name])
                    logger.info(f"Refining neighbors for {model_name}: {len(model_grid)} tasks.")
                else:
                    logger.warning(f"No Phase 1 best found for {chain}/{model_name}. Skipping.")
                    continue
            else:
                model_grid = grid
            
            suffix = "_refine" if args.refine_from else ""
            checkpoint_file = Path(args.out_dir) / f"checkpoint_{chain}_{model_name}{suffix}.json"
            completed = []
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    completed = json.load(f)
            
            tasks = []
            for p in model_grid:
                if p in completed: continue
                tasks.append(p)
            
            with ProcessPoolExecutor(
                max_workers=args.workers, 
                initializer=init_worker, 
                initargs=(sema, eval_items, labels_dict),
                mp_context=multiprocessing.get_context('spawn')
            ) as executor:
                futures = {
                    executor.submit(
                        _run_single_config, chain, model_name, p, adapter_base_cfg
                    ): p for p in tasks
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"{chain}/{model_name}"):
                    res = future.result()
                    if res.get("error") is None:
                        chain_results.append(res)
                        completed.append(futures[future])
                        
                        # 체크포인트 저장 (간헐적)
                        if len(completed) % 10 == 0:
                            with open(checkpoint_file, 'w') as f:
                                json.dump(completed, f)
            
            # 모델별 최고 성능 찾기
            model_df = pd.DataFrame([r for r in chain_results if r["model"] == model_name])
            if not model_df.empty:
                # Priority: PR-AUC > ROC-AUC > Best F1
                model_df = model_df.sort_values(by=["pr_auc", "roc_auc", "best_f1"], ascending=False)
                best = model_df.iloc[0]
                
                if chain not in all_best_params: all_best_params[chain] = {}
                all_best_params[chain][model_name] = {
                    "hid_dim": int(best["hid_dim"]),
                    "lr": float(best["lr"]),
                    "epoch": int(best["epoch"]),
                    "val_pr_auc": float(best["pr_auc"]),
                    "val_roc_auc": float(best["roc_auc"])
                }
                logger.info(f"Best for {chain}/{model_name}: {all_best_params[chain][model_name]}")
        
        # 체인별 전체 로그 저장
        pd.DataFrame(chain_results).to_csv(Path(args.out_dir) / f"search_results_{chain}.csv", index=False)
        
        # 체인별 최적 파라미터 개별 저장 (Adapter 연동용)
        best_params_file = Path("configs/legacy/best_params/") / f"best_params_{chain}.json"
        os.makedirs(best_params_file.parent, exist_ok=True)
        with open(best_params_file, 'w') as f:
            json.dump(all_best_params[chain], f, indent=4)

    logger.info("Grid Search Completed.")

if __name__ == "__main__":
    perform_search()
