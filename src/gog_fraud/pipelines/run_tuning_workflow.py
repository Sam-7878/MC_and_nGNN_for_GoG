# src/gog_fraud/pipelines/run_tuning_workflow.py
import subprocess
import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MasterWorkflow")

def run_cmd(cmd, cwd="."):
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env={**os.environ, "PYTHONPATH": "src"})
    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        return False
    return True

def run_workflow():
    py_exec = "/mnt/d/_Work/MC_and_nGNN_for_GoG/.venv/bin/python3"
    out_dir = "docs/work_reports/legacy_param_search/"
    
    # 1. Phase 1: Coarse Screening
    logger.info("=== Phase 1: Coarse Screening ===")
    p1_dir = os.path.join(out_dir, "phase1")
    os.makedirs(p1_dir, exist_ok=True)
    
    # Check if results already exist (resume capability)
    # For now, we assume the user might have already started it. 
    # The script itself has resumption logic via checkpoints.
    p1_cmd = [
        py_exec, "src/gog_fraud/pipelines/search_legacy_params.py",
        "--chains", "bsc,ethereum,polygon",
        "--workers", "8",
        "--gpu_limit", "1",
        "--coarse",
        "--out_dir", p1_dir
    ]
    if not run_cmd(p1_cmd):
        return

    # 2. Phase 2: Refinement
    logger.info("=== Phase 2: Refinement ===")
    p2_dir = os.path.join(out_dir, "phase2")
    os.makedirs(p2_dir, exist_ok=True)
    
    for chain in ["bsc", "ethereum", "polygon"]:
        p1_best_file = Path(f"configs/legacy/best_params/best_params_{chain}.json")
        if not p1_best_file.exists():
            logger.warning(f"No Phase 1 best file for {chain}. Skipping refinement.")
            continue
            
        p2_cmd = [
            py_exec, "src/gog_fraud/pipelines/search_legacy_params.py",
            "--chains", chain,
            "--workers", "8",
            "--gpu_limit", "1",
            "--refine_from", str(p1_best_file),
            "--out_dir", p2_dir
        ]
        if not run_cmd(p2_cmd):
            logger.error(f"Refinement failed for {chain}")
            # Continue to other chains anyway

    # 3. Final Benchmark
    logger.info("=== Stage 3: Final Comparative Benchmark ===")
    # Note: run_fraud_benchmark.py uses the injected best_params via LegacyBatchRunner
    # whose code we updated to load from configs/legacy/best_params/
    bench_cmd = [
        py_exec, "src/gog_fraud/pipelines/run_fraud_benchmark.py"
    ]
    run_cmd(bench_cmd)

    logger.info("Master Workflow Completed.")

if __name__ == "__main__":
    run_workflow()
