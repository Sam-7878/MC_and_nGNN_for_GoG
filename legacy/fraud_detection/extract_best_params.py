import os
import json
import csv
import ast
import re
from pathlib import Path

def parse_txt(file_path):
    results = []
    # 2026:04:03_23:37:52 --> Tested DOMINANT with {'hid_dim': 4, 'lr': 0.003, 'epoch': 20}: Avg AUC=0.5567, Std AUC=0.0000, Avg AP=0.3636, Std AP=0.0000
    pattern = re.compile(r"Tested (\w+) with (\{.*\}): Avg AUC=([\d\.]+).*Avg AP=([\d\.]+)")
    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                model = m.group(1)
                params_str = m.group(2)
                auc = float(m.group(3))
                ap = float(m.group(4))
                results.append({
                    'model': model,
                    'params': ast.literal_eval(params_str),
                    'auc': auc,
                    'ap': ap
                })
    return results

def parse_csv(file_path):
    results = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['Model']
            params_str = row['Best Params']
            auc = float(row['Val AUC'])
            ap = float(row['Val AP'])
            results.append({
                'model': model,
                'params': ast.literal_eval(params_str),
                'auc': auc,
                'ap': ap
            })
    return results

def extract_best(results):
    # Group by model
    grouped = {}
    for r in results:
        model = r['model']
        if model not in grouped:
            grouped[model] = []
        grouped[model].append(r)
    
    # Find best per model (based on AUC, then AP)
    best_params = {}
    for model, runs in grouped.items():
        # Sort descending
        best_run = max(runs, key=lambda x: (x['auc'], x['ap']))
        best_params[model] = best_run['params']
    return best_params

def main():
    base_dir = Path("../../results/legacy_benchmark")
    out_dir = Path("../../configs/legacy/best_params")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = {
        'bsc': ('main-results_bsc_log.csv', parse_csv),
        'polygon': ('main-results_polygon_log.csv', parse_csv),
        'ethereum': ('main-results_ethereum_log.txt', parse_txt)
    }
    
    for chain, (filename, parse_func) in files.items():
        file_path = base_dir / filename
        if not file_path.exists():
            print(f"Warning: {file_path} not found.")
            continue
        
        print(f"Parsing {chain} from {filename}...")
        results = parse_func(file_path)
        best_params = extract_best(results)
        
        out_file = out_dir / f"best_params_{chain}.json"
        with open(out_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Saved best params for {chain} to {out_file}")

if __name__ == "__main__":
    main()
