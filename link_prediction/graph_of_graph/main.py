"""Training a SEAL-CI model."""

import torch
import numpy as np
from utils import tab_printer
from train import Trainer
from param_parser import parameter_parser
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parameter_parser()
    tab_printer(args)

    results = {
        f"{dataset}_{metric}": []
        for dataset in ["train", "test"]
        for metric in ["precision", "recall", "f1_macro", "f1_micro"]
    }

    for seed in [42, 43, 44]:  
        print(seed)
        set_seed(seed)

        trainer = Trainer(args, seed)
        trainer.fit()
        metrics = trainer.score()
        
        for set_name, set_results in metrics.items():
            for metric, value in set_results.items():
                results_key = f"{set_name}_{metric}"
                if results_key in results:
                    results[results_key].append(value)
                else:
                    print(f"Unexpected result key: {results_key}, initializing...")
                    results[results_key] = [value]

    print(results)
    for metric_key in sorted(results.keys()):
        mean_value = np.mean(results[metric_key])
        std_dev = np.std(results[metric_key])
        print(f"{metric_key}: Mean = {mean_value:.4f}, Std Dev = {std_dev:.4f}")

if __name__ == "__main__":
    main()
