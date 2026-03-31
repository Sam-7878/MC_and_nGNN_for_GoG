import os
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .dataset import FraudDataset

log = logging.getLogger(__name__)

class StreamingDataset(FraudDataset):
    """
    Extends FraudDataset to enforce chronological splitting instead of random stratified splits.
    Scans the earliest transaction timestamp for each contract.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chronological_order = []
        
    def prepare_streaming_splits(self, transactions_root: str, train_ratio: float = 0.8):
        """
        Calculates the 80/20 chronological split based on the earliest transaction timestamp.
        """
        log.info("[StreamingDataset] Scanning transaction timestamps for sorting ... This may take a moment.")
        
        tx_dir = Path(transactions_root) / self.cfg.chain if self.cfg.chain else Path(transactions_root)
        
        contract_timestamps = {}
        missing = 0
        
        # For each contract in labels, find its earliest timestamp
        for cid in self.labels.keys():
            contract_dir = tx_dir / str(cid)
            if not contract_dir.exists():
                missing += 1
                continue
                
            csv_files = list(contract_dir.glob("*.csv"))
            if not csv_files:
                missing += 1
                continue
            
            # Read just the first row of all CSVs in the contract folder
            min_ts = float('inf')
            for csv_f in csv_files:
                try:
                    df = pd.read_csv(csv_f, nrows=1)
                    # Normalize columns to lowercase for check
                    cols_lower = [c.lower() for c in df.columns]
                    
                    if 'timestamp' in cols_lower:
                        idx = cols_lower.index('timestamp')
                        ts = int(df.iloc[0, idx])
                        min_ts = min(min_ts, ts)
                    elif 'block_number' in cols_lower:
                        idx = cols_lower.index('block_number')
                        blk = int(df.iloc[0, idx])
                        min_ts = min(min_ts, blk)
                    elif 'blocknumber' in cols_lower:
                        idx = cols_lower.index('blocknumber')
                        blk = int(df.iloc[0, idx])
                        min_ts = min(min_ts, blk)
                except Exception:
                    continue
            
            if min_ts != float('inf'):
                contract_timestamps[cid] = min_ts
            else:
                missing += 1
                
        log.info(f"[StreamingDataset] Located timestamps for {len(contract_timestamps)} contracts. Missing/Empty: {missing}")
        
        # Sort chronologically
        sorted_contracts = sorted(contract_timestamps.items(), key=lambda x: x[1])
        
        # Build split indices
        n_total = len(sorted_contracts)
        n_train = int(n_total * train_ratio)
        
        train_cids = {x[0] for x in sorted_contracts[:n_train]}
        stream_cids = {x[0] for x in sorted_contracts[n_train:]}
        
        self.chronological_order = [x[0] for x in sorted_contracts[n_train:]]
        
        log.info(f"[StreamingDataset] Chronological Split - Historical Context: {len(train_cids)}, Streaming Sequence: {len(stream_cids)}")
        
        # Assign back to the base datastructures dynamically
        self.train_graphs = [g for g in self.transaction_graphs if getattr(g, "contract_id", "") in train_cids]
        
        # In this dataset, the explicit streaming sequence retains chronological ordering
        stream_unsorted = {getattr(g, "contract_id", ""): g for g in self.transaction_graphs if getattr(g, "contract_id", "") in stream_cids}
        self.stream_graphs = [stream_unsorted[c] for c in self.chronological_order if c in stream_unsorted]
        
        return self.train_graphs, self.stream_graphs

    def get_streaming_graphs(self):
        """ Returns the strictly ordered testing graphs """
        return getattr(self, "stream_graphs", self.test_graphs)
