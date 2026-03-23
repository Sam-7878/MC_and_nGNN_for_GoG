
import yaml, torch
from gog_fraud.data.io.dataset import FraudDataset
from gog_fraud.training.loops.level1 import _prepare_level1_loader

with open("./configs/benchmark/strict_smoke.yaml") as f:
    cfg = yaml.safe_load(f)

ds = FraudDataset.from_config(cfg["dataset"])

# Case 1: List[TransactionGraph] → auto-unwrap
train_loader = _prepare_level1_loader(
    ds.train_graphs[:16],
    split_name="train",
    batch_size=4,
    shuffle=True,
)
print("Case1 OK | type:", type(train_loader).__name__, "| batches:", len(train_loader))

# Case 2: valid graphs
valid_loader = _prepare_level1_loader(
    ds.valid_graphs[:8],
    split_name="valid",
    batch_size=4,
    shuffle=False,
)
print("Case2 OK | type:", type(valid_loader).__name__, "| batches:", len(valid_loader))

# Case 3: single batch inspection
batch = next(iter(train_loader))
print("batch.x     :", tuple(batch.x.shape))
print("batch.edge_index:", tuple(batch.edge_index.shape))
print("batch.x dtype   :", batch.x.dtype)
print("batch.edge_index dtype:", batch.edge_index.dtype)

