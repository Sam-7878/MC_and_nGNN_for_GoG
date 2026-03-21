import argparse
from pathlib import Path
from sys import path
from typing import Optional

import torch
from torch_geometric.loader import DataLoader

from gog_fraud.models.level1.model import Level1Model, Level1ModelConfig


def load_graph_list(path: str):
    # graphs = torch.load(path, map_location="cpu")
    graphs = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(graphs, list):
        raise TypeError(f"Expected a list of PyG Data objects, but got: {type(graphs)}")
    return graphs


def load_model_from_checkpoint(checkpoint_path: str, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_config" not in ckpt:
        raise KeyError("Checkpoint must contain 'model_config'")
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain 'model_state_dict'")

    model_cfg = Level1ModelConfig(**ckpt["model_config"])
    model = Level1Model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_cfg, ckpt, device


@torch.no_grad()
def export_level1_embeddings(
    data_path: str,
    checkpoint_path: str,
    output_path: str,
    batch_size: int = 32,
    device: Optional[str] = None,
):
    graphs = load_graph_list(data_path)
    model, model_cfg, ckpt, device = load_model_from_checkpoint(checkpoint_path, device=device)

    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_graph_id = []
    all_embedding = []
    all_logits = []
    all_score = []
    all_label = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)

        all_graph_id.append(out.graph_id.detach().cpu().view(-1))
        all_embedding.append(out.embedding.detach().cpu())
        all_logits.append(out.logits.detach().cpu())
        all_score.append(out.score.detach().cpu())

        if out.label is not None:
            all_label.append(out.label.detach().cpu())

    bundle = {
        "graph_id": torch.cat(all_graph_id, dim=0),
        "embedding": torch.cat(all_embedding, dim=0),
        "logits": torch.cat(all_logits, dim=0),
        "score": torch.cat(all_score, dim=0),
        "label": torch.cat(all_label, dim=0) if len(all_label) > 0 else None,
        "metadata": {
            "model_config": ckpt.get("model_config", {}),
            "trainer_config": ckpt.get("trainer_config", {}),
            "checkpoint_epoch": ckpt.get("epoch", None),
            "num_graphs": len(graphs),
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    print(f"Saved Level1 embedding bundle to: {output_path}")

    return bundle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    export_level1_embeddings(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        output_path=args.output_path,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
