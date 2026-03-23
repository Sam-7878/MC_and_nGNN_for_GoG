import json
import math
from pathlib import Path

import torch
import yaml

from gog_fraud.data.io.dataset import FraudDataset

try:
    from pygod.detector import DOMINANT
except Exception as e:
    DOMINANT = None
    _import_error = repr(e)
else:
    _import_error = None


TOP5 = [
    "0x000de668684839f97d4845f32a43e913366ec08c",
    "0x018a2e4d1afc7d7671cdc428f88452b5cf31599a",
    "0x01c4c105076bdb01ba329543ff99c85f4097a9c9",
    "0x01d35cbc2070a3b76693ce2b6364eae24eb88591",
    "0x031e995926966156484b5b3159628d68e5335d1c",
]


def _to_float(v):
    try:
        if v is None:
            return None
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _tensor_stats(x: torch.Tensor):
    if x is None:
        return None

    out = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "numel": int(x.numel()),
    }

    try:
        out["finite_all"] = bool(torch.isfinite(x).all().item())
        out["nan_count"] = int(torch.isnan(x).sum().item())
        out["inf_count"] = int(torch.isinf(x).sum().item())
    except Exception:
        out["finite_all"] = None
        out["nan_count"] = None
        out["inf_count"] = None

    try:
        xf = x.float()
        out["min"] = _to_float(xf.min().item())
        out["max"] = _to_float(xf.max().item())
        out["mean"] = _to_float(xf.mean().item())
        out["std"] = _to_float(xf.std().item()) if xf.numel() > 1 else 0.0
    except Exception:
        out["min"] = None
        out["max"] = None
        out["mean"] = None
        out["std"] = None

    try:
        head_rows = min(5, x.shape[0]) if x.ndim >= 1 else 0
        if x.ndim == 2 and head_rows > 0:
            out["head"] = x[:head_rows].detach().cpu().tolist()
        elif x.ndim == 1 and head_rows > 0:
            out["head"] = x[:head_rows].detach().cpu().tolist()
        else:
            out["head"] = None
    except Exception:
        out["head"] = None

    return out


def _edge_stats(edge_index: torch.Tensor, num_nodes: int):
    if edge_index is None:
        return None

    out = {
        "shape": list(edge_index.shape),
        "dtype": str(edge_index.dtype),
    }

    try:
        out["num_edges"] = int(edge_index.size(1))
    except Exception:
        out["num_edges"] = None

    try:
        out["min_index"] = int(edge_index.min().item())
        out["max_index"] = int(edge_index.max().item())
        out["index_in_range"] = bool(
            edge_index.min().item() >= 0 and edge_index.max().item() < num_nodes
        )
    except Exception:
        out["min_index"] = None
        out["max_index"] = None
        out["index_in_range"] = None

    try:
        self_loops = (edge_index[0] == edge_index[1]).sum().item()
        out["self_loops"] = int(self_loops)
    except Exception:
        out["self_loops"] = None

    try:
        src = edge_index[0]
        dst = edge_index[1]
        deg = torch.bincount(src, minlength=num_nodes) + torch.bincount(dst, minlength=num_nodes)
        out["degree_min"] = int(deg.min().item()) if deg.numel() else None
        out["degree_max"] = int(deg.max().item()) if deg.numel() else None
        out["degree_mean"] = _to_float(deg.float().mean().item()) if deg.numel() else None
        out["isolated_nodes"] = int((deg == 0).sum().item()) if deg.numel() else None
    except Exception:
        out["degree_min"] = None
        out["degree_max"] = None
        out["degree_mean"] = None
        out["isolated_nodes"] = None

    return out


def _label_of(ds, cid):
    labels = getattr(ds, "labels", None)
    if isinstance(labels, dict):
        rec = labels.get(cid)
        if isinstance(rec, dict):
            for k in ["label", "y", "target"]:
                if k in rec:
                    return rec[k]
        return rec
    return None


def _split_of(ds, cid):
    for split_name in ["train_graphs", "valid_graphs", "test_graphs"]:
        arr = getattr(ds, split_name, None) or []
        for item in arr:
            if getattr(item, "contract_id", None) == cid:
                return split_name.replace("_graphs", "")
    return None


def _find_graph(ds, cid):
    for split_name in ["train_graphs", "valid_graphs", "test_graphs"]:
        arr = getattr(ds, split_name, None) or []
        for item in arr:
            if getattr(item, "contract_id", None) == cid:
                return item
    for item in getattr(ds, "tx_graphs", []) or []:
        if getattr(item, "contract_id", None) == cid:
            return item
    return None


def _dominant_debug(data):
    result = {
        "pygod_import_error": _import_error,
        "fit_ok": False,
        "fit_error": None,
        "detector_type": "DOMINANT" if DOMINANT is not None else None,
        "attrs_after_fit": {},
        "score_probe": {},
    }

    if DOMINANT is None:
        result["fit_error"] = "pygod.detector.DOMINANT import failed"
        return result

    detector = None
    try:
        detector = DOMINANT(epoch=5, verbose=0)
        detector.fit(data)
        result["fit_ok"] = True
    except Exception as e:
        result["fit_error"] = repr(e)
        return result

    attrs = {}
    for name in [
        "decision_scores_",
        "decision_score_",
        "threshold_",
        "label_",
    ]:
        attrs[name] = hasattr(detector, name)
    result["attrs_after_fit"] = attrs

    probe = {}

    if hasattr(detector, "decision_scores_"):
        try:
            s = getattr(detector, "decision_scores_")
            t = torch.as_tensor(s).view(-1).float()
            probe["decision_scores_"] = {
                "ok": True,
                "len": int(t.numel()),
                "min": _to_float(t.min().item()) if t.numel() else None,
                "max": _to_float(t.max().item()) if t.numel() else None,
                "mean": _to_float(t.mean().item()) if t.numel() else None,
            }
        except Exception as e:
            probe["decision_scores_"] = {"ok": False, "error": repr(e)}

    if hasattr(detector, "decision_score_"):
        try:
            s = getattr(detector, "decision_score_")
            t = torch.as_tensor(s).view(-1).float()
            probe["decision_score_"] = {
                "ok": True,
                "len": int(t.numel()),
                "min": _to_float(t.min().item()) if t.numel() else None,
                "max": _to_float(t.max().item()) if t.numel() else None,
                "mean": _to_float(t.mean().item()) if t.numel() else None,
            }
        except Exception as e:
            probe["decision_score_"] = {"ok": False, "error": repr(e)}

    probe["has_decision_function"] = hasattr(detector, "decision_function")
    if hasattr(detector, "decision_function"):
        try:
            s = detector.decision_function(data)
            t = torch.as_tensor(s).view(-1).float()
            probe["decision_function(data)"] = {
                "ok": True,
                "len": int(t.numel()),
                "min": _to_float(t.min().item()) if t.numel() else None,
                "max": _to_float(t.max().item()) if t.numel() else None,
                "mean": _to_float(t.mean().item()) if t.numel() else None,
            }
        except Exception as e:
            probe["decision_function(data)"] = {"ok": False, "error": repr(e)}

    result["score_probe"] = probe
    return result


def main():
    root = Path(__file__).resolve().parents[3]
    cfg_path = root / "configs" / "benchmark" / "strict_smoke.yaml"
    out_dir = root.parent / "_data" / "results" / "strict_smoke" / "dominant_warning_top5"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = FraudDataset.from_config(cfg["dataset"])

    index = []

    for cid in TOP5:
        item = _find_graph(ds, cid)
        if item is None:
            payload = {
                "contract_id": cid,
                "found": False,
                "reason": "contract_id not found in dataset splits",
            }
        else:
            data = item.graph if hasattr(item, "graph") else item

            payload = {
                "contract_id": cid,
                "found": True,
                "split": _split_of(ds, cid),
                "label": _label_of(ds, cid),
                "wrapper_type": type(item).__name__,
                "data_type": type(data).__name__,
                "num_nodes": int(getattr(data, "num_nodes", 0) or 0),
                "num_edges": int(data.edge_index.size(1)) if getattr(data, "edge_index", None) is not None else None,
                "has_x": getattr(data, "x", None) is not None,
                "has_edge_index": getattr(data, "edge_index", None) is not None,
                "x_stats": _tensor_stats(getattr(data, "x", None)),
                "edge_stats": _edge_stats(getattr(data, "edge_index", None), int(getattr(data, "num_nodes", 0) or 0)),
                "dominant_debug": _dominant_debug(data),
            }

        out_path = out_dir / f"{cid}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        index.append(
            {
                "contract_id": cid,
                "json_path": str(out_path),
                "found": payload.get("found", False),
                "split": payload.get("split"),
                "label": payload.get("label"),
            }
        )

    index_path = out_dir / "_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(index)} JSON files to: {out_dir}")
    print(f"Index file: {index_path}")


if __name__ == "__main__":
    main()
