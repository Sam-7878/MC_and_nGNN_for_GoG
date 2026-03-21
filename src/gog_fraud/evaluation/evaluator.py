from typing import Dict, Iterable, Optional, Union

import torch

from gog_fraud.evaluation.fraud_metrics import (
    bce_loss_from_logits,
    binary_classification_metrics,
    find_best_f1_threshold,
    multi_topk_metrics,
)


def _concat_or_none(items):
    if len(items) == 0:
        return None
    return torch.cat(items, dim=0)


def load_prediction_bundle(path: str):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected dict bundle, got: {type(bundle)}")
    return bundle


class Level1Evaluator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def collect_predictions(self, model, loader) -> Dict:
        model = model.to(self.device)
        model.eval()

        all_graph_id = []
        all_embedding = []
        all_logits = []
        all_score = []
        all_label = []

        for batch in loader:
            batch = batch.to(self.device)
            out = model(batch)

            all_graph_id.append(out.graph_id.detach().cpu().view(-1))
            all_embedding.append(out.embedding.detach().cpu())
            all_logits.append(out.logits.detach().cpu())
            all_score.append(out.score.detach().cpu())

            if out.label is not None:
                all_label.append(out.label.detach().cpu())

        bundle = {
            "graph_id": _concat_or_none(all_graph_id),
            "embedding": _concat_or_none(all_embedding),
            "logits": _concat_or_none(all_logits),
            "score": _concat_or_none(all_score),
            "label": _concat_or_none(all_label),
            "metadata": {
                "device": self.device,
                "num_graphs": int(_concat_or_none(all_graph_id).numel()) if len(all_graph_id) > 0 else 0,
            },
        }
        return bundle

    def evaluate_bundle(
        self,
        bundle: Dict,
        threshold: float = 0.5,
        topk: Optional[Union[int, Iterable[int]]] = None,
        search_best_threshold: bool = False,
    ) -> Dict[str, float]:
        if "score" not in bundle:
            raise KeyError("Bundle must contain `score`")
        if bundle.get("label", None) is None:
            raise ValueError("Bundle must contain `label` for evaluation")

        y_true = bundle["label"].view(-1)
        y_score = bundle["score"].view(-1)

        metrics = binary_classification_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
        )

        if bundle.get("logits", None) is not None:
            metrics["bce_loss"] = bce_loss_from_logits(bundle["logits"], y_true)

        if topk is not None:
            if isinstance(topk, int):
                topk = [topk]
            metrics.update(multi_topk_metrics(y_true, y_score, ks=topk))

        if search_best_threshold:
            metrics.update(find_best_f1_threshold(y_true, y_score))

        return metrics

    def evaluate_loader(
        self,
        model,
        loader,
        threshold: float = 0.5,
        topk: Optional[Union[int, Iterable[int]]] = None,
        search_best_threshold: bool = False,
        return_bundle: bool = False,
    ):
        bundle = self.collect_predictions(model, loader)
        metrics = self.evaluate_bundle(
            bundle=bundle,
            threshold=threshold,
            topk=topk,
            search_best_threshold=search_best_threshold,
        )

        if return_bundle:
            return {
                "metrics": metrics,
                "bundle": bundle,
            }
        return metrics

    def evaluate_bundle_file(
        self,
        path: str,
        threshold: float = 0.5,
        topk: Optional[Union[int, Iterable[int]]] = None,
        search_best_threshold: bool = False,
    ):
        bundle = load_prediction_bundle(path)
        return self.evaluate_bundle(
            bundle=bundle,
            threshold=threshold,
            topk=topk,
            search_best_threshold=search_best_threshold,
        )

