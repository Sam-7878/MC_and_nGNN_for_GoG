"""
pipelines/fusion.py

Level 1 score + Level 2 score → Final fraud score

Fusion 전략 계층:
  1. WeightedSumFusion   : 고정 weight 가중 합산
  2. CalibratedFusion    : temperature scaling 기반 보정 후 합산
  3. LearnedFusion       : MLP 기반 학습형 결합
  4. FusionEnsemble      : 복수 전략의 평균/투표
"""

from __future__ import annotations

import abc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


# ══════════════════════════════════════════════════════════════════════════════
# I/O Types
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FusionInput:
    """
    Level 1 / Level 2 예측값의 표준 컨테이너.

    Attributes:
        level1_score  : [N] or [N, 1]   sigmoid score from Level 1
        level2_score  : [N] or [N, 1]   sigmoid score from Level 2
        level1_logits : [N] or [N, 1]   optional raw logits from Level 1
        level2_logits : [N] or [N, 1]   optional raw logits from Level 2
        label         : [N] or [N, 1]   optional ground truth (0/1)
        graph_id      : [N]             optional graph identifier
        aux           : arbitrary extra metadata
    """
    level1_score:  torch.Tensor
    level2_score:  torch.Tensor
    level1_logits: Optional[torch.Tensor] = None
    level2_logits: Optional[torch.Tensor] = None
    label:         Optional[torch.Tensor] = None
    graph_id:      Optional[torch.Tensor] = None
    aux:           Dict[str, Any]         = field(default_factory=dict)

    def __post_init__(self):
        self.level1_score = self.level1_score.float().view(-1)
        self.level2_score = self.level2_score.float().view(-1)

        if self.level1_logits is not None:
            self.level1_logits = self.level1_logits.float().view(-1)
        if self.level2_logits is not None:
            self.level2_logits = self.level2_logits.float().view(-1)
        if self.label is not None:
            self.label = self.label.float().view(-1)
        if self.graph_id is not None:
            self.graph_id = self.graph_id.long().view(-1)

        n1 = self.level1_score.size(0)
        n2 = self.level2_score.size(0)
        if n1 != n2:
            raise ValueError(
                f"level1_score and level2_score must have the same length, "
                f"got {n1} vs {n2}"
            )

    @property
    def n(self) -> int:
        return int(self.level1_score.size(0))


@dataclass
class FusionOutput:
    """
    Fusion 결과 컨테이너.

    Attributes:
        score          : [N]   최종 fraud probability
        logits         : [N]   최종 logits (log-odds 공간)
        level1_score   : [N]   원본 Level 1 score (참고용)
        level2_score   : [N]   원본 Level 2 score (참고용)
        label          : [N]   ground truth (있을 경우)
        graph_id       : [N]   graph identifier (있을 경우)
        metadata       : dict  fusion 방법, weight 등 기록
    """
    score:        torch.Tensor
    logits:       torch.Tensor
    level1_score: torch.Tensor
    level2_score: torch.Tensor
    label:        Optional[torch.Tensor] = None
    graph_id:     Optional[torch.Tensor] = None
    metadata:     Dict[str, Any]         = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Base interface
# ══════════════════════════════════════════════════════════════════════════════

class BaseFusion(abc.ABC):
    """
    모든 Fusion 전략의 공통 인터페이스.
    하위 클래스는 반드시 `_combine` 을 구현해야 합니다.
    """

    @abc.abstractmethod
    def _combine(self, fusion_input: FusionInput) -> torch.Tensor:
        """
        Returns:
            logits [N] in log-odds space
        """
        raise NotImplementedError

    def fuse(self, fusion_input: FusionInput) -> FusionOutput:
        logits = self._combine(fusion_input)
        score  = torch.sigmoid(logits)

        return FusionOutput(
            score=score,
            logits=logits,
            level1_score=fusion_input.level1_score,
            level2_score=fusion_input.level2_score,
            label=fusion_input.label,
            graph_id=fusion_input.graph_id,
            metadata=self._metadata(),
        )

    def _metadata(self) -> Dict[str, Any]:
        return {"strategy": self.__class__.__name__}

    def __call__(self, fusion_input: FusionInput) -> FusionOutput:
        return self.fuse(fusion_input)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 1: WeightedSumFusion
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WeightedSumConfig:
    level1_weight: float = 0.4
    level2_weight: float = 0.6
    clip_logits:   bool  = True
    clip_range:    Tuple[float, float] = (-10.0, 10.0)


class WeightedSumFusion(BaseFusion):
    """
    가장 단순한 fusion 전략.

    최종 score = sigmoid(
        w1 * logit(level1_score) + w2 * logit(level2_score)
    )

    weight는 자동으로 정규화됩니다.
    """

    def __init__(self, cfg: WeightedSumConfig):
        self.cfg = cfg
        total = cfg.level1_weight + cfg.level2_weight
        if total <= 0:
            raise ValueError("level1_weight + level2_weight must be > 0")
        self.w1 = cfg.level1_weight / total
        self.w2 = cfg.level2_weight / total

    @staticmethod
    def _score_to_logit(score: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        score = score.clamp(eps, 1.0 - eps)
        return torch.log(score / (1.0 - score))

    def _combine(self, fusion_input: FusionInput) -> torch.Tensor:
        # score → logit 공간에서 합산
        l1_logit = self._score_to_logit(fusion_input.level1_score)
        l2_logit = self._score_to_logit(fusion_input.level2_score)

        combined = self.w1 * l1_logit + self.w2 * l2_logit

        if self.cfg.clip_logits:
            lo, hi = self.cfg.clip_range
            combined = combined.clamp(lo, hi)

        return combined

    def _metadata(self) -> Dict[str, Any]:
        return {
            "strategy": "WeightedSumFusion",
            "level1_weight": self.w1,
            "level2_weight": self.w2,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 2: CalibratedFusion
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CalibratedFusionConfig:
    level1_weight:      float = 0.4
    level2_weight:      float = 0.6
    level1_temperature: float = 1.0   # > 1 → softer, < 1 → sharper
    level2_temperature: float = 1.0
    level1_bias:        float = 0.0   # logit 공간 bias (보정용)
    level2_bias:        float = 0.0
    clip_logits:        bool  = True
    clip_range:         Tuple[float, float] = (-10.0, 10.0)


class CalibratedFusion(BaseFusion):
    """
    각 Level의 logit을 temperature scaling + bias로 보정한 후 합산.

    calibrated_logit_i = (raw_logit_i / T_i) + bias_i
    final_logit = w1 * calibrated_logit_1 + w2 * calibrated_logit_2

    temperature > 1 이면 score를 0.5 방향으로 끌어당김 (confidence 낮춤)
    temperature < 1 이면 score를 0/1 방향으로 밀어냄 (confidence 높임)
    bias > 0 이면 양성 방향 이동 (recall 증가, precision 감소)
    """

    def __init__(self, cfg: CalibratedFusionConfig):
        self.cfg = cfg
        total = cfg.level1_weight + cfg.level2_weight
        if total <= 0:
            raise ValueError("level1_weight + level2_weight must be > 0")
        self.w1 = cfg.level1_weight / total
        self.w2 = cfg.level2_weight / total

    @staticmethod
    def _score_to_logit(score: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        score = score.clamp(eps, 1.0 - eps)
        return torch.log(score / (1.0 - score))

    def _calibrate(
        self,
        score: torch.Tensor,
        temperature: float,
        bias: float,
    ) -> torch.Tensor:
        logit = self._score_to_logit(score)
        return (logit / max(temperature, 1e-8)) + bias

    def _combine(self, fusion_input: FusionInput) -> torch.Tensor:
        cal1 = self._calibrate(
            fusion_input.level1_score,
            self.cfg.level1_temperature,
            self.cfg.level1_bias,
        )
        cal2 = self._calibrate(
            fusion_input.level2_score,
            self.cfg.level2_temperature,
            self.cfg.level2_bias,
        )

        combined = self.w1 * cal1 + self.w2 * cal2

        if self.cfg.clip_logits:
            lo, hi = self.cfg.clip_range
            combined = combined.clamp(lo, hi)

        return combined

    def _metadata(self) -> Dict[str, Any]:
        return {
            "strategy":           "CalibratedFusion",
            "level1_weight":      self.w1,
            "level2_weight":      self.w2,
            "level1_temperature": self.cfg.level1_temperature,
            "level2_temperature": self.cfg.level2_temperature,
            "level1_bias":        self.cfg.level1_bias,
            "level2_bias":        self.cfg.level2_bias,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 3: LearnedFusion
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LearnedFusionConfig:
    hidden_dim:    int   = 32
    num_layers:    int   = 2
    dropout:       float = 0.1
    use_batchnorm: bool  = False
    # 입력 feature 선택
    use_logits:    bool  = True   # logit도 추가 feature로 사용할지
    # 초기화: level1_weight, level2_weight로 출력 layer 초기화
    init_level1_weight: float = 0.4
    init_level2_weight: float = 0.6


def _build_mlp(
    in_dim:   int,
    hidden_dim: int,
    out_dim:  int,
    num_layers: int,
    dropout:  float,
    use_batchnorm: bool,
) -> nn.Sequential:
    """
    Generic MLP builder.
    num_layers=1 → Linear(in_dim, out_dim) only
    num_layers>=2 → in_dim → hidden_dim × (num_layers-1) → out_dim
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    if num_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    layers: List[nn.Module] = []

    layers.append(nn.Linear(in_dim, hidden_dim))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.ELU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class LearnedFusionNet(nn.Module):
    """
    입력:
        - level1_score, level2_score (항상 포함)
        - level1_logit, level2_logit (cfg.use_logits=True일 때)

    추가 파생 feature:
        - score_diff    = level2_score - level1_score
        - score_product = level1_score * level2_score
        - score_max     = max(level1_score, level2_score)

    총 입력 차원:
        use_logits=True  → 2 scores + 2 logits + 3 derived = 7
        use_logits=False → 2 scores + 3 derived             = 5
    """

    def __init__(self, cfg: LearnedFusionConfig):
        super().__init__()
        self.cfg    = cfg
        self.in_dim = 16 if cfg.use_logits else 8

        self.net = _build_mlp(
            in_dim=self.in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=1,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_batchnorm=cfg.use_batchnorm,
        )
        self._init_weights()

    def _init_weights(self):
        """
        마지막 Linear를 weighted sum에 가까운 값으로 초기화해서
        학습 초기 안정성을 확보합니다.
        """
        last_linear = None
        for m in reversed(list(self.net.modules())):
            if isinstance(m, nn.Linear):
                last_linear = m
                break

        if last_linear is not None:
            nn.init.zeros_(last_linear.bias)
            # score index 0,1에 초기 weight 부여
            with torch.no_grad():
                w = last_linear.weight.data
                w.zero_()
                if w.size(1) >= 2:
                    w[0, 0] = self.cfg.init_level1_weight
                    w[0, 1] = self.cfg.init_level2_weight

    @staticmethod
    def _score_to_logit(score: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        score = score.clamp(eps, 1.0 - eps)
        return torch.log(score / (1.0 - score))

    def _build_feature(
        self,
        level1_score: torch.Tensor,
        level2_score: torch.Tensor,
        level1_logits: Optional[torch.Tensor],
        level2_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        s1 = level1_score.view(-1, 1)
        s2 = level2_score.view(-1, 1)

        diff    = (s2 - s1)
        product = (s1 * s2)
        s_max   = torch.max(s1, s2)

        if self.cfg.use_logits:
            l1 = (level1_logits.view(-1, 1)
                  if level1_logits is not None
                  else self._score_to_logit(level1_score).view(-1, 1))
            l2 = (level2_logits.view(-1, 1)
                  if level2_logits is not None
                  else self._score_to_logit(level2_score).view(-1, 1))
            feat = torch.cat([s1, s2, l1, l2, diff, product, s_max], dim=-1)
        else:
            feat = torch.cat([s1, s2, diff, product, s_max], dim=-1)

        return feat.float()

    def forward(
        self,
        level1_score:  torch.Tensor,
        level2_score:  torch.Tensor,
        level1_logits: Optional[torch.Tensor] = None,
        level2_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat   = self._build_feature(level1_score, level2_score, level1_logits, level2_logits)
        logits = self.net(feat)
        return logits.view(-1)


class LearnedFusion(BaseFusion):
    """
    학습형 Fusion 전략.
    LearnedFusionNet을 내장하며 train/eval 모드를 지원합니다.
    """

    def __init__(self, cfg: LearnedFusionConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or "cpu"
        self.net = LearnedFusionNet(cfg).to(self.device)
        self.net.eval() ## 기본적으로 eval 모드로 시작

    def _combine(self, fusion_input: FusionInput) -> torch.Tensor:
        device = next(self.net.parameters()).device

        l1s = fusion_input.level1_score.to(device)
        l2s = fusion_input.level2_score.to(device)
        l1l = fusion_input.level1_logits.to(device) if fusion_input.level1_logits is not None else None
        l2l = fusion_input.level2_logits.to(device) if fusion_input.level2_logits is not None else None

        return self.net(l1s, l2s, l1l, l2l)

    def train_mode(self):
        self.net.train()
        return self

    def eval_mode(self):
        self.net.eval()
        return self

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def _metadata(self) -> Dict[str, Any]:
        return {
            "strategy":    "LearnedFusion",
            "hidden_dim":  self.cfg.hidden_dim,
            "num_layers":  self.cfg.num_layers,
            "use_logits":  self.cfg.use_logits,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 4: FusionEnsemble
# ══════════════════════════════════════════════════════════════════════════════

AggregationMode = Literal["mean", "max", "vote"]


class FusionEnsemble(BaseFusion):
    """
    여러 BaseFusion 전략을 결합하는 앙상블.

    mode="mean"  → 각 전략의 score를 평균
    mode="max"   → 각 전략의 score를 최대값
    mode="vote"  → 각 전략의 binary 예측(threshold 기반)을 다수결
    """

    def __init__(
        self,
        strategies:   List[BaseFusion],
        mode:         AggregationMode = "mean",
        vote_threshold: float = 0.5,
    ):
        if len(strategies) == 0:
            raise ValueError("FusionEnsemble requires at least one strategy")
        self.strategies     = strategies
        self.mode           = mode
        self.vote_threshold = vote_threshold

    def _combine(self, fusion_input: FusionInput) -> torch.Tensor:
        """각 전략의 logit을 합산/선택."""
        logits_list = [s._combine(fusion_input) for s in self.strategies]
        stacked     = torch.stack(logits_list, dim=0)   # [n_strategies, N]

        if self.mode == "mean":
            return stacked.mean(dim=0)

        if self.mode == "max":
            return stacked.max(dim=0).values

        if self.mode == "vote":
            # 각 전략의 binary 예측을 평균내어 vote
            votes  = (torch.sigmoid(stacked) >= self.vote_threshold).float()
            mean_v = votes.mean(dim=0)
            # 0.5 이상이면 양성 logit (+3), 미만이면 음성 logit (-3)
            return torch.where(mean_v >= 0.5,
                               torch.full_like(mean_v, 3.0),
                               torch.full_like(mean_v, -3.0))

        raise ValueError(f"Unsupported ensemble mode: {self.mode}")

    def _metadata(self) -> Dict[str, Any]:
        return {
            "strategy":   "FusionEnsemble",
            "mode":       self.mode,
            "n_strategies": len(self.strategies),
            "sub_strategies": [s._metadata()["strategy"] for s in self.strategies],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d > 0 else 0.0


def compute_fusion_metrics(
    output: FusionOutput,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if output.label is None:
        raise ValueError("FusionOutput.label is required for metric computation")

    y_true = output.label.view(-1).float().detach().cpu()
    y_score = output.score.view(-1).float().detach().cpu()
    y_pred = (y_score >= threshold).long()
    y_t_long = y_true.long()

    tp = int(((y_pred == 1) & (y_t_long == 1)).sum().item())
    tn = int(((y_pred == 0) & (y_t_long == 0)).sum().item())
    fp = int(((y_pred == 1) & (y_t_long == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_t_long == 1)).sum().item())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, len(y_true))
    f1 = _safe_div(2 * precision * recall, precision + recall)

    roc_auc = 0.0
    pr_auc = 0.0
    bce = float(
        F.binary_cross_entropy(
            y_score.clamp(1e-6, 1 - 1e-6),
            y_true,
        ).item()
    )

    unique_classes = torch.unique(y_t_long)

    # ROC AUC: 두 클래스가 모두 있어야 정의 가능
    ## sklearn.metrics.roc_auc_score은 양성/음성 클래스가 모두 존재하지 않으면 ValueError를 발생시키므로, 사전에 체크하여 예외 처리합니다.
    if unique_classes.numel() >= 2:
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = float(roc_auc_score(y_true.numpy(), y_score.numpy()))
        except Exception:
            roc_auc = 0.0

    # PR AUC: positive class가 하나 이상 있어야 의미 있음
    ## sklearn.metrics.average_precision_score은 양성 클래스가 하나도 없으면 ValueError를 발생시키므로, 사전에 체크하여 예외 처리합니다.
    if int((y_t_long == 1).sum().item()) > 0:
        try:
            from sklearn.metrics import average_precision_score
            pr_auc = float(average_precision_score(y_true.numpy(), y_score.numpy()))
        except Exception:
            pr_auc = 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "bce_loss": bce,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }





# ══════════════════════════════════════════════════════════════════════════════
# FusionTrainer  (LearnedFusion 전용)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FusionTrainerConfig:
    lr:                float = 1e-3
    weight_decay:      float = 1e-4
    epochs:            int   = 20
    batch_size:        int   = 64
    max_grad_norm:     Optional[float] = 1.0
    pos_weight:        Optional[float] = None
    early_stopping_patience: int = 5
    val_metric:        str   = "pr_auc"    # "pr_auc" | "f1" | "roc_auc"


class FusionTrainer:
    """
    LearnedFusion을 학습시키는 트레이너.

    학습 데이터는 FusionInput 형태로 주어집니다.
    Label이 없으면 학습할 수 없습니다.
    """

    def __init__(
        self,
        fusion:  LearnedFusion,
        cfg:     FusionTrainerConfig,
    ):
        self.fusion    = fusion
        self.cfg       = cfg
        self.device    = fusion.device

        pos_weight = None
        if cfg.pos_weight is not None:
            pos_weight = torch.tensor(
                [cfg.pos_weight], dtype=torch.float32, device=self.device
            )
        self.loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.AdamW(
            fusion.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    def _split_batches(
        self,
        fusion_input: FusionInput,
    ):
        """FusionInput을 batch_size 단위로 분할."""
        n  = fusion_input.n
        bs = self.cfg.batch_size

        indices = torch.randperm(n)
        for start in range(0, n, bs):
            idx = indices[start: start + bs]

            l1s = fusion_input.level1_score[idx].to(self.device)
            l2s = fusion_input.level2_score[idx].to(self.device)
            lbl = fusion_input.label[idx].to(self.device)

            l1l = fusion_input.level1_logits[idx].to(self.device) \
                if fusion_input.level1_logits is not None else None
            l2l = fusion_input.level2_logits[idx].to(self.device) \
                if fusion_input.level2_logits is not None else None

            yield l1s, l2s, l1l, l2l, lbl

    def _train_epoch(self, train_input: FusionInput) -> float:
        self.fusion.train_mode()

        total_loss = 0.0
        n_batches  = 0

        for l1s, l2s, l1l, l2l, lbl in self._split_batches(train_input):
            self.optimizer.zero_grad(set_to_none=True)

            logits = self.fusion.net(l1s, l2s, l1l, l2l)
            loss   = self.loss_fn(logits.view(-1, 1), lbl.view(-1, 1))
            loss.backward()

            if self.cfg.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.fusion.parameters(), self.cfg.max_grad_norm
                )

            self.optimizer.step()
            total_loss += float(loss.item())
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval(self, fusion_input: FusionInput) -> Dict[str, float]:
        self.fusion.eval_mode()
        out = self.fusion.fuse(fusion_input)
        return compute_fusion_metrics(out)

    def _select_monitor(self, metrics: Dict[str, float]) -> float:
        return float(metrics.get(self.cfg.val_metric, metrics.get("pr_auc", 0.0)))

    def fit(
        self,
        train_input: FusionInput,
        valid_input: FusionInput,
        verbose:     bool = True,
    ) -> Dict[str, Any]:
        if train_input.label is None:
            raise ValueError("train_input must contain labels")
        if valid_input.label is None:
            raise ValueError("valid_input must contain labels")

        best_score   = float("-inf")
        best_state   = None
        patience_cnt = 0
        history      = []

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss   = self._train_epoch(train_input)
            valid_metrics = self._eval(valid_input)
            monitor       = self._select_monitor(valid_metrics)

            history.append({
                "epoch":         epoch,
                "train_loss":    train_loss,
                "valid_metrics": valid_metrics,
            })

            if monitor > best_score:
                best_score   = monitor
                best_state   = {
                    k: v.clone() for k, v in self.fusion.state_dict().items()
                }
                patience_cnt = 0
            else:
                patience_cnt += 1

            if verbose:
                print(
                    f"[FusionTrainer Epoch {epoch:03d}] "
                    f"train_loss={train_loss:.4f}  "
                    f"valid_{self.cfg.val_metric}="
                    f"{monitor:.4f}  "
                    f"(patience={patience_cnt}/{self.cfg.early_stopping_patience})"
                )

            if patience_cnt >= self.cfg.early_stopping_patience:
                if verbose:
                    print(f"  → Early stopping at epoch {epoch}")
                break

        # best weight 복원
        if best_state is not None:
            self.fusion.load_state_dict(best_state)

        self.fusion.eval_mode()

        return {
            "history":     history,
            "best_score":  best_score,
            "best_metric": self.cfg.val_metric,
            "best_mode":   "max",  # Assuming PR-AUC/F1 are max metrics
            "epochs_ran":  len(history),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Save / Load
# ══════════════════════════════════════════════════════════════════════════════

def save_learned_fusion(
    fusion: LearnedFusion,
    path:   str,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": fusion.state_dict(),
            "config":     asdict(fusion.cfg),
        },
        path,
    )
    return str(path)


def load_learned_fusion(
    path:   str,
    device: Optional[str] = None,
) -> LearnedFusion:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg        = LearnedFusionConfig(**checkpoint["config"])
    fusion     = LearnedFusion(cfg=cfg, device=device or "cpu")
    fusion.load_state_dict(checkpoint["state_dict"])
    fusion.eval_mode()
    return fusion


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

FusionStrategy = Literal["weighted_sum", "calibrated", "learned", "ensemble"]


def build_fusion(
    strategy: FusionStrategy,
    **kwargs,
) -> BaseFusion:
    """
    Config dict 기반 Fusion 전략 팩토리.

    Examples:
        build_fusion("weighted_sum", level1_weight=0.3, level2_weight=0.7)
        build_fusion("calibrated",   level1_temperature=1.5)
        build_fusion("learned",      hidden_dim=64, num_layers=3)
        build_fusion(
            "ensemble",
            strategies=[
                build_fusion("weighted_sum"),
                build_fusion("calibrated"),
            ],
            mode="mean",
        )
    """
    if strategy == "weighted_sum":
        cfg = WeightedSumConfig(**{
            k: v for k, v in kwargs.items()
            if k in WeightedSumConfig.__dataclass_fields__
        })
        return WeightedSumFusion(cfg)

    if strategy == "calibrated":
        cfg = CalibratedFusionConfig(**{
            k: v for k, v in kwargs.items()
            if k in CalibratedFusionConfig.__dataclass_fields__
        })
        return CalibratedFusion(cfg)

    if strategy == "learned":
        cfg = LearnedFusionConfig(**{
            k: v for k, v in kwargs.items()
            if k in LearnedFusionConfig.__dataclass_fields__
        })
        device = kwargs.get("device", None)
        return LearnedFusion(cfg=cfg, device=device)

    if strategy == "ensemble":
        strategies = kwargs.get("strategies", [])
        mode       = kwargs.get("mode", "mean")
        threshold  = kwargs.get("vote_threshold", 0.5)
        return FusionEnsemble(
            strategies=strategies,
            mode=mode,
            vote_threshold=threshold,
        )

    raise ValueError(
        f"Unknown fusion strategy: '{strategy}'. "
        f"Choose from: weighted_sum, calibrated, learned, ensemble"
    )
