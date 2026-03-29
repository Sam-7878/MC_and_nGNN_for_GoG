# Hierarchical GNN for Contract Fraud Detection

계약 내부 **로컬 트랜잭션 그래프**와 계약 간 **글로벌 관계 그래프**를 함께 활용하여 스마트 컨트랙트 이상 탐지 / fraud detection을 수행하는 **계층형 GNN 파이프라인**입니다.

본 프로젝트는 다음 목표를 가집니다.

- contract 내부 transaction structure를 local graph로 인코딩
- contract 간 relation을 global graph로 인코딩
- local + global representation을 결합하여 fraud 여부 분류
- signed-log 기반 수치 안정화, class imbalance 대응, checkpoint 관리, ablation 실험, uncertainty estimation까지 포함한 실험 파이프라인 제공

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Key Features](#2-key-features)
- [3. Project Structure](#3-project-structure)
- [4. Requirements](#4-requirements)
- [5. Installation](#5-installation)
- [6. Data Format](#6-data-format)
- [7. Configuration](#7-configuration)
- [8. Training](#8-training)
- [9. Inference / Evaluation](#9-inference--evaluation)
- [10. MC-Dropout Uncertainty](#10-mc-dropout-uncertainty)
- [11. Ablation Experiments](#11-ablation-experiments)
- [12. Output Artifacts](#12-output-artifacts)
- [13. Recommended Experiment Flow](#13-recommended-experiment-flow)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Experiment Checklist](#15-experiment-checklist)
- [16. Citation / Notes](#16-citation--notes)

---

# 1. Overview

이 프로젝트는 **Hierarchical GNN** 구조를 사용합니다.

전체 흐름은 다음과 같습니다.

1. 각 contract의 local transaction graph를 읽음
2. local graph encoder로 contract-level embedding 생성
3. contract 간 global graph 상에서 global encoder 수행
4. local embedding과 global embedding을 결합
5. classifier로 fraud / non-fraud 이진 분류 수행

핵심 아이디어는 다음과 같습니다.

- **Local graph**는 개별 contract 내부 패턴을 잘 반영
- **Global graph**는 contract 간 상호작용 및 구조적 맥락을 반영
- 두 표현을 결합하면 local-only baseline보다 더 풍부한 판단 가능

---

# 2. Key Features

- JSON 기반 local contract graph 로딩
- `.pt` 기반 global contract graph 로딩
- NaN / Inf 안정화 전처리
- signed-log transform 지원
- class imbalance 대응용 class weights
- early stopping
- best checkpoint 자동 저장 / 로드
- Accuracy, Precision, Recall, F1, AUROC, AUPRC, Specificity 계산
- per-sample prediction CSV 저장
- MC-Dropout 기반 uncertainty estimation
- ablation 실험 자동화

---

# 3. Project Structure

```text
project_root/
├── config/
│   ├── config.yaml
│   └── ablations.yaml
│
├── data/
│   ├── dataset.py
│   └── collate.py
│
├── engine/
│   ├── trainer.py
│   └── evaluator.py
│
├── model/
│   ├── local_encoder.py
│   ├── pooling.py
│   ├── global_encoder.py
│   ├── classifier.py
│   └── hierarchical_gnn.py
│
├── utils/
│   └── metrics.py
│
├── outputs/
│   └── ...
│
├── train.py
├── infer.py
├── train_ablation.py
├── README.md
└── requirements.txt
