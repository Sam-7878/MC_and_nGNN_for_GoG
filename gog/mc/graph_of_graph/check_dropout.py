# 진단 스크립트: check_dropout.py
import torch
import numpy as np
from pygod.detector import DOMINANT, DONE, GAE, AnomalyDAE, CoLA
from torch_geometric.data import Data

def inspect_model_dropout(model_name, model_class):
    print(f"\n{'='*60}")
    print(f"🔍 {model_name} 구조 분석")
    print(f"{'='*60}")

    detector = model_class(hid_dim=8, lr=0.01, epoch=5, gpu=-1)

    # dummy data로 fit (구조 초기화 필요)
    x = torch.randn(50, 7)
    edge_index = torch.randint(0, 50, (2, 100))
    y = torch.zeros(50, dtype=torch.long)
    y[:3] = 1
    data = Data(x=x, edge_index=edge_index, y=y)
    detector.fit(data)

    print(f"\n📌 detector.model 존재 여부: {hasattr(detector, 'model')}")

    if not hasattr(detector, 'model'):
        print("  ❌ detector.model 없음 → MC Dropout 적용 불가")
        return

    print(f"\n📌 전체 모듈 목록:")
    dropout_found = False
    for name, module in detector.model.named_modules():
        module_type = module.__class__.__name__
        print(f"  [{name}] {module_type}")
        if module_type.startswith('Dropout'):
            dropout_found = True
            print(f"    ✅ Dropout 발견! p={module.p}")

    if not dropout_found:
        print(f"\n  ❌ {model_name}에 Dropout 레이어 없음!")
    else:
        print(f"\n  ✅ Dropout 존재 확인")

    # decision_function 반복 호출 시 분산 확인
    print(f"\n📌 eval 모드에서 decision_function 10회 반복:")
    scores = []
    for i in range(10):
        s = detector.decision_function(data)
        scores.append(s)
    scores = np.stack(scores)
    std_eval = scores.std(axis=0).mean()
    print(f"  eval 모드 std (=불확실성): {std_eval:.6f}")
    if std_eval == 0.0:
        print(f"  → ❌ 완전히 deterministic (Dropout 비활성화 상태)")

    # train 모드 강제 후 확인
    print(f"\n📌 강제 train 모드에서 decision_function 10회 반복:")
    if hasattr(detector, 'model'):
        for module in detector.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

    scores_train = []
    for i in range(10):
        s = detector.decision_function(data)
        scores_train.append(s)
    scores_train = np.stack(scores_train)
    std_train = scores_train.std(axis=0).mean()
    print(f"  train 모드 강제 후 std: {std_train:.6f}")
    if std_train == 0.0:
        print(f"  → ❌ 여전히 0: Dropout이 forward pass에 영향 없음")
    else:
        print(f"  → ✅ std > 0: Dropout이 정상 작동")


models = {
    'DOMINANT':   DOMINANT,
    'DONE':       DONE,
    'GAE':        GAE,
    'AnomalyDAE': AnomalyDAE,
    'CoLA':       CoLA,
}

for name, cls in models.items():
    inspect_model_dropout(name, cls)
