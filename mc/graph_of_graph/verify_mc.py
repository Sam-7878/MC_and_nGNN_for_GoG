# 빠른 검증: verify_mc.py
import torch
import numpy as np
from pygod.detector import DOMINANT
from torch_geometric.data import Data

def verify_mc_dropout():
    print("🔬 MC Dropout 검증 시작\n")

    # 작은 dummy 데이터
    x          = torch.randn(100, 7)
    edge_index = torch.randint(0, 100, (2, 300))
    y          = torch.zeros(100, dtype=torch.long)
    y[:5]      = 1
    data = Data(x=x, edge_index=edge_index, y=y)

    detector = DOMINANT(hid_dim=8, lr=0.01, epoch=30, gpu=-1)
    detector.fit(data)

    print("📌 방법 A: 기존 방식 (decision_function 직접 호출)")
    scores_old = [detector.decision_function(data) for _ in range(10)]
    scores_old = np.stack(scores_old)
    print(f"  → std (불확실성): {scores_old.std(axis=0).mean():.6f}")

    print("\n📌 방법 B: model.eval() 무력화 후 호출")
    original_eval = detector.model.eval
    detector.model.eval = lambda: detector.model
    detector.model.train()

    scores_new = []
    try:
        for _ in range(10):
            s = detector.decision_function(data)
            scores_new.append(s)
    finally:
        detector.model.eval = original_eval
        detector.model.eval()

    scores_new = np.stack(scores_new)
    std_new    = scores_new.std(axis=0).mean()
    print(f"  → std (불확실성): {std_new:.6f}")

    if std_new > 1e-6:
        print("  ✅ MC Dropout 정상 작동!")
    else:
        print("  ❌ 여전히 0: 모델 자체에 Dropout이 없는 구조")
        print("     → 해결책 A (직접 forward pass) 적용 필요")

if __name__ == "__main__":
    verify_mc_dropout()
