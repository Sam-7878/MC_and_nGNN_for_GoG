import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# =======================================================================
# 1. Dataset Loader (individual.py가 생성한 .pt 파일 로드)
# =======================================================================
class TransactionDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)
        # .pt 파일에서 Data 객체 리스트와 slice 정보를 로드
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return f'{self.split}_data.pt'

# =======================================================================
# 2. Model Architecture
# =======================================================================
class GoGMCModel(torch.nn.Module):
    """Graph of Graphs model with Monte Carlo Dropout"""
    
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x

# =======================================================================
# 3. MC Trainer
# =======================================================================
class MCTrainer:
    def __init__(self, model, device, mc_samples_eval=10):
        self.model = model
        self.device = device
        self.mc_samples_eval = mc_samples_eval
    
    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch in tqdm(loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            logits = self.model(batch)
            loss = criterion(logits, batch.y.view(-1)) # Shape 맞춤
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y.view(-1)).sum().item()
            total += batch.num_graphs
                
        return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0
    
    def evaluate(self, loader, criterion):
        self.model.train()  # Model-level MC Sampling을 위해 Dropout 켜기
        
        all_preds, all_probs, all_labels, all_uncertainties = [], [], [], []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                batch = batch.to(self.device)
                
                # Model-level MC Sampling
                mc_logits = []
                for _ in range(self.mc_samples_eval):
                    logits = self.model(batch)
                    mc_logits.append(F.softmax(logits, dim=1))
                
                mc_probs = torch.stack(mc_logits, dim=0)
                mean_probs = mc_probs.mean(dim=0)
                std_probs = mc_probs.std(dim=0)
                
                uncertainty = std_probs.mean(dim=1)
                pred = mean_probs.argmax(dim=1)
                loss = criterion(mean_probs.log(), batch.y.view(-1))
                
                total_loss += loss.item() * batch.num_graphs
                all_preds.extend(pred.cpu().numpy())
                all_probs.extend(mean_probs.cpu().numpy())
                all_labels.extend(batch.y.view(-1).cpu().numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy())
                
        metrics = {
            'loss': total_loss / len(all_labels),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='macro'),
            'uncertainty': np.mean(all_uncertainties)
        }
        
        try:
            probs_array = np.array(all_probs)
            if probs_array.shape[1] == 2:
                metrics['auc'] = roc_auc_score(all_labels, probs_array[:, 1])
            else:
                metrics['auc'] = roc_auc_score(all_labels, probs_array, multi_class='ovr')
        except ValueError:
            metrics['auc'] = 0.0
            
        return metrics

# =======================================================================
# 4. Main Routine
# =======================================================================
def main():
    parser = argparse.ArgumentParser(description="MC-GoG Training with Pre-processed .pt files")
    parser.add_argument('--chain', type=str, required=True)
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--mc_eval', type=int, default=10) # Eval용 MC 샘플링 횟수
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Lightning Fast PyG Native Training")
    print(f"Device: {device}")

    # 데이터셋 경로 (individual.py에서 생성한 경로)
    data_root = f'../../_data/dataset/GCN/{args.chain}'
    
    print("📂 Loading datasets...")
    train_dataset = TransactionDataset(root=f'{data_root}/train', split='train')
    val_dataset = TransactionDataset(root=f'{data_root}/val', split='val')
    test_dataset = TransactionDataset(root=f'{data_root}/test', split='test')
    
    print(f"✅ Loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test graphs")

    # ---------------------------------------------------------
    # Robust Scaler 고속 적용 (단일 행렬 연산)
    # ---------------------------------------------------------
    print("📊 Applying RobustScaler to node features...")
    scaler = RobustScaler()
    train_x = train_dataset.data.x.numpy()
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler.fit(train_x)

    def scale_dataset(dataset):
        x = dataset.data.x.numpy()
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x_scaled = np.clip(scaler.transform(x), -10, 10)
        dataset.data.x = torch.tensor(x_scaled, dtype=torch.float32)
        return dataset

    train_dataset = scale_dataset(train_dataset)
    val_dataset = scale_dataset(val_dataset)
    test_dataset = scale_dataset(test_dataset)

    # ---------------------------------------------------------
    # 동적 차원 할당 및 모델 초기화
    # ---------------------------------------------------------
    input_dim = train_dataset.num_node_features
    print(f"🧠 Detected node features dimension: {input_dim}")

    model = GoGMCModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.n_classes,
        dropout=args.dropout
    ).to(device)

    # ---------------------------------------------------------
    # Class Weights 자동 계산
    # ---------------------------------------------------------
    y_train = train_dataset.data.y.view(-1)
    class_counts = torch.bincount(y_train, minlength=args.n_classes)
    total_samples = len(y_train)
    print(f"📈 Train Class Distribution: {class_counts.tolist()}")
    
    class_weights = total_samples / (args.n_classes * class_counts.float())
    class_weights = torch.nan_to_num(class_weights, nan=1.0, posinf=1.0).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = MCTrainer(model, device, mc_samples_eval=args.mc_eval)
    
    # 학습 루프
    best_val_f1 = 0
    patience_counter = 0
    patience_limit = 20
    
    model_save_path = f'../../_data/GoG/{args.chain}/best_model_pyg_{args.chain}.pt'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("\n🔥 Starting Training Loop...")
    for epoch in range(args.epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
        val_metrics = trainer.evaluate(val_loader, criterion)
        
        scheduler.step(val_metrics['f1'])
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}, "
              f"Uncertainty: {val_metrics['uncertainty']:.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model_state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'input_dim': input_dim
            }, model_save_path)
            print("  🌟 Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break

    print(f"\n✅ Training completed! Best Val F1: {best_val_f1:.2%}")
    
    # 최종 테스트 평가
    print("\n🎯 Final Test Evaluation")
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded best model from epoch {checkpoint['epoch']}")
    
    test_metrics = trainer.evaluate(test_loader, criterion)
    
    print("\n" + "="*50)
    print("📊 FINAL TEST RESULTS")
    print("="*50)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test ROC AUC: {test_metrics['auc']:.4f}")
    print(f"Mean Uncertainty: {test_metrics['uncertainty']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()