import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    f1_score, 
    roc_auc_score, 
    accuracy_score,
    classification_report
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def safe_collate_fn(batch):
    """Safely collate batch, filtering out None values"""
    # Filter out None values
    batch = [b for b in batch if b is not None]
    
    # Handle empty batch
    if not batch:
        print("⚠️  Warning: Empty batch after filtering None")
        return None
    
    # Use PyG's Batch.from_data_list
    try:
        return Batch.from_data_list(batch)
    except Exception as e:
        print(f"⚠️  Batch collation failed: {str(e)}")
        return None



class MCGraphDataset(Dataset):
    """Graph dataset with Monte Carlo sampling support"""
    
    def __init__(self, data_dir, split, mc_samples=1, training=False, scaler=None):
        self.data_dir = data_dir
        self.split = split
        self.mc_samples = mc_samples
        self.training = training
        self.scaler = scaler
        
        # Load split information
        labels_df = pd.read_csv(os.path.join(data_dir, 'labels_split.csv'))
        self.labels_df = labels_df  # ✅ FIXED: Save labels_df
        self.graph_ids = labels_df[labels_df['Split'] == split].index.tolist()
        
        print(f"Loaded {len(self.graph_ids)} graphs (MC={mc_samples}, Training={training})")
        
        # Fit scaler on training data
        if self.scaler is None and split == 'train':
            print(f"Fitting RobustScaler on training data...")
            all_features = []
            
            sample_size = min(100, len(self.graph_ids))
            for gid in self.graph_ids[:sample_size]:
                filepath = os.path.join(self.data_dir, f'{gid}.json')
                
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    features_dict = data['features']
                    node_ids = sorted([int(k) for k in features_dict.keys()])
                    features = np.array([features_dict[str(i)] for i in node_ids], dtype=np.float32)
                    
                    # Clip extreme values
                    features = np.clip(features, -100, 100)
                    features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
                    
                    all_features.append(features)
                    
                except Exception as e:
                    continue
            
            if all_features:
                all_features = np.vstack(all_features)
                
                self.scaler = RobustScaler()
                self.scaler.fit(all_features)
                
                print(f"✅ Feature normalization fitted (RobustScaler):")
                print(f"   Median (first 5): {self.scaler.center_[:5]}")
                print(f"   IQR (first 5): {self.scaler.scale_[:5]}")
            else:
                print(f"⚠️  Warning: No valid features for scaler fitting")
                self.scaler = None
    
    def _load_graph(self, graph_id):
        """Load and preprocess a single graph"""
        filepath = os.path.join(self.data_dir, f'{graph_id}.json')
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            features_dict = data['features']
            node_ids = sorted([int(k) for k in features_dict.keys()])
            features = np.array([features_dict[str(i)] for i in node_ids], dtype=np.float32)
            
            # Clip and clean
            features = np.clip(features, -100, 100)
            
            if np.isnan(features).any() or np.isinf(features).any():
                print(f"⚠️  Cleaning NaN/Inf in graph {graph_id}")
                features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # Apply scaler
            if self.scaler is not None:
                try:
                    features = self.scaler.transform(features)
                    features = np.clip(features, -10, 10)
                    
                    if np.isnan(features).any() or np.isinf(features).any():
                        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
                        
                except Exception as e:
                    print(f"⚠️  Scaler transform failed for graph {graph_id}: {str(e)}")
                    feat_min = features.min(axis=0)
                    feat_max = features.max(axis=0)
                    feat_range = feat_max - feat_min
                    feat_range[feat_range == 0] = 1.0
                    features = (features - feat_min) / feat_range
                    features = features * 2 - 1
            
            x = torch.tensor(features, dtype=torch.float)
            edges = data['edges']
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            y = torch.tensor([data['label']], dtype=torch.long)
            
            return Data(x=x, edge_index=edge_index, y=y)
            
        except Exception as e:
            print(f"⚠️  Error loading graph {graph_id}: {str(e)}")
            return None
    
    def __len__(self):
        return len(self.graph_ids)
    
    def __getitem__(self, idx):
        max_retries = 5
        
        for retry in range(max_retries):
            try:
                graph_id = self.graph_ids[(idx + retry) % len(self.graph_ids)]
                
                if self.training and self.mc_samples > 1:
                    graphs = []
                    for _ in range(self.mc_samples):
                        graph = self._load_graph(graph_id)
                        if graph is not None:
                            graphs.append(graph)
                    
                    if graphs:
                        return graphs
                else:
                    graph = self._load_graph(graph_id)
                    if graph is not None:
                        return graph
                        
            except Exception as e:
                continue
        
        return self._load_graph(self.graph_ids[0])


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
        
        # Handle NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Graph convolutions
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
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x

# ===================== MC Trainer (안정화) =====================
class MCTrainer:
    def __init__(self, model, device, mc_samples_train=1, mc_samples_eval=10, class_weights=None):
        self.model = model
        self.device = device
        self.mc_samples_train = mc_samples_train
        self.mc_samples_eval = mc_samples_eval
        self.class_weights = class_weights
    
    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(loader, desc="Training", leave=False):
            if batch is None:
                continue
                
            batch = batch.to(self.device)
            
            # Safety check
            if batch.x is None or torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                print(f"⚠️  Skipping batch with invalid features")
                continue
            
            optimizer.zero_grad()
            
            try:
                logits = self.model(batch)
                loss = criterion(logits, batch.y)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Skipping batch with NaN/Inf loss")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item() * batch.num_graphs
                pred = logits.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.num_graphs
                
            except Exception as e:
                print(f"⚠️  Error in training batch: {str(e)}")
                continue
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def evaluate(self, loader, criterion):
        """Evaluate with MC dropout for uncertainty estimation"""
        self.model.train()  # ✅ Keep dropout active for MC sampling
        
        all_preds = []
        all_probs = []
        all_labels = []
        all_uncertainties = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                if batch is None:
                    continue
                    
                batch = batch.to(self.device)
                
                # Safety check
                if batch.x is None or torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                    continue
                
                try:
                    # MC sampling
                    mc_logits = []
                    for _ in range(self.mc_samples_eval):
                        logits = self.model(batch)
                        mc_logits.append(F.softmax(logits, dim=1))
                    
                    # Aggregate predictions
                    mc_probs = torch.stack(mc_logits, dim=0)  # [mc_samples, batch_size, n_classes]
                    mean_probs = mc_probs.mean(dim=0)
                    std_probs = mc_probs.std(dim=0)
                    
                    # Uncertainty (mean of std across classes)
                    uncertainty = std_probs.mean(dim=1)
                    
                    # Prediction
                    pred = mean_probs.argmax(dim=1)
                    
                    # Loss
                    loss = criterion(mean_probs.log(), batch.y)
                    total_loss += loss.item() * batch.num_graphs
                    
                    # Collect results
                    all_preds.extend(pred.cpu().numpy())
                    all_probs.extend(mean_probs.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
                    all_uncertainties.extend(uncertainty.cpu().numpy())
                    
                except Exception as e:
                    print(f"⚠️  Error in evaluation batch: {str(e)}")
                    continue
        
        if not all_labels:
            print("⚠️  No valid batches in evaluation!")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'uncertainty': 0.0
            }
        
        # Convert to numpy
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_uncertainties = np.array(all_uncertainties)
        
        # Metrics
        acc = accuracy_score(all_labels, all_preds)  # ✅ 이제 정의됨!
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # AUC (multi-class)
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        avg_loss = total_loss / len(all_labels)
        avg_uncertainty = all_uncertainties.mean()
        
        return {
            'loss': avg_loss,
            'accuracy': acc,
            'f1': f1,
            'auc': auc,
            'uncertainty': avg_uncertainty
        }


# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser(description="MC-GoG Training with Enhanced Features")
    parser.add_argument('--chain', type=str, required=True)
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--mc_train', type=int, default=1)
    parser.add_argument('--mc_eval', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Stabilized MC Training")
    print(f"Device: {device}, MC Train/Eval: {args.mc_train}/{args.mc_eval}")
    
    data_dir = f'../../_data/GoG/{args.chain}'
    
    # Clean up old/incompatible checkpoints
    model_save_path = os.path.join(data_dir, f'best_model_{args.chain}.pt')
    old_model_paths = [
        os.path.join(data_dir, 'best_mc_model.pth'),
        os.path.join(data_dir, 'best_model.pth'),
    ]
    
    input_dim = 24
    
    for old_path in old_model_paths:
        if os.path.exists(old_path):
            print(f"🗑️  Removing old checkpoint: {old_path}")
            os.remove(old_path)
    
    if os.path.exists(model_save_path):
        try:
            checkpoint = torch.load(model_save_path, map_location='cpu')
            saved_input_dim = checkpoint.get('input_dim', 9)
            
            if saved_input_dim != input_dim:
                print(f"⚠️  Incompatible checkpoint (input_dim={saved_input_dim} vs {input_dim})")
                print(f"🗑️  Removing incompatible checkpoint...")
                os.remove(model_save_path)
        except Exception as e:
            print(f"⚠️  Corrupted checkpoint. Removing...")
            os.remove(model_save_path)
    
    # Datasets
    train_dataset = MCGraphDataset(data_dir, 'train', mc_samples=1, training=True)
    val_dataset = MCGraphDataset(data_dir, 'val', mc_samples=1, training=False, scaler=train_dataset.scaler)
    test_dataset = MCGraphDataset(data_dir, 'test', mc_samples=1, training=False, scaler=train_dataset.scaler)
    
    print(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # DataLoaders
    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    # ✅ FIXED: Use 'Category' instead of 'Label'
    print(f"📊 CSV columns: {train_dataset.labels_df.columns.tolist()}")
    
    class_counts = train_dataset.labels_df['Category'].value_counts().sort_index()
    total_samples = len(train_dataset.labels_df)
    
    print(f"📊 Class distribution: {class_counts.to_dict()}")
    
    # Calculate class weights
    class_weights = []
    for i in range(args.n_classes):
        if i in class_counts.index:
            weight = total_samples / (args.n_classes * class_counts[i])
        else:
            print(f"⚠️  Warning: Class {i} not in training set, using weight=1.0")
            weight = 1.0
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Model
    model = GoGMCModel(input_dim, args.hidden_dim, args.n_classes, dropout=args.dropout)
    model = model.to(device)
    
    print(f"✅ Model: input_dim={input_dim}, hidden={args.hidden_dim}, classes={args.n_classes}")
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    trainer = MCTrainer(model, device, mc_samples_train=args.mc_train, 
                        mc_samples_eval=args.mc_eval, class_weights=class_weights)
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    patience = 20
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        
        # Validate
        val_metrics = trainer.evaluate(val_loader, criterion)
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2%}, "
              f"F1: {val_metrics['f1']:.2%}, "
              f"AUC: {val_metrics['auc']:.2%}, "
              f"Uncertainty: {val_metrics['uncertainty']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])
        
        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'scaler': train_dataset.scaler,
                'input_dim': input_dim,
                'hidden_dim': args.hidden_dim,
                'n_classes': args.n_classes,
                'dropout': args.dropout
            }, model_save_path)
            
            print(f"✅ New best F1: {best_val_f1:.2%}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"\n✅ Training completed! Best Val F1: {best_val_f1:.2%}")
    
    # Final Test Evaluation
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
    print(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Test F1 Score: {test_metrics['f1']:.2%}")
    print(f"Test AUC: {test_metrics['auc']:.2%}")
    print(f"Average Uncertainty: {test_metrics['uncertainty']:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()

