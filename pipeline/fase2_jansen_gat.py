#!/usr/bin/env python3
"""
fase2_jansen_gat.py — GATv2Conv ensemble para predecir errores de trayectoria.

Input:  grafo del mecanismo (8 nodos, 24 edges) + 13 global features (MN)
Output: 5 targets (foot_err, knee_err, hip_err, stride_m, ground_ratio)

Uso: python3 fase2_jansen_gat.py [--dataset data/jansen_dataset.pt]
"""
import os, sys, argparse, pickle, logging, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from config_jansen import (
    N_NODE_FEATS, N_EDGE_FEATS, N_GLOBAL_FEATS, N_TARGETS,
    N_ENSEMBLE, MAX_EPOCHS, PATIENCE, LR, BATCH_SIZE, DROPOUT,
    MODEL_DIR, DEVICE
)

os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "12.0.1")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("fase2_jansen")

torch.manual_seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO
# ═══════════════════════════════════════════════════════════════════════════════

class JansenGAT(nn.Module):
    """GATv2 para el mecanismo de Jansen (8 nodos, 12 barras)."""

    def __init__(self, in_ch=N_NODE_FEATS, hidden=128, heads=4,
                 edge_dim=N_EDGE_FEATS, n_global=N_GLOBAL_FEATS,
                 out_ch=N_TARGETS, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GATv2Conv(in_ch, hidden, heads=heads,
                               edge_dim=edge_dim, concat=True)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=2,
                               edge_dim=edge_dim, concat=True)
        self.conv3 = GATv2Conv(hidden * 2, 64, heads=1,
                               edge_dim=edge_dim, concat=False)
        self.drop = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(64 + n_global, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_ch)
        )
        self.n_global = n_global

    def forward(self, x, edge_index, edge_attr, batch, global_feats=None,
                use_dropout=True):
        x = F.gelu(self.conv1(x, edge_index, edge_attr))
        if use_dropout:
            x = self.drop(x)
        x = F.gelu(self.conv2(x, edge_index, edge_attr))
        if use_dropout:
            x = self.drop(x)
        x = F.gelu(self.conv3(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)

        if global_feats is not None:
            x = torch.cat([x, global_feats], dim=-1)
        else:
            x = torch.cat([x, torch.zeros(x.size(0), self.n_global,
                                           device=x.device)], dim=-1)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class JansenPhysicsLoss(nn.Module):
    """Pérdida de física para el mecanismo de Jansen."""
    def forward(self, pred, global_feats):
        foot_err = pred[:, 0]
        knee_err = pred[:, 1]
        stride = pred[:, 3]
        ground = pred[:, 4]

        losses = {}
        # Errores deben ser positivos
        losses['err_positive'] = torch.mean(torch.relu(-foot_err) + torch.relu(-knee_err))
        # Stride debe ser positivo
        losses['stride_positive'] = torch.mean(torch.relu(-stride))
        # Ground ratio entre 0 y 1
        losses['ground_bounds'] = torch.mean(torch.relu(-ground) + torch.relu(ground - 1.0))
        # Foot error y knee error correlacionados (si uno es alto, el otro también)
        losses['err_correlation'] = torch.mean(torch.relu(foot_err - knee_err - 2.0))

        total = (1.0 * losses['err_positive'] +
                 0.5 * losses['stride_positive'] +
                 0.3 * losses['ground_bounds'] +
                 0.1 * losses['err_correlation'])
        losses['total'] = total
        return losses


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def train_ensemble(dataset, scalers_dir=None):
    """Entrena ensemble de N_ENSEMBLE modelos con K-Fold."""
    if scalers_dir is None:
        scalers_dir = os.path.join(MODEL_DIR, "scalers_jansen")
    os.makedirs(scalers_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Prefiltrar escalando targets
    ys = np.stack([d.y.numpy() for d in dataset])
    log.info(f"Target stats antes de normalizar:")
    names = ['foot_err', 'knee_err', 'hip_err', 'stride_m', 'ground']
    for i, n in enumerate(names):
        log.info(f"  {n}: mean={ys[:,i].mean():.4f} std={ys[:,i].std():.4f}")

    kf = KFold(n_splits=N_ENSEMBLE, shuffle=True, random_state=42)
    physics_loss = JansenPhysicsLoss()
    models, scalers = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        log.info(f"\n{'='*50}")
        log.info(f"Modelo {fold+1}/{N_ENSEMBLE} — train={len(train_idx)}, val={len(val_idx)}")

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]

        # Fit scaler en train
        train_ys = np.stack([d.y.numpy() for d in train_data])
        scaler = StandardScaler()
        scaler.fit(train_ys)

        # Normalizar
        def normalize(data_list):
            out = []
            for d in data_list:
                d2 = d.clone()
                y_norm = scaler.transform(d2.y.numpy().reshape(1, -1)).flatten()
                d2.y = torch.tensor(y_norm, dtype=torch.float)
                out.append(d2)
            return out

        train_norm = normalize(train_data)
        val_norm = normalize(val_data)

        train_loader = DataLoader(train_norm, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=2)
        val_loader = DataLoader(val_norm, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=2)

        model = JansenGAT().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=15, T_mult=2, eta_min=1e-5)

        best_val, patience_cnt, best_state = float('inf'), 0, None

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(DEVICE)
                opt.zero_grad()
                pred = model(batch.x, batch.edge_index, batch.edge_attr,
                           batch.batch,
                           batch.global_feats.view(-1, N_GLOBAL_FEATS))
                mse = F.mse_loss(pred, batch.y.view(-1, N_TARGETS))
                phys = physics_loss(pred, batch.global_feats.view(-1, N_GLOBAL_FEATS))['total']
                loss = mse + 0.05 * phys
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_loss += loss.item() * batch.num_graphs
            sched.step()

            # Validación
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    pred = model(batch.x, batch.edge_index, batch.edge_attr,
                               batch.batch,
                               batch.global_feats.view(-1, N_GLOBAL_FEATS),
                               use_dropout=False)
                    val_loss += F.mse_loss(pred,
                                          batch.y.view(-1, N_TARGETS)).item() * batch.num_graphs

            val_loss /= len(val_data)

            if val_loss < best_val:
                best_val = val_loss
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    log.info(f"  Early stop en epoch {epoch}")
                    break

            if epoch % 20 == 0:
                log.info(f"  epoch {epoch:3d} | val={val_loss:.5f} | "
                         f"best={best_val:.5f} | patience={patience_cnt}")

        model.load_state_dict(best_state)
        model_path = os.path.join(MODEL_DIR, f"jansen_gat_{fold}.pt")
        torch.save(best_state, model_path)

        scaler_path = os.path.join(scalers_dir, f"scaler_jansen_{fold}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': scaler}, f)

        models.append(model)
        scalers.append(scaler)
        log.info(f"  Modelo {fold+1} guardado — best val_loss={best_val:.5f}")

    return models, scalers


def load_ensemble(scalers_dir=None):
    """Carga modelos y scalers ya entrenados."""
    if scalers_dir is None:
        scalers_dir = os.path.join(MODEL_DIR, "scalers_jansen")
    models, scalers = [], []
    for fold in range(N_ENSEMBLE):
        model_path = os.path.join(MODEL_DIR, f"jansen_gat_{fold}.pt")
        scaler_path = os.path.join(scalers_dir, f"scaler_jansen_{fold}.pkl")
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Modelo {fold} no encontrado: {model_path}")
        m = JansenGAT().to(DEVICE)
        m.load_state_dict(torch.load(model_path, map_location=DEVICE))
        m.eval()
        models.append(m)
        with open(scaler_path, 'rb') as f:
            scalers.append(pickle.load(f)['scaler'])
    return models, scalers


def predict_ensemble(models, scalers, data):
    """Predicción con ensemble completo. Retorna (mean, std) en escala real."""
    batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
    gf = data.global_feats.to(DEVICE).view(1, -1)
    preds = []
    with torch.no_grad():
        for model, scaler in zip(models, scalers):
            pred_n = model(data.x.to(DEVICE), data.edge_index.to(DEVICE),
                          data.edge_attr.to(DEVICE), batch_vec, gf,
                          use_dropout=False).cpu().numpy().flatten()
            pred_r = scaler.inverse_transform(pred_n.reshape(1, -1)).flatten()
            preds.append(pred_r)
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Entrenar GNN Jansen")
    parser.add_argument("--dataset", type=str, default="data/jansen_dataset.pt")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  Fase 2: Entrenamiento GNN Jansen (GATv2Conv Ensemble)")
    log.info("=" * 60)

    ds_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.dataset)
    log.info(f"Cargando dataset: {ds_path}")
    dataset = torch.load(ds_path, weights_only=False)
    log.info(f"Dataset: {len(dataset)} muestras")

    # Filtrar válidos
    valid_ds = [d for d in dataset if d.is_valid.item() > 0.5]
    log.info(f"Válidos: {len(valid_ds)} ({100*len(valid_ds)/len(dataset):.1f}%)")

    if len(valid_ds) < 100:
        log.error("Muy pocas muestras válidas. Incrementa n-samples en fase1.")
        return

    # Note: MAX_EPOCHS from config is used in train_ensemble via the module default

    models, scalers = train_ensemble(valid_ds)
    log.info(f"\n[OK] Ensemble de {len(models)} modelos entrenado")


if __name__ == "__main__":
    main()
