# Modelo espacial (CNN) -- experimento pedido pelo colega de competição.
# Em vez de tratar cada célula do grid como uma linha independente (como o
# LightGBM/XGBoost fazem), aqui o modelo enxerga o MAPA inteiro de uma vez
# (todas as células ao mesmo tempo, com suas vizinhanças) e prevê o mapa de
# anomalia do mês seguinte inteiro numa passada só.
#
# Risco conhecido (documentado no README/REFERENCIAS.md): temos só ~700
# "imagens" mensais de treino, bem pouco pra CNN -- por isso a rede aqui é
# pequena e com bastante regularização (dropout + weight decay), e a
# validação é um holdout cronológico simples (não dá pra fazer walk-forward
# completo com deep learning sem multiplicar demais o tempo de treino).
#
# Roda 100% local (não precisa do Kaggle) -- usa PyTorch CPU.

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from scipy.ndimage import uniform_filter

DATA_DIR = r"C:\Users\China Link\.cache\kagglehub\competitions\previsao-climatica-de-precipitacao-sobre-a-america-do-sul"
OUT_DIR = r"C:\Users\China Link\Desktop\worcap-precipitacao"

YEAR_START = 1965
MAX_LAG = 2
SMOOTH_SIZE = 3
ONI_LAGS = (0, 1, 2)
EPOCH_YEAR = 1940
GRID_STRIDE = 2          # reduz o grid pra caber na RAM local (~1/4 das células)
VAL_MONTHS = 60          # últimos 5 anos como validação (holdout cronológico)

BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
HIDDEN_CH = 16
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def load(name):
    ds = xr.open_dataset(os.path.join(DATA_DIR, name))
    return ds.isel(lat=slice(None, None, GRID_STRIDE), lon=slice(None, None, GRID_STRIDE))


def month_index(dates):
    dates = pd.to_datetime(dates)
    return (dates.year.values - EPOCH_YEAR) * 12 + (dates.month.values - 1)


def load_oni(path, n_series):
    season_to_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    lookup = np.full(n_series, np.nan, dtype=np.float32)
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            seas, yr, _total, anom = parts[0], int(parts[1]), parts[2], float(parts[3])
            month = season_to_month.get(seas)
            if month is None:
                continue
            idx = (yr - EPOCH_YEAR) * 12 + (month - 1)
            if 0 <= idx < n_series:
                lookup[idx] = anom
    return lookup


print("Carregando dados (grid reduzido em 1/%d x 1/%d)..." % (GRID_STRIDE, GRID_STRIDE), flush=True)
ds_tp_all = load("treino_tp.nc")["tp"]
tp_full_1940 = ds_tp_all.values
ds_test = load("teste_features.nc")
tp_test_obs = ds_test["tp_ultima_obs"].values

ds_tp = ds_tp_all.sel(time=slice(f"{YEAR_START}-01-01", None))
ds_alvo = load("treino_tp_alvo.nc")["tp_alvo"].sel(time=slice(f"{YEAR_START}-01-01", None))

n_lat, n_lon = ds_tp.sizes["lat"], ds_tp.sizes["lon"]
n_time = ds_tp.sizes["time"] - 1
print(f"grid={n_lat}x{n_lon}  n_time={n_time}", flush=True)

clim_tp = ds_tp.groupby("time.month").mean("time")
clim_tp_arr = clim_tp.transpose("month", "lat", "lon").values  # (12, lat, lon)
tp_full_series = np.concatenate([tp_full_1940, tp_test_obs[1:]], axis=0)
month_pos = np.arange(tp_full_series.shape[0]) % 12
anom_full_series = tp_full_series - clim_tp_arr[month_pos]     # (time_total, lat, lon)
del tp_full_1940, tp_test_obs
gc.collect()

anom_smooth = uniform_filter(anom_full_series, size=(1, SMOOTH_SIZE, SMOOTH_SIZE), mode="nearest")
oni_lookup = load_oni(os.path.join(OUT_DIR, "oni.ascii.txt"), anom_full_series.shape[0])

month_t = ds_tp["time.month"].values[:n_time]
month_next = (month_t % 12) + 1
clim_tp_next = clim_tp_arr[month_next - 1]                      # (n_time, lat, lon)
idx_t = month_index(pd.DatetimeIndex(ds_tp.time.values[:n_time]))

# Monta os "canais" de entrada, cada um um mapa (n_time, lat, lon)
channels = []
for lag in range(0, MAX_LAG + 1):
    channels.append(anom_full_series[idx_t - lag][:n_time])
channels.append(anom_smooth[idx_t][:n_time])  # suavização 3x3, lag 0
for lag in ONI_LAGS:
    oni_vals = oni_lookup[idx_t - lag].astype(np.float32)
    channels.append(np.broadcast_to(oni_vals[:, None, None], (n_time, n_lat, n_lon)))
month_sin = np.sin(2 * np.pi * month_next.astype(np.float32) / 12)
month_cos = np.cos(2 * np.pi * month_next.astype(np.float32) / 12)
channels.append(np.broadcast_to(month_sin[:, None, None], (n_time, n_lat, n_lon)))
channels.append(np.broadcast_to(month_cos[:, None, None], (n_time, n_lat, n_lon)))
channels.append(clim_tp_next)

X = np.stack(channels, axis=1).astype(np.float32)  # (n_time, n_channels, lat, lon)
N_CHANNELS = X.shape[1]
print("X pronto:", X.shape, f"({N_CHANNELS} canais)", flush=True)
del channels
gc.collect()

y_true = ds_alvo.values[:n_time].astype(np.float32)            # (n_time, lat, lon)
y_resid = y_true - clim_tp_next

# normaliza cada canal (média/desvio calculados só no treino, pra não vazar
# informação do período de validação)
n_train = n_time - VAL_MONTHS
mean_ch = X[:n_train].mean(axis=(0, 2, 3), keepdims=True)
std_ch = X[:n_train].std(axis=(0, 2, 3), keepdims=True) + 1e-6
X_norm = (X - mean_ch) / std_ch

X_tr, X_va = X_norm[:n_train], X_norm[n_train:]
y_tr, y_va = y_resid[:n_train], y_resid[n_train:]
true_va, clim_va = y_true[n_train:], clim_tp_next[n_train:]
print(f"treino: {X_tr.shape[0]} meses | validação: {X_va.shape[0]} meses (últimos {VAL_MONTHS})", flush=True)


class SpatialCNN(nn.Module):
    """CNN pequena e bem regularizada -- sem downsampling (precisamos do
    mapa de saída no mesmo tamanho da entrada), poucos canais escondidos
    pra não estourar overfit com só ~600 exemplos de treino."""

    def __init__(self, in_ch, hidden_ch=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_ch, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


device = torch.device("cpu")
model = SpatialCNN(N_CHANNELS, HIDDEN_CH, DROPOUT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

X_tr_t = torch.from_numpy(X_tr)
y_tr_t = torch.from_numpy(y_tr)
X_va_t = torch.from_numpy(X_va).to(device)
y_va_t = torch.from_numpy(y_va).to(device)

n_train_samples = X_tr_t.shape[0]
best_val = float("inf")
best_state = None
patience_left = PATIENCE

print("\nTreinando CNN espacial...", flush=True)
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    perm = torch.randperm(n_train_samples)
    train_loss = 0.0
    for i in range(0, n_train_samples, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb, yb = X_tr_t[idx].to(device), y_tr_t[idx].to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        train_loss += loss.item() * xb.shape[0]
    train_loss /= n_train_samples

    model.eval()
    with torch.no_grad():
        val_pred = model(X_va_t)
        val_loss = loss_fn(val_pred, y_va_t).item()

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_left = PATIENCE
    else:
        patience_left -= 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"  epoch {epoch:3d}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}  "
              f"(sem melhora há {PATIENCE - patience_left})", flush=True)

    if patience_left <= 0:
        print(f"  early stopping na epoch {epoch} (sem melhora por {PATIENCE} epochs)", flush=True)
        break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pred_resid_va = model(X_va_t).numpy()

pred_final_va = np.clip(pred_resid_va + clim_va, 0, None)
rmse_cnn = np.sqrt(np.mean((pred_final_va - true_va) ** 2))
rmse_climatology = np.sqrt(np.mean((clim_va - true_va) ** 2))

print("\n=== Resultado (validação = últimos %d meses) ===" % VAL_MONTHS, flush=True)
print(f"RMSE climatologia (baseline): {rmse_climatology:.4f}", flush=True)
print(f"RMSE CNN espacial:            {rmse_cnn:.4f}", flush=True)
print(f"Ganho sobre climatologia:     {100 * (1 - rmse_cnn / rmse_climatology):.1f}%", flush=True)
print("\n(compare com o RMSE do LightGBM/XGBoost no mesmo tipo de recorte -- "
      "ver local_model_compare.py / kaggle_notebook_v5.2.py)", flush=True)

torch.save(model.state_dict(), os.path.join(OUT_DIR, "cnn_spatial_model.pt"))
print("\nModelo salvo em cnn_spatial_model.pt", flush=True)
print("=== FIM ===", flush=True)
