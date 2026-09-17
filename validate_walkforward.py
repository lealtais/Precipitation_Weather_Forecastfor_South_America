# Validação walk-forward (estilo TimeSeriesSplit do sklearn, técnica do Rob
# Mulla) em vez de validar só numa "seleção enviesada" de 4 eventos de El
# Niño. Aqui cobrimos várias janelas de tempo diferentes ao longo de todo o
# histórico, cada uma treinando só com o passado e validando num pedaço do
# futuro nunca visto -- é o jeito honesto de saber se o ONI generaliza.

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
from scipy.ndimage import uniform_filter
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = r"C:\Users\China Link\.cache\kagglehub\competitions\previsao-climatica-de-precipitacao-sobre-a-america-do-sul"
OUT_DIR = r"C:\Users\China Link\Desktop\worcap-precipitacao"

FEATURE_VARS = [
    "t2", "cloud_cover", "shum_850", "surface_pressure",
    "u_850", "v_850", "temperature_850", "rel_hum_850", "geopotential_850",
]

YEAR_START = 1965
MAX_LAG = 2
SMOOTH_LAGS = (0, 1)
SMOOTH_SIZES = (3,)
ONI_LAGS = (0, 1, 2)
EPOCH_YEAR = 1940
N_SPLITS = 5
TEST_SIZE_MONTHS = 60    # ~5 anos por fold
GAP_MONTHS = 2           # = MAX_LAG, evita qualquer sombra de vazamento
MAX_TRAIN_MONTHS = 200   # janela de treino FIXA (deslizante, não crescente) -- evita OOM nos folds mais tardios (RAM está bem curta agora)


def load(name):
    return xr.open_dataset(os.path.join(DATA_DIR, name))


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


def to2d(arr, n):
    values = arr.values if hasattr(arr, "values") else arr
    return values.reshape(values.shape[0], -1).astype(np.float32)[:n]


print("Carregando dados...", flush=True)
ds_tp_all = load("treino_tp.nc")["tp"]
tp_full_1940 = ds_tp_all.values
ds_test = load("teste_features.nc")
tp_test_obs = ds_test["tp_ultima_obs"].values

ds_tp = ds_tp_all.sel(time=slice(f"{YEAR_START}-01-01", None))
ds_alvo = load("treino_tp_alvo.nc")["tp_alvo"].sel(time=slice(f"{YEAR_START}-01-01", None))

lat = ds_tp.lat.values
lon = ds_tp.lon.values
LON2D, LAT2D = np.meshgrid(lon, lat)
lat_flat = LAT2D.reshape(-1).astype(np.float32)
lon_flat = LON2D.reshape(-1).astype(np.float32)
n_grid = lat_flat.shape[0]
n_time = ds_tp.sizes["time"] - 1

clim_tp = ds_tp.groupby("time.month").mean("time")
clim_tp_arr = clim_tp.transpose("month", "lat", "lon").values
tp_full_series = np.concatenate([tp_full_1940, tp_test_obs[1:]], axis=0)
month_pos = np.arange(tp_full_series.shape[0]) % 12
anom_full_series = tp_full_series - clim_tp_arr[month_pos]
del tp_full_1940, tp_test_obs
gc.collect()

anom_full_series_smooth = {
    size: uniform_filter(anom_full_series, size=(1, size, size), mode="nearest")
    for size in SMOOTH_SIZES
}

oni_lookup = load_oni(os.path.join(OUT_DIR, "oni.ascii.txt"), anom_full_series.shape[0])

month_t = ds_tp["time.month"]
month_next = ((month_t % 12) + 1)
clim_tp_next = clim_tp.sel(month=month_next)
idx_t = month_index(pd.DatetimeIndex(ds_tp.time.values[:n_time]))

X = {}
for lag in range(0, MAX_LAG + 1):
    X[f"anom_tp_lag{lag}"] = to2d(anom_full_series[idx_t - lag], n_time).reshape(-1)
for size in SMOOTH_SIZES:
    for lag in SMOOTH_LAGS:
        X[f"anom_tp_smooth{size}_lag{lag}"] = to2d(anom_full_series_smooth[size][idx_t - lag], n_time).reshape(-1)
for lag in ONI_LAGS:
    X[f"oni_lag{lag}"] = np.repeat(oni_lookup[idx_t - lag].astype(np.float32), n_grid)
X["lat"] = np.tile(lat_flat, n_time)
X["lon"] = np.tile(lon_flat, n_time)
months_next_arr = month_next.values[:n_time].astype(np.float32)
X["month_sin"] = np.repeat(np.sin(2 * np.pi * months_next_arr / 12).astype(np.float32), n_grid)
X["month_cos"] = np.repeat(np.cos(2 * np.pi * months_next_arr / 12).astype(np.float32), n_grid)
X["clim_tp_next"] = to2d(clim_tp_next, n_time).reshape(-1)
y_true = to2d(ds_alvo, n_time).reshape(-1)
del ds_alvo, clim_tp_next
gc.collect()

for v in FEATURE_VARS:
    print(f"Processando {v}...", flush=True)
    da = load(f"treino_{v}.nc")[v].sel(time=slice(f"{YEAR_START}-01-01", None))
    clim_v = da.groupby("time.month").mean("time")
    anom_v = da - clim_v.sel(month=month_t)
    X[f"anom_{v}"] = to2d(anom_v, n_time).reshape(-1)
    del da, clim_v, anom_v
    gc.collect()

df = pd.DataFrame(X)
df["y_true"] = y_true
df["y_resid"] = df["y_true"] - df["clim_tp_next"]
del X, y_true
gc.collect()
print("df pronto:", df.shape, flush=True)

ONI_COLS = [f"oni_lag{lag}" for lag in ONI_LAGS]
FEATURES_WITH_ONI = [c for c in df.columns if c not in ("y_true", "y_resid")]
FEATURES_NO_ONI = [c for c in FEATURES_WITH_ONI if c not in ONI_COLS]

tss = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)
time_idx_arr = np.arange(n_time)


def cap_train_window(tr_time_idx, max_months):
    """Janela de treino deslizante: mantém só os `max_months` mais recentes
    antes da validação, em vez de deixar crescer sem limite (evita OOM)."""
    if len(tr_time_idx) <= max_months:
        return tr_time_idx
    return tr_time_idx[-max_months:]

results = {"com_oni": [], "sem_oni": []}
oof_pred = {"com_oni": np.full(len(df), np.nan, dtype=np.float32),
            "sem_oni": np.full(len(df), np.nan, dtype=np.float32)}
oof_covered = np.zeros(len(df), dtype=bool)

for fold, (tr_time_idx, va_time_idx) in enumerate(tss.split(time_idx_arr)):
    tr_time_idx = cap_train_window(tr_time_idx, MAX_TRAIN_MONTHS)
    va_start_month = pd.Timestamp(f"{YEAR_START}-01-01") + pd.DateOffset(months=int(va_time_idx.min()) + 1)
    va_end_month = pd.Timestamp(f"{YEAR_START}-01-01") + pd.DateOffset(months=int(va_time_idx.max()) + 1)
    print(f"\n=== Fold {fold+1}/{N_SPLITS}: valida {va_start_month:%Y-%m} a {va_end_month:%Y-%m} "
          f"({len(va_time_idx)} meses, treina com {len(tr_time_idx)} meses anteriores) ===", flush=True)

    train_mask = np.repeat(np.isin(time_idx_arr, tr_time_idx), n_grid)
    valid_mask = np.repeat(np.isin(time_idx_arr, va_time_idx), n_grid)
    oof_covered |= valid_mask

    true_va = df.loc[valid_mask, "y_true"].values
    clim_va = df.loc[valid_mask, "clim_tp_next"].values

    for label, feats in [("com_oni", FEATURES_WITH_ONI), ("sem_oni", FEATURES_NO_ONI)]:
        X_tr = df.loc[train_mask, feats]
        y_tr = df.loc[train_mask, "y_resid"]
        X_va = df.loc[valid_mask, feats]
        y_va = df.loc[valid_mask, "y_resid"]

        m = lgb.LGBMRegressor(
            objective="regression", n_estimators=3000, learning_rate=0.03,
            num_leaves=127, max_bin=127, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=200, reg_lambda=1.0,
            n_jobs=4, random_state=42,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = np.clip(m.predict(X_va, num_iteration=m.best_iteration_) + clim_va, 0, None)
        rmse = np.sqrt(np.mean((pred - true_va) ** 2))
        results[label].append(rmse)
        oof_pred[label][valid_mask] = pred
        print(f"  {label:10s} RMSE = {rmse:.4f}", flush=True)
        del X_tr, y_tr, X_va, y_va
        gc.collect()

print("\n=== Resumo (média simples dos folds) ===", flush=True)
print(f"COM  ONI: {np.mean(results['com_oni']):.4f}  (por fold: {[round(x,4) for x in results['com_oni']]})", flush=True)
print(f"SEM  ONI: {np.mean(results['sem_oni']):.4f}  (por fold: {[round(x,4) for x in results['sem_oni']]})", flush=True)
n_wins_oni = sum(a < b for a, b in zip(results["com_oni"], results["sem_oni"]))
print(f"ONI venceu em {n_wins_oni} de {N_SPLITS} folds", flush=True)

# RMSE agregado sobre todas as previsões out-of-fold juntas (mais correto
# estatisticamente que a média simples por fold, principalmente se os folds
# tiverem tamanhos diferentes)
true_all = df.loc[oof_covered, "y_true"].values
print("\n=== RMSE agregado (out-of-fold pooling, todas as previsões juntas) ===", flush=True)
for label in ("com_oni", "sem_oni"):
    p = oof_pred[label][oof_covered]
    rmse_oof = np.sqrt(np.mean((p - true_all) ** 2))
    print(f"{label}: {rmse_oof:.4f}", flush=True)
