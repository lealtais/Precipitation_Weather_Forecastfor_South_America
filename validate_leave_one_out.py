# Valida se o ONI realmente generaliza pra um El Niño que o modelo nunca viu,
# em vez de só "decorar" os 4 eventos históricos usados na validação normal.
# Pra cada um dos 4 eventos, treina excluindo ele e valida só nele (leave-one-
# event-out), comparando um modelo COM e SEM as features de ONI.

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
from scipy.ndimage import uniform_filter

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

EL_NINO_WINDOWS = [
    ("1982-06", "1983-05"),
    ("1997-06", "1998-05"),
    ("2009-06", "2010-05"),
    ("2015-06", "2016-05"),
]


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

origin_dates = pd.DatetimeIndex(ds_tp.time.values[:n_time])
target_dates = origin_dates + pd.DateOffset(months=1)

fold_masks = []
for start, end in EL_NINO_WINDOWS:
    m = (target_dates >= pd.Timestamp(start)) & (target_dates <= pd.Timestamp(end))
    fold_masks.append(np.repeat(m, n_grid))

results = {"com_oni": [], "sem_oni": []}
for i, (start, end) in enumerate(EL_NINO_WINDOWS):
    held_out_mask = fold_masks[i]
    train_mask_i = ~held_out_mask   # inclui os outros 3 eventos + meses normais
    print(f"\n=== Fold {i+1}/4: hold-out = {start} a {end} ({held_out_mask.sum()/n_grid:.0f} meses) ===", flush=True)

    true_ho = df.loc[held_out_mask, "y_true"].values
    clim_ho = df.loc[held_out_mask, "clim_tp_next"].values

    for label, feats in [("com_oni", FEATURES_WITH_ONI), ("sem_oni", FEATURES_NO_ONI)]:
        X_tr = df.loc[train_mask_i, feats]
        y_tr = df.loc[train_mask_i, "y_resid"]
        X_ho = df.loc[held_out_mask, feats]
        y_ho = df.loc[held_out_mask, "y_resid"]

        m = lgb.LGBMRegressor(
            objective="regression", n_estimators=3000, learning_rate=0.03,
            num_leaves=127, max_bin=127, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=200, reg_lambda=1.0,
            n_jobs=4, random_state=42,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_ho, y_ho)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = np.clip(m.predict(X_ho, num_iteration=m.best_iteration_) + clim_ho, 0, None)
        rmse = np.sqrt(np.mean((pred - true_ho) ** 2))
        results[label].append(rmse)
        print(f"  {label:10s} RMSE = {rmse:.4f}", flush=True)
        del X_tr, y_tr, X_ho, y_ho
        gc.collect()

print("\n=== Resumo (média dos 4 folds) ===", flush=True)
print(f"COM  ONI: {np.mean(results['com_oni']):.4f}  (por fold: {[round(x,4) for x in results['com_oni']]})", flush=True)
print(f"SEM  ONI: {np.mean(results['sem_oni']):.4f}  (por fold: {[round(x,4) for x in results['sem_oni']]})", flush=True)
