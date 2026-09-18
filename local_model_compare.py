# Comparação local de modelos (LightGBM x XGBoost x Random Forest) + GridSearchCV,
# adaptada pra caber na RAM da máquina local (~2GB livres). A versão "de verdade"
# roda no Kaggle (kaggle_notebook_v5.2.py) com o grid completo -- aqui usamos uma
# amostra espacial do grid (1 a cada GRID_STRIDE células em lat e lon) e uma
# janela de treino fixa/deslizante, só pra ter uma leitura direcional rápida
# sem esperar o Kaggle.

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
import xgboost as xgb
from scipy.ndimage import uniform_filter
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

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

GRID_STRIDE = 3          # mantém 1 em cada 3 células de lat e de lon (~1/9 do grid)
N_SPLITS = 3
TEST_SIZE_MONTHS = 60
GAP_MONTHS = MAX_LAG
MAX_TRAIN_MONTHS = 200   # janela de treino deslizante, evita RAM crescer sem limite

MODEL_TYPES = ["lightgbm", "xgboost", "random_forest"]
# lightgbm_rf e hist_gb já mostraram ser piores e mais lentos no notebook v5
# completo no Kaggle -- tirados daqui pra economizar tempo/RAM local.

RF_SEED = 42


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


def to2d(arr, n):
    values = arr.values if hasattr(arr, "values") else arr
    return values.reshape(values.shape[0], -1).astype(np.float32)[:n]


print("Carregando dados (grid reduzido em 1/%d x 1/%d)..." % (GRID_STRIDE, GRID_STRIDE), flush=True)
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

print(f"n_time={n_time}  n_grid={n_grid} (reduzido)  linhas totais={n_time * n_grid:,}", flush=True)

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

FEATURES_ALL = [c for c in df.columns if c not in ("y_true", "y_resid")]


def fit_predict(model_type, X_tr, y_tr, X_va, y_va, seed=42, overrides=None):
    overrides = overrides or {}
    if model_type == "lightgbm":
        params = dict(objective="regression", n_estimators=3000, learning_rate=0.03,
                      num_leaves=127, max_bin=127, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, min_child_samples=200, reg_lambda=1.0,
                      n_jobs=4, random_state=seed)
        params.update(overrides)
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        return m, m.predict(X_va, num_iteration=m.best_iteration_), m.best_iteration_
    if model_type == "xgboost":
        params = dict(tree_method="hist", n_estimators=3000, learning_rate=0.03,
                      max_depth=8, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=50, reg_lambda=1.0, n_jobs=4, random_state=seed,
                      early_stopping_rounds=50)
        params.update(overrides)
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_iter = getattr(m, "best_iteration", None) or params["n_estimators"]
        return m, m.predict(X_va), best_iter
    if model_type == "random_forest":
        params = dict(n_estimators=200, max_depth=14, min_samples_leaf=50,
                      max_features=0.6, n_jobs=4, random_state=seed)
        params.update(overrides)
        m = RandomForestRegressor(**params)
        m.fit(X_tr, y_tr)
        return m, m.predict(X_va), params["n_estimators"]
    raise ValueError(f"model_type desconhecido: {model_type}")


def cap_train_window(tr_time_idx, max_months):
    if len(tr_time_idx) <= max_months:
        return tr_time_idx
    return tr_time_idx[-max_months:]


tss = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)
time_idx_arr = np.arange(n_time)

results = {t: [] for t in MODEL_TYPES}
best_iters = {t: [] for t in MODEL_TYPES}
oof_pred = {t: np.full(len(df), np.nan, dtype=np.float32) for t in MODEL_TYPES}
oof_covered = np.zeros(len(df), dtype=bool)

for fold, (tr_time_idx, va_time_idx) in enumerate(tss.split(time_idx_arr)):
    tr_time_idx = cap_train_window(tr_time_idx, MAX_TRAIN_MONTHS)
    va_start = pd.Timestamp(f"{YEAR_START}-01-01") + pd.DateOffset(months=int(va_time_idx.min()) + 1)
    va_end = pd.Timestamp(f"{YEAR_START}-01-01") + pd.DateOffset(months=int(va_time_idx.max()) + 1)
    print(f"\n=== Fold {fold+1}/{N_SPLITS}: valida {va_start:%Y-%m} a {va_end:%Y-%m} "
          f"(treina com {len(tr_time_idx)} meses) ===", flush=True)

    train_mask = np.repeat(np.isin(time_idx_arr, tr_time_idx), n_grid)
    valid_mask = np.repeat(np.isin(time_idx_arr, va_time_idx), n_grid)
    oof_covered |= valid_mask

    true_va = df.loc[valid_mask, "y_true"].values
    clim_va = df.loc[valid_mask, "clim_tp_next"].values
    X_tr = df.loc[train_mask, FEATURES_ALL]
    y_tr = df.loc[train_mask, "y_resid"]
    X_va = df.loc[valid_mask, FEATURES_ALL]
    y_va = df.loc[valid_mask, "y_resid"]

    for model_type in MODEL_TYPES:
        m, pred_resid, best_iter = fit_predict(model_type, X_tr, y_tr, X_va, y_va)
        pred = np.clip(pred_resid + clim_va, 0, None)
        rmse = np.sqrt(np.mean((pred - true_va) ** 2))
        results[model_type].append(rmse)
        best_iters[model_type].append(best_iter)
        oof_pred[model_type][valid_mask] = pred
        print(f"  {model_type:14s} RMSE = {rmse:.4f}  (best_iter={best_iter})", flush=True)
        del m
        gc.collect()
    del X_tr, y_tr, X_va, y_va
    gc.collect()

print("\n=== Resumo (média simples dos folds) ===", flush=True)
for t in MODEL_TYPES:
    print(f"{t:14s}: {np.mean(results[t]):.4f}  (por fold: {[round(x,4) for x in results[t]]})", flush=True)

true_all = df.loc[oof_covered, "y_true"].values
print("\n=== RMSE agregado (out-of-fold pooling) ===", flush=True)
rmse_oof = {}
for t in MODEL_TYPES:
    p = oof_pred[t][oof_covered]
    rmse_oof[t] = np.sqrt(np.mean((p - true_all) ** 2))
    print(f"{t:14s}: {rmse_oof[t]:.4f}", flush=True)

BEST_MODEL_TYPE = min(rmse_oof, key=rmse_oof.get)
print(f"\n>>> Melhor tipo de modelo (local, grid reduzido): {BEST_MODEL_TYPE} <<<", flush=True)


### GridSearchCV no modelo vencedor, respeitando ordem temporal (mesmo truque
### do v5.2: TimeSeriesSplit em nível de linha, escalado por n_grid)
N_ESTIMATORS_GRID = max(int(np.median(best_iters[BEST_MODEL_TYPE])), 50)

PARAM_GRIDS = {
    "lightgbm": {"num_leaves": [63, 127, 255], "learning_rate": [0.02, 0.05],
                 "min_child_samples": [100, 300]},
    "xgboost": {"max_depth": [6, 8, 10], "learning_rate": [0.02, 0.05],
                "min_child_weight": [20, 100]},
    "random_forest": {"max_depth": [10, 14, 20], "min_samples_leaf": [20, 50, 100]},
}


def build_base_estimator(model_type, n_estimators):
    if model_type == "lightgbm":
        return lgb.LGBMRegressor(objective="regression", n_estimators=n_estimators,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  reg_lambda=1.0, n_jobs=4, random_state=42)
    if model_type == "xgboost":
        return xgb.XGBRegressor(tree_method="hist", n_estimators=n_estimators, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4, random_state=42)
    if model_type == "random_forest":
        return RandomForestRegressor(n_estimators=200, max_features=0.6, n_jobs=4, random_state=42)
    raise ValueError(model_type)


X_all = df[FEATURES_ALL]
y_all = df["y_resid"]

grid_test_size = TEST_SIZE_MONTHS * n_grid
grid_gap = GAP_MONTHS * n_grid
row_tss_grid = TimeSeriesSplit(n_splits=N_SPLITS, test_size=grid_test_size, gap=grid_gap)
param_grid = PARAM_GRIDS[BEST_MODEL_TYPE]
n_combos = 1
for v in param_grid.values():
    n_combos *= len(v)
print(f"\nGridSearchCV ({BEST_MODEL_TYPE}): {n_combos} combinações x {N_SPLITS} folds "
      f"= {n_combos * N_SPLITS} treinos -- sem pressa.", flush=True)

grid_search = GridSearchCV(
    build_base_estimator(BEST_MODEL_TYPE, N_ESTIMATORS_GRID), param_grid,
    cv=row_tss_grid, scoring="neg_root_mean_squared_error", n_jobs=1, verbose=2,
)
grid_search.fit(X_all, y_all)

print("\nMelhores hiperparâmetros (GridSearchCV, local/grid reduzido):", grid_search.best_params_, flush=True)
print(f"Melhor RMSE (resíduo, CV interno do GridSearchCV): {-grid_search.best_score_:.4f}", flush=True)
print("\n=== FIM ===", flush=True)
