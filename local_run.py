# Versão local, econômica em memória (roda no seu PC em vez do Kaggle).
# Processa um arquivo NetCDF por vez e libera a memória antes de seguir pro
# próximo, pra caber nos poucos GB livres. Gera submission.csv pra você subir
# direto no Kaggle (Submit Predictions), sem precisar rodar notebook lá.

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

# Recorte mais curto que o notebook do Kaggle, propositalmente, pra caber em
# poucos GB de RAM livre. Se rodar bem e sobrar memória, pode aumentar depois.
YEAR_START = 1965
MAX_LAG = 2
SMOOTH_LAGS = (0, 1)          # lags (em meses) que também ganham versão suavizada espacialmente
SMOOTH_SIZES = (3,)           # tamanhos do kernel espacial (5x5 testado e não ajudou)


def load(name):
    return xr.open_dataset(os.path.join(DATA_DIR, name))


print("Carregando tp (treino) e teste_features...", flush=True)
ds_tp_all = load("treino_tp.nc")["tp"]
tp_full_1940 = ds_tp_all.values   # (996, lat, lon) -- mantido pequeno o bastante (313MB) pra montar os lags

ds_test = load("teste_features.nc")
tp_test_obs = ds_test["tp_ultima_obs"].values   # (24, lat, lon)

ds_tp = ds_tp_all.sel(time=slice(f"{YEAR_START}-01-01", None))
ds_alvo = load("treino_tp_alvo.nc")["tp_alvo"].sel(time=slice(f"{YEAR_START}-01-01", None))

lat = ds_tp.lat.values
lon = ds_tp.lon.values
LON2D, LAT2D = np.meshgrid(lon, lat)
lat_flat = LAT2D.reshape(-1).astype(np.float32)
lon_flat = LON2D.reshape(-1).astype(np.float32)
n_grid = lat_flat.shape[0]
n_time = ds_tp.sizes["time"] - 1   # última linha não tem alvo

print(f"n_time={n_time}  n_grid={n_grid}  linhas totais={n_time * n_grid:,}", flush=True)

# --- climatologia de tp e série contínua de anomalia p/ os lags -----------
clim_tp = ds_tp.groupby("time.month").mean("time")
clim_tp_arr = clim_tp.transpose("month", "lat", "lon").values   # (12, lat, lon)

tp_full_series = np.concatenate([tp_full_1940, tp_test_obs[1:]], axis=0)
month_pos = np.arange(tp_full_series.shape[0]) % 12
anom_full_series = tp_full_series - clim_tp_arr[month_pos]
del tp_full_1940, tp_test_obs
gc.collect()

# médias espaciais da anomalia (padrão de chuva em escala regional, não só pontual)
anom_full_series_smooth = {
    size: uniform_filter(anom_full_series, size=(1, size, size), mode="nearest")
    for size in SMOOTH_SIZES
}

EPOCH_YEAR = 1940


def month_index(dates):
    dates = pd.to_datetime(dates)
    return (dates.year.values - EPOCH_YEAR) * 12 + (dates.month.values - 1)


# --- índice ONI (El Niño/La Niña), NOAA -- proxy direto do estado do ENSO --
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


oni_lookup = load_oni(os.path.join(OUT_DIR, "oni.ascii.txt"), anom_full_series.shape[0])
assert not np.isnan(oni_lookup[month_index(pd.DatetimeIndex([f"{YEAR_START}-01-01"]))[0]:]).any(), \
    "faltou ONI em alguma parte do período usado -- baixe uma versão mais atual do oni.ascii.txt"

month_t = ds_tp["time.month"]
month_next = ((month_t % 12) + 1)
clim_tp_next = clim_tp.sel(month=month_next)

idx_t = month_index(pd.DatetimeIndex(ds_tp.time.values[:n_time]))
assert idx_t.min() - MAX_LAG >= 0, "YEAR_START cedo demais para o MAX_LAG escolhido"


def to2d(arr, n):
    values = arr.values if hasattr(arr, "values") else arr
    return values.reshape(values.shape[0], -1).astype(np.float32)[:n]


X = {}
for lag in range(0, MAX_LAG + 1):
    X[f"anom_tp_lag{lag}"] = to2d(anom_full_series[idx_t - lag], n_time).reshape(-1)
for size in SMOOTH_SIZES:
    for lag in SMOOTH_LAGS:
        X[f"anom_tp_smooth{size}_lag{lag}"] = to2d(anom_full_series_smooth[size][idx_t - lag], n_time).reshape(-1)

X["oni"] = np.repeat(oni_lookup[idx_t].astype(np.float32), n_grid)

X["lat"] = np.tile(lat_flat, n_time)
X["lon"] = np.tile(lon_flat, n_time)
months_next_arr = month_next.values[:n_time].astype(np.float32)
X["month_sin"] = np.repeat(np.sin(2 * np.pi * months_next_arr / 12).astype(np.float32), n_grid)
X["month_cos"] = np.repeat(np.cos(2 * np.pi * months_next_arr / 12).astype(np.float32), n_grid)
X["clim_tp_next"] = to2d(clim_tp_next, n_time).reshape(-1)
y_true = to2d(ds_alvo, n_time).reshape(-1)

del ds_alvo, clim_tp_next
gc.collect()

# --- features atmosféricas: um arquivo por vez, libera antes do próximo ---
for v in FEATURE_VARS:
    print(f"Processando {v}...", flush=True)
    da = load(f"treino_{v}.nc")[v].sel(time=slice(f"{YEAR_START}-01-01", None))
    clim_v = da.groupby("time.month").mean("time")
    anom_v = da - clim_v.sel(month=month_t)
    X[f"anom_{v}"] = to2d(anom_v, n_time).reshape(-1)
    del da, clim_v, anom_v
    gc.collect()

time_idx = np.repeat(np.arange(n_time), n_grid)

df_train = pd.DataFrame(X)
df_train["y_true"] = y_true
df_train["y_resid"] = df_train["y_true"] - df_train["clim_tp_next"]
del X, y_true
gc.collect()

print("df_train pronto:", df_train.shape, flush=True)

# --- split de validação: anos de El Niño forte, não "últimos N meses" -----
# 2023-2024 (período de teste real) foi um dos El Niño mais fortes já
# registrados. Validar em "últimos 5 anos" genéricos não testa o modelo
# nesse regime específico -- por isso o RMSE real (1.956) veio pior que a
# validação genérica indicava (1.80). Validamos agora especificamente nos
# picos dos El Niño fortes/muito fortes do passado (ONI >= ~1.5).
EL_NINO_WINDOWS = [
    ("1982-06", "1983-05"),
    ("1997-06", "1998-05"),
    ("2009-06", "2010-05"),
    ("2015-06", "2016-05"),
]

FEATURES = [c for c in df_train.columns if c not in ("y_true", "y_resid")]

origin_dates = pd.DatetimeIndex(ds_tp.time.values[:n_time])
target_dates = origin_dates + pd.DateOffset(months=1)   # mês que estamos prevendo

is_el_nino_target = np.zeros(n_time, dtype=bool)
for start, end in EL_NINO_WINDOWS:
    is_el_nino_target |= (target_dates >= pd.Timestamp(start)) & (target_dates <= pd.Timestamp(end))

print(f"Meses de validação (El Niño forte): {is_el_nino_target.sum()} de {n_time}", flush=True)
assert is_el_nino_target.sum() > 0, "YEAR_START corta todas as janelas de El Niño escolhidas"

valid_mask = np.repeat(is_el_nino_target, n_grid)
train_mask = ~valid_mask

X_tr, y_tr = df_train.loc[train_mask, FEATURES], df_train.loc[train_mask, "y_resid"]
X_va, y_va = df_train.loc[valid_mask, FEATURES], df_train.loc[valid_mask, "y_resid"]

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=127,
    max_bin=127,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=200,
    reg_lambda=1.0,
    n_jobs=4,
    random_state=42,
    importance_type="gain",
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

pred_resid_va = model.predict(X_va, num_iteration=model.best_iteration_)
clim_full_va = df_train.loc[valid_mask, "clim_tp_next"].values
pred_full_va = np.clip(pred_resid_va + clim_full_va, 0, None)
true_full_va = df_train.loc[valid_mask, "y_true"].values

rmse_model = np.sqrt(np.mean((pred_full_va - true_full_va) ** 2))
rmse_clim = np.sqrt(np.mean((clim_full_va - true_full_va) ** 2))
print(f"RMSE modelo:       {rmse_model:.4f} mm/dia", flush=True)
print(f"RMSE climatologia: {rmse_clim:.4f} mm/dia", flush=True)
print(f"Ganho sobre a climatologia: {100 * (1 - rmse_model / rmse_clim):.1f}%", flush=True)

imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importance (gain relativo):", flush=True)
print((100 * imp / imp.sum()).round(1).to_string(), flush=True)

del df_train, X_tr, X_va, y_tr, y_va
gc.collect()

# --- features de teste -----------------------------------------------------
month_origin_test = ds_test["time_origem"].dt.month
month_target_test = ds_test["time"].dt.month
idx_origin_test = month_index(pd.DatetimeIndex(ds_test["time_origem"].values))
n_time_test = ds_test.sizes["time"]

Xt = {}
for lag in range(0, MAX_LAG + 1):
    Xt[f"anom_tp_lag{lag}"] = anom_full_series[idx_origin_test - lag].reshape(n_time_test, -1).astype(np.float32).reshape(-1)
for size in SMOOTH_SIZES:
    for lag in SMOOTH_LAGS:
        Xt[f"anom_tp_smooth{size}_lag{lag}"] = anom_full_series_smooth[size][idx_origin_test - lag].reshape(n_time_test, -1).astype(np.float32).reshape(-1)

Xt["oni"] = np.repeat(oni_lookup[idx_origin_test].astype(np.float32), n_grid)

Xt["lat"] = np.tile(lat_flat, n_time_test)
Xt["lon"] = np.tile(lon_flat, n_time_test)
months_target_arr = month_target_test.values.astype(np.float32)
Xt["month_sin"] = np.repeat(np.sin(2 * np.pi * months_target_arr / 12).astype(np.float32), n_grid)
Xt["month_cos"] = np.repeat(np.cos(2 * np.pi * months_target_arr / 12).astype(np.float32), n_grid)
clim_tp_next_test = clim_tp.sel(month=month_target_test)
Xt["clim_tp_next"] = to2d(clim_tp_next_test, n_time_test).reshape(-1)

for v in FEATURE_VARS:
    print(f"Processando teste: {v}...", flush=True)
    da_full = load(f"treino_{v}.nc")[v].sel(time=slice(f"{YEAR_START}-01-01", None))
    clim_v = da_full.groupby("time.month").mean("time")
    anom_test_v = ds_test[v] - clim_v.sel(month=month_origin_test)
    Xt[f"anom_{v}"] = to2d(anom_test_v, n_time_test).reshape(-1)
    del da_full, clim_v, anom_test_v
    gc.collect()

df_test = pd.DataFrame(Xt)[FEATURES]

pred_resid_test = model.predict(df_test, num_iteration=model.best_iteration_)
pred_final = np.clip(pred_resid_test + Xt["clim_tp_next"], 0, None)

times = pd.to_datetime(ds_test["time"].values)
latlon_str = np.array([f"{la:.2f}_{lo:.2f}" for la, lo in zip(lat_flat, lon_flat)])
ids = np.concatenate([np.char.add(f"{t.year}_{t.month:02d}_", latlon_str) for t in times])

df_pred = pd.DataFrame({"id": ids, "tp_mm_day": pred_final})
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
sub_out = sub[["id"]].merge(df_pred, on="id", how="left")
assert sub_out["tp_mm_day"].isna().sum() == 0

out_path = os.path.join(OUT_DIR, "submission.csv")
sub_out.to_csv(out_path, index=False)
print("Salvo em:", out_path, flush=True)
