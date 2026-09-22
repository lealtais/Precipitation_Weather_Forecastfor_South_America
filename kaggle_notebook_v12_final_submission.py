import os
import gc
import urllib.request
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
import xgboost as xgb
from scipy.ndimage import uniform_filter

try:
    import kagglehub
    DATA_DIR = kagglehub.competition_download("previsao-climatica-de-precipitacao-sobre-a-america-do-sul")
except Exception:
    DATA_DIR = "/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul"

WORK_DIR = next((d for d in ("/kaggle/working", "/content") if os.path.isdir(d)), ".")
print(f"Diretório de dados: {DATA_DIR}")
print(f"Diretório de saída: {WORK_DIR}")

FEATURE_VARS = ["cloud_cover", "surface_pressure", "rel_hum_850", "geopotential_850"]
YEAR_START = 1970
EPOCH_YEAR = 1940
EPS = 1e-4

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
ONI_PATH = os.path.join(WORK_DIR, "oni.ascii.txt")
try:
    urllib.request.urlretrieve(ONI_URL, ONI_PATH)
    print("Índice ONI atualizado com sucesso da NOAA.")
except Exception as e:
    print(f"Aviso ao baixar ONI: {e}")


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
    lookup = np.full(n_series, 0.0, dtype=np.float32)
    if not os.path.exists(path):
        return lookup
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            seas, yr, anom = parts[0], int(parts[1]), float(parts[3])
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


ds_tp_all = load("treino_tp.nc")["tp"]
ds_test = load("teste_features.nc")
tp_test_obs = ds_test["tp_ultima_obs"].values
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

ds_tp = ds_tp_all.sel(time=slice(f"{YEAR_START}-01-01", None))
ds_alvo = load("treino_tp_alvo.nc")["tp_alvo"].sel(time=slice(f"{YEAR_START}-01-01", None))

lat = ds_tp.lat.values
lon = ds_tp.lon.values
LON2D, LAT2D = np.meshgrid(lon, lat)
lat_flat = LAT2D.reshape(-1).astype(np.float32)
lon_flat = LON2D.reshape(-1).astype(np.float32)
n_grid = lat_flat.shape[0]
n_time = ds_tp.sizes["time"] - 1

print(f"Total meses de treino: {n_time} | Grid: {n_grid} pontos", flush=True)

clim_mean = ds_tp.groupby("time.month").mean("time")
clim_std = ds_tp.groupby("time.month").std("time")
clim_mean_arr = clim_mean.transpose("month", "lat", "lon").values.astype(np.float32)
clim_std_arr = np.maximum(clim_std.transpose("month", "lat", "lon").values.astype(np.float32), EPS)

tp_full_all = ds_tp_all.values.astype(np.float32)
tp_full_series = np.concatenate([tp_full_all, tp_test_obs[1:]], axis=0).astype(np.float32)
times_full = pd.to_datetime(
    np.concatenate([ds_tp_all.time.values, ds_test["time_origem"].values[1:]])
)
months_full = times_full.month.values.astype(int)

anom_full_series = tp_full_series - clim_mean_arr[months_full - 1]
zscore_full_series = anom_full_series / clim_std_arr[months_full - 1]
zscore_smooth_3x3 = uniform_filter(zscore_full_series, size=(1, 3, 3), mode="nearest")

oni_lookup = load_oni(ONI_PATH, tp_full_series.shape[0])

valid_idx = month_index(pd.DatetimeIndex(ds_tp.time.values[:n_time]))
oni_train = oni_lookup[valid_idx]
z_train_subset = zscore_full_series[valid_idx]
oni_centered = oni_train - np.mean(oni_train)
oni_var = np.sum(oni_centered ** 2) + EPS
enso_sensitivity_map = np.sum(oni_centered[:, None, None] * z_train_subset, axis=0) / oni_var
enso_sens_flat = enso_sensitivity_map.reshape(-1).astype(np.float32)

del tp_full_all, tp_test_obs, z_train_subset
gc.collect()

print("Construindo matriz de features...", flush=True)
month_t = ds_tp["time.month"]
month_next = ((month_t % 12) + 1)
idx_t = month_index(pd.DatetimeIndex(ds_tp.time.values[:n_time]))

months_next_arr = month_next.values[:n_time].astype(np.float32)
clim_next_m = clim_mean_arr[month_next.values[:n_time] - 1]
clim_next_s = clim_std_arr[month_next.values[:n_time] - 1]

X = {}
X["lat"] = np.tile(lat_flat, n_time)
X["lon"] = np.tile(lon_flat, n_time)
X["month_sin"] = np.repeat(np.sin(2 * np.pi * months_next_arr / 12).astype(np.float32), n_grid)
X["month_cos"] = np.repeat(np.cos(2 * np.pi * months_next_arr / 12).astype(np.float32), n_grid)
X["clim_mean_next"] = clim_next_m.reshape(n_time, -1).reshape(-1)
X["clim_std_next"] = clim_next_s.reshape(n_time, -1).reshape(-1)

for lag in [0, 1, 2]:
    X[f"z_lag_{lag}"] = to2d(zscore_full_series[idx_t - lag], n_time).reshape(-1)
    X[f"z_smooth_lag_{lag}"] = to2d(zscore_smooth_3x3[idx_t - lag], n_time).reshape(-1)

X["z_lag_12"] = to2d(zscore_full_series[idx_t - 12], n_time).reshape(-1)
z_rolling_3m = (zscore_full_series[idx_t] + zscore_full_series[idx_t - 1] + zscore_full_series[idx_t - 2]) / 3.0
X["z_rolling_3m"] = to2d(z_rolling_3m, n_time).reshape(-1)

for lag in [0, 1, 2]:
    oni_val = oni_lookup[idx_t - lag].astype(np.float32)
    X[f"oni_lag_{lag}"] = np.repeat(oni_val, n_grid)
    X[f"enso_impact_lag_{lag}"] = np.repeat(oni_val, n_grid) * np.tile(enso_sens_flat, n_time)

for v in FEATURE_VARS:
    da = load(f"treino_{v}.nc")[v].sel(time=slice(f"{YEAR_START}-01-01", None))
    clim_v = da.groupby("time.month").mean("time")
    anom_v = da - clim_v.sel(month=month_t)
    X[f"anom_{v}"] = to2d(anom_v, n_time).reshape(-1)
    del da, clim_v, anom_v
    gc.collect()

y_raw = to2d(ds_alvo, n_time).reshape(-1)
clim_m_flat = X["clim_mean_next"]
clim_s_flat = X["clim_std_next"]
y_target_z = (y_raw - clim_m_flat) / clim_s_flat

df_train = pd.DataFrame(X)
df_train["target_z"] = y_target_z
FEATURES = [c for c in df_train.columns if c != "target_z"]

print(f"Shape do Dataset de Treino: {df_train.shape}", flush=True)

sample_weight = np.clip(clim_s_flat ** 2, np.percentile(clim_s_flat ** 2, 1), np.percentile(clim_s_flat ** 2, 99))

print("Treinando LightGBM (config validada por walk-forward: oni_sensmap + sample_weight sigma^2)...", flush=True)
lgb_model = lgb.LGBMRegressor(
    n_estimators=450, learning_rate=0.04, num_leaves=63,
    min_child_samples=500, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=2.0, random_state=123, n_jobs=-1,
)
lgb_model.fit(df_train[FEATURES], df_train["target_z"], sample_weight=sample_weight)

print("Treinando XGBoost (blend validado por walk-forward: +0.13% de ganho sobre LightGBM puro)...", flush=True)
xgb_model = xgb.XGBRegressor(
    n_estimators=450, learning_rate=0.04, max_depth=6,
    min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=2.0, tree_method="hist", random_state=42, n_jobs=-1,
)
xgb_model.fit(df_train[FEATURES], df_train["target_z"], sample_weight=sample_weight)

del df_train
gc.collect()

print("Processando features do conjunto de teste...", flush=True)
month_origin_test = ds_test["time_origem"].dt.month
month_target_test = ds_test["time"].dt.month
idx_origin_test = month_index(pd.DatetimeIndex(ds_test["time_origem"].values))
n_time_test = ds_test.sizes["time"]

months_target_arr = month_target_test.values.astype(np.float32)
clim_next_m_test = clim_mean_arr[month_target_test.values - 1]
clim_next_s_test = clim_std_arr[month_target_test.values - 1]

Xt = {}
Xt["lat"] = np.tile(lat_flat, n_time_test)
Xt["lon"] = np.tile(lon_flat, n_time_test)
Xt["month_sin"] = np.repeat(np.sin(2 * np.pi * months_target_arr / 12).astype(np.float32), n_grid)
Xt["month_cos"] = np.repeat(np.cos(2 * np.pi * months_target_arr / 12).astype(np.float32), n_grid)
Xt["clim_mean_next"] = clim_next_m_test.reshape(n_time_test, -1).reshape(-1)
Xt["clim_std_next"] = clim_next_s_test.reshape(n_time_test, -1).reshape(-1)

for lag in [0, 1, 2]:
    Xt[f"z_lag_{lag}"] = zscore_full_series[idx_origin_test - lag].reshape(n_time_test, -1).astype(np.float32).reshape(-1)
    Xt[f"z_smooth_lag_{lag}"] = zscore_smooth_3x3[idx_origin_test - lag].reshape(n_time_test, -1).astype(np.float32).reshape(-1)

Xt["z_lag_12"] = zscore_full_series[idx_origin_test - 12].reshape(n_time_test, -1).astype(np.float32).reshape(-1)
z_rolling_3m_test = (zscore_full_series[idx_origin_test] + zscore_full_series[idx_origin_test - 1] + zscore_full_series[idx_origin_test - 2]) / 3.0
Xt["z_rolling_3m"] = z_rolling_3m_test.reshape(n_time_test, -1).astype(np.float32).reshape(-1)

for lag in [0, 1, 2]:
    oni_val_t = oni_lookup[idx_origin_test - lag].astype(np.float32)
    Xt[f"oni_lag_{lag}"] = np.repeat(oni_val_t, n_grid)
    Xt[f"enso_impact_lag_{lag}"] = np.repeat(oni_val_t, n_grid) * np.tile(enso_sens_flat, n_time_test)

for v in FEATURE_VARS:
    da_train = load(f"treino_{v}.nc")[v].sel(time=slice(f"{YEAR_START}-01-01", None))
    clim_v = da_train.groupby("time.month").mean("time")
    anom_test_v = ds_test[v] - clim_v.sel(month=month_origin_test)
    Xt[f"anom_{v}"] = to2d(anom_test_v, n_time_test).reshape(-1)
    del da_train, clim_v, anom_test_v
    gc.collect()

df_test = pd.DataFrame(Xt)[FEATURES]

print("Gerando previsões (blend 60% XGBoost + 40% LightGBM)...", flush=True)
pred_z_lgb = lgb_model.predict(df_test)
pred_z_xgb = xgb_model.predict(df_test)
pred_z = 0.60 * pred_z_xgb + 0.40 * pred_z_lgb
pred_tp_raw = (pred_z * Xt["clim_std_next"]) + Xt["clim_mean_next"]
pred_tp_final = np.clip(pred_tp_raw, 0.0, None)

times = pd.to_datetime(ds_test["time"].values)
latlon_str = np.array([f"{la:.2f}_{lo:.2f}" for la, lo in zip(lat_flat, lon_flat)])
ids = np.concatenate([np.char.add(f"{t.year}_{t.month:02d}_", latlon_str) for t in times])

df_pred = pd.DataFrame({"id": ids, "tp_mm_day": pred_tp_final})
sub_out = sub[["id"]].merge(df_pred, on="id", how="left")
assert sub_out["tp_mm_day"].isna().sum() == 0, "Erro: existem IDs com valor NaN!"

out_path = os.path.join(WORK_DIR, "submission.csv")
sub_out.to_csv(out_path, index=False)
print(f"Submissão gerada em: {out_path}")
print(f"Média prevista: {pred_tp_final.mean():.3f} mm/dia | Max: {pred_tp_final.max():.2f}")
