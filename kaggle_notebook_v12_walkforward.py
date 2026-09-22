import os
import gc
import time
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

FEATURE_VARS = [
    "t2", "cloud_cover", "shum_850", "surface_pressure",
    "u_850", "v_850", "temperature_850", "rel_hum_850", "geopotential_850",
]
ANOM_ONLY_VARS = ["cloud_cover", "surface_pressure", "rel_hum_850", "geopotential_850"]

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


def to2d(arr, n=None):
    values = arr.values if hasattr(arr, "values") else arr
    out = values.reshape(values.shape[0], -1).astype(np.float32)
    return out[:n] if n is not None else out


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


print("Carregando variáveis de treino (histórico completo)...", flush=True)

ds_tp_full = load("treino_tp.nc")["tp"]
times_full = pd.to_datetime(ds_tp_full.time.values)
months_full = times_full.month.values.astype(int)
years_full = times_full.year.values.astype(int)
n_full = ds_tp_full.sizes["time"]

tp_full = ds_tp_full.values.astype(np.float32)
alvo_full = load("treino_tp_alvo.nc")["tp_alvo"].values.astype(np.float32)

assert alvo_full.shape[0] >= n_full - 1, (
    f"treino_tp_alvo.nc tem {alvo_full.shape[0]} meses, esperado >= {n_full - 1}. "
    "Confira o alinhamento entre treino_tp.nc e treino_tp_alvo.nc antes de continuar."
)
LAST_VALID_ORIGIN = n_full - 1

DEFAULT_STRIDE = "1" if WORK_DIR.startswith("/kaggle") else "2"
GRID_STRIDE = int(os.environ.get("GRID_STRIDE", DEFAULT_STRIDE))

RAW_VARS = {}
for v in FEATURE_VARS:
    da = load(f"treino_{v}.nc")[v]
    RAW_VARS[v] = da.values[:, ::GRID_STRIDE, ::GRID_STRIDE].astype(np.float32)
    del da
    gc.collect()

tp_full = tp_full[:, ::GRID_STRIDE, ::GRID_STRIDE]
alvo_full = alvo_full[:, ::GRID_STRIDE, ::GRID_STRIDE]

lat = ds_tp_full.lat.values[::GRID_STRIDE]
lon = ds_tp_full.lon.values[::GRID_STRIDE]
LON2D, LAT2D = np.meshgrid(lon, lat)
lat_flat = LAT2D.reshape(-1).astype(np.float32)
lon_flat = LON2D.reshape(-1).astype(np.float32)
n_grid = lat_flat.shape[0]

oni_lookup = load_oni(ONI_PATH, n_full)

print(f"Histórico total: {n_full} meses | Grid: {n_grid} pontos", flush=True)


def build_climatology(values_full, months_full, train_mask):
    clim_mean = np.zeros((12,) + values_full.shape[1:], dtype=np.float32)
    clim_std = np.zeros((12,) + values_full.shape[1:], dtype=np.float32)
    for m in range(1, 13):
        sel = train_mask & (months_full == m)
        subset = values_full[sel]
        clim_mean[m - 1] = subset.mean(axis=0)
        clim_std[m - 1] = np.maximum(subset.std(axis=0), EPS)
    return clim_mean, clim_std


def to_zscore(values_full, months_full, clim_mean, clim_std):
    anom = values_full - clim_mean[months_full - 1]
    z = anom / clim_std[months_full - 1]
    return anom.astype(np.float32), z.astype(np.float32)


def build_enso_sensitivity(oni_full, z_full, train_idx):
    oni_train = oni_full[train_idx]
    z_train = z_full[train_idx]
    oni_centered = oni_train - oni_train.mean()
    oni_var = np.sum(oni_centered ** 2) + EPS
    sens_map = np.sum(oni_centered[:, None, None] * z_train, axis=0) / oni_var
    return sens_map.reshape(-1).astype(np.float32)


def build_feature_table(idx, months_target, clim_mean, clim_std, z_full, z_smooth_full,
                         oni_full, enso_sens_flat, raw_anom_clim, physical_clim,
                         use_enso_impact, use_physical, physical_raw_mode):
    n = len(idx)
    X = {}
    X["lat"] = np.tile(lat_flat, n)
    X["lon"] = np.tile(lon_flat, n)
    X["month_sin"] = np.repeat(np.sin(2 * np.pi * months_target / 12).astype(np.float32), n_grid)
    X["month_cos"] = np.repeat(np.cos(2 * np.pi * months_target / 12).astype(np.float32), n_grid)

    clim_next_m = clim_mean[months_target - 1]
    clim_next_s = clim_std[months_target - 1]
    X["clim_mean_next"] = clim_next_m.reshape(n, -1).reshape(-1)
    X["clim_std_next"] = clim_next_s.reshape(n, -1).reshape(-1)

    for lag in [0, 1, 2]:
        X[f"z_lag_{lag}"] = to2d(z_full[idx - lag]).reshape(-1)
        X[f"z_smooth_lag_{lag}"] = to2d(z_smooth_full[idx - lag]).reshape(-1)
    X["z_lag_12"] = to2d(z_full[idx - 12]).reshape(-1)
    z_roll = (z_full[idx] + z_full[idx - 1] + z_full[idx - 2]) / 3.0
    X["z_rolling_3m"] = to2d(z_roll).reshape(-1)

    for lag in [0, 1, 2]:
        oni_val = oni_full[idx - lag].astype(np.float32)
        X[f"oni_lag_{lag}"] = np.repeat(oni_val, n_grid)
        if use_enso_impact:
            X[f"enso_impact_lag_{lag}"] = np.repeat(oni_val, n_grid) * np.tile(enso_sens_flat, n)

    if use_physical:
        u = RAW_VARS["u_850"][idx]
        v = RAW_VARS["v_850"][idx]
        shum = RAW_VARS["shum_850"][idx]
        t2 = RAW_VARS["t2"][idx]
        t850 = RAW_VARS["temperature_850"][idx]

        flux_u = u * shum
        flux_v = v * shum
        flux_mag = np.sqrt(flux_u ** 2 + flux_v ** 2)
        instab = t2 - t850

        if physical_raw_mode == "raw":
            X["flux_u"] = to2d(flux_u).reshape(-1)
            X["flux_v"] = to2d(flux_v).reshape(-1)
            X["flux_mag"] = to2d(flux_mag).reshape(-1)
            X["thermal_instability"] = to2d(instab).reshape(-1)
        else:
            cm = physical_clim
            X["flux_u"] = to2d(flux_u - cm["flux_u"][months_full[idx] - 1]).reshape(-1)
            X["flux_v"] = to2d(flux_v - cm["flux_v"][months_full[idx] - 1]).reshape(-1)
            X["flux_mag"] = to2d(flux_mag - cm["flux_mag"][months_full[idx] - 1]).reshape(-1)
            X["thermal_instability"] = to2d(instab - cm["instab"][months_full[idx] - 1]).reshape(-1)

        del u, v, shum, t2, t850, flux_u, flux_v, flux_mag, instab

    for v in ANOM_ONLY_VARS:
        clim_v = raw_anom_clim[v]
        anom_v = RAW_VARS[v][idx] - clim_v[months_full[idx] - 1]
        X[f"anom_{v}"] = to2d(anom_v).reshape(-1)

    return pd.DataFrame(X)


def build_physical_climatology(train_mask):
    u = RAW_VARS["u_850"]
    v = RAW_VARS["v_850"]
    shum = RAW_VARS["shum_850"]
    t2 = RAW_VARS["t2"]
    t850 = RAW_VARS["temperature_850"]
    flux_u = u * shum
    flux_v = v * shum
    flux_mag = np.sqrt(flux_u ** 2 + flux_v ** 2)
    instab = t2 - t850
    out = {}
    for name, arr in [("flux_u", flux_u), ("flux_v", flux_v), ("flux_mag", flux_mag), ("instab", instab)]:
        cm = np.zeros((12,) + arr.shape[1:], dtype=np.float32)
        for m in range(1, 13):
            sel = train_mask & (months_full == m)
            cm[m - 1] = arr[sel].mean(axis=0)
        out[name] = cm
    del u, v, shum, t2, t850, flux_u, flux_v, flux_mag, instab
    gc.collect()
    return out


YEAR_START = 1970

FOLDS = [
    dict(train_end=1994, val_start=1995, val_end=1999),
    dict(train_end=1999, val_start=2000, val_end=2004),
    dict(train_end=2004, val_start=2005, val_end=2009),
    dict(train_end=2009, val_start=2010, val_end=2014),
    dict(train_end=2014, val_start=2015, val_end=2019),
]

CONFIGS = {
    "baseline_sem_oni":        dict(use_oni=False, use_enso_impact=False, use_physical=False, physical_mode=None),
    "oni_simples":             dict(use_oni=True,  use_enso_impact=False, use_physical=False, physical_mode=None),
    "oni_sensmap":             dict(use_oni=True,  use_enso_impact=True,  use_physical=False, physical_mode=None),
    "oni_sensmap_fisica_anom": dict(use_oni=True,  use_enso_impact=True,  use_physical=True,  physical_mode="anom"),
    "oni_sensmap_fisica_raw":  dict(use_oni=True,  use_enso_impact=True,  use_physical=True,  physical_mode="raw"),
}

N_ESTIMATORS_CV = 200
QUICK_MODE = False
if QUICK_MODE:
    FOLDS = FOLDS[:2]
    CONFIGS = {k: v for k, v in list(CONFIGS.items())[:3]}
    N_ESTIMATORS_CV = 80

results = []

for fold_i, fold in enumerate(FOLDS):
    t_fold_start = time.time()
    train_mask = (years_full >= YEAR_START) & (years_full < fold["train_end"])
    val_mask = (years_full >= fold["val_start"]) & (years_full < fold["val_end"])

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    train_idx = train_idx[(train_idx >= 12) & (train_idx < LAST_VALID_ORIGIN)]
    val_idx = val_idx[(val_idx >= 12) & (val_idx < LAST_VALID_ORIGIN)]

    months_target_train = months_full[train_idx] % 12 + 1
    months_target_val = months_full[val_idx] % 12 + 1

    print(f"\n=== Fold {fold_i+1}/{len(FOLDS)} | treino até {fold['train_end']} "
          f"| val {fold['val_start']}-{fold['val_end']} "
          f"| {len(train_idx)} treino / {len(val_idx)} val (meses) ===", flush=True)

    clim_mean, clim_std = build_climatology(tp_full, months_full, train_mask)
    anom_full, z_full = to_zscore(tp_full, months_full, clim_mean, clim_std)
    z_smooth_full = uniform_filter(z_full, size=(1, 3, 3), mode="nearest")

    raw_anom_clim = {}
    for v in ANOM_ONLY_VARS:
        cm, _ = build_climatology(RAW_VARS[v], months_full, train_mask)
        raw_anom_clim[v] = cm

    enso_sens_flat = build_enso_sensitivity(oni_lookup, z_full, train_idx)
    physical_clim = build_physical_climatology(train_mask)

    y_train_raw = to2d(alvo_full[train_idx]).reshape(-1)
    y_val_raw = to2d(alvo_full[val_idx]).reshape(-1)
    clim_train_flat = clim_mean[months_target_train - 1].reshape(len(train_idx), -1).reshape(-1)
    clim_val_flat = clim_mean[months_target_val - 1].reshape(len(val_idx), -1).reshape(-1)
    std_train_flat = clim_std[months_target_train - 1].reshape(len(train_idx), -1).reshape(-1)
    std_val_flat = clim_std[months_target_val - 1].reshape(len(val_idx), -1).reshape(-1)

    y_train_z = (y_train_raw - clim_train_flat) / std_train_flat
    baseline_rmse = rmse(y_val_raw, np.clip(clim_val_flat, 0.0, None))

    for cfg_name, cfg in CONFIGS.items():
        t_cfg = time.time()

        df_train = build_feature_table(
            train_idx, months_target_train, clim_mean, clim_std, z_full, z_smooth_full,
            oni_lookup, enso_sens_flat, raw_anom_clim, physical_clim,
            use_enso_impact=cfg["use_enso_impact"], use_physical=cfg["use_physical"],
            physical_raw_mode=cfg["physical_mode"],
        )
        if not cfg["use_oni"]:
            df_train = df_train[[c for c in df_train.columns if not c.startswith("oni_") and not c.startswith("enso_impact")]]
        features = list(df_train.columns)

        lgb_model = lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS_CV, learning_rate=0.05, num_leaves=63,
            min_child_samples=500, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, random_state=123, n_jobs=-1, verbosity=-1,
        )
        lgb_model.fit(df_train[features], y_train_z)
        del df_train
        gc.collect()

        df_val = build_feature_table(
            val_idx, months_target_val, clim_mean, clim_std, z_full, z_smooth_full,
            oni_lookup, enso_sens_flat, raw_anom_clim, physical_clim,
            use_enso_impact=cfg["use_enso_impact"], use_physical=cfg["use_physical"],
            physical_raw_mode=cfg["physical_mode"],
        )[features]

        pred_z = lgb_model.predict(df_val)
        pred_mm = np.clip(pred_z * std_val_flat + clim_val_flat, 0.0, None)
        model_rmse = rmse(y_val_raw, pred_mm)
        gain = 100 * (1 - model_rmse / baseline_rmse)

        results.append(dict(
            fold=fold_i + 1, config=cfg_name,
            rmse_model=round(model_rmse, 4), rmse_clim=round(baseline_rmse, 4),
            ganho_pct=round(gain, 2), segundos=round(time.time() - t_cfg, 1),
        ))
        print(f"  [{cfg_name:26s}] RMSE modelo={model_rmse:.4f} | clim={baseline_rmse:.4f} "
              f"| ganho={gain:+.2f}% | {time.time()-t_cfg:.1f}s", flush=True)

        del df_val, lgb_model
        gc.collect()

    del clim_mean, clim_std, anom_full, z_full, z_smooth_full, enso_sens_flat, physical_clim, raw_anom_clim
    gc.collect()
    print(f"--- Fold {fold_i+1} concluído em {time.time()-t_fold_start:.1f}s ---", flush=True)

df_results = pd.DataFrame(results)
summary = df_results.groupby("config").agg(
    rmse_medio=("rmse_model", "mean"),
    rmse_std=("rmse_model", "std"),
    ganho_medio_pct=("ganho_pct", "mean"),
).sort_values("rmse_medio")

print("\n" + "=" * 72)
print("RESUMO WALK-FORWARD (média entre folds, menor RMSE = melhor)")
print("=" * 72)
print(summary.to_string())

out_path = os.path.join(WORK_DIR, "walkforward_results.csv")
df_results.to_csv(out_path, index=False)
print(f"\nResultado detalhado por fold salvo em: {out_path}")
