# ============================================================================
# WORCAP/INPE — Previsão climática de precipitação sobre a América do Sul
# Versão 8 — modelo FINAL pra submissão, usando o que já validamos:
#   - Tipo de modelo: XGBoost (venceu LightGBM/LightGBM-RF/RandomForest/
#     HistGB em walk-forward + OOF pooling, tanto no Kaggle v5.2 quanto no
#     teste local com grid reduzido)
#   - Hiperparâmetros: achados pelo GridSearchCV (local, grid reduzido):
#     learning_rate=0.05, max_depth=6, min_child_weight=20
#   - Features: mesmas do v5.2 (anomalias com lag, suavização 3x3, ONI com
#     lags 0/1/2, lat/lon, mês cíclico, climatologia) -- essa combinação já
#     foi validada com walk-forward de verdade (não é a validação enviesada
#     do antigo v7 que piorou o resultado real)
#
# Esse notebook NÃO faz comparação de modelo nem GridSearch de novo (já
# decidido) -- vai direto pro treino final + submissão, pra ser mais rápido.
#
# Como usar no Kaggle:
#   1. Novo Notebook -> a competição já aparece sozinha em Input (não precisa
#      "Add Input" manual, o código baixa os dados por conta própria)
#   2. Internet: ON (precisa baixar o índice ONI da NOAA)
#   3. Cole cada bloco "### CÉLULA N" numa célula separada
#   4. Save Version -> Save & Run All (Commit) -- roda no servidor do
#      Kaggle mesmo com o navegador fechado
#   5. Quando terminar, vai no notebook -> aba "Output" -> acha o
#      submission.csv -> "Submit to Competition"
# ============================================================================

### CÉLULA 1 — imports, download dos dados da competição e do índice ONI
import os
import gc
import urllib.request
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import kagglehub
from scipy.ndimage import uniform_filter

DATA_DIR = kagglehub.competition_download("previsao-climatica-de-precipitacao-sobre-a-america-do-sul")
print("Dados da competição baixados em:", DATA_DIR, flush=True)
WORK_DIR = next((d for d in ("/kaggle/working", "/content") if os.path.isdir(d)), ".")

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

# Hiperparâmetros vencedores do GridSearchCV (local_model_compare.py)
BEST_PARAMS = dict(learning_rate=0.05, max_depth=6, min_child_weight=20)
N_ESTIMATORS_FINAL = 400   # ~1.2x da mediana de best_iteration observada nos folds
ENSEMBLE_SEEDS = (42, 7, 123)

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
ONI_PATH = os.path.join(WORK_DIR, "oni.ascii.txt")
urllib.request.urlretrieve(ONI_URL, ONI_PATH)
print("ONI baixado:", ONI_PATH, flush=True)


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


### CÉLULA 2 — carregar dados, climatologia e série contínua de anomalia
print("Carregando dados...", flush=True)
ds_tp_all = load("treino_tp.nc")["tp"]
tp_full_1940 = ds_tp_all.values
ds_test = load("teste_features.nc")
tp_test_obs = ds_test["tp_ultima_obs"].values

ds_tp = ds_tp_all.sel(time=slice(f"{YEAR_START}-01-01", None))
ds_alvo = load("treino_tp_alvo.nc")["tp_alvo"].sel(time=slice(f"{YEAR_START}-01-01", None))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

lat = ds_tp.lat.values
lon = ds_tp.lon.values
LON2D, LAT2D = np.meshgrid(lon, lat)
lat_flat = LAT2D.reshape(-1).astype(np.float32)
lon_flat = LON2D.reshape(-1).astype(np.float32)
n_grid = lat_flat.shape[0]
n_time = ds_tp.sizes["time"] - 1

print(f"n_time={n_time}  n_grid={n_grid}  linhas totais={n_time * n_grid:,}", flush=True)

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

oni_lookup = load_oni(ONI_PATH, anom_full_series.shape[0])
_start_idx = month_index(pd.DatetimeIndex([f"{YEAR_START}-01-01"]))[0]
assert not np.isnan(oni_lookup[_start_idx:]).any(), \
    "faltou ONI em parte do período -- confira se o notebook tem Internet habilitada"


### CÉLULA 3 — montar a tabela de features completa (treino)
month_t = ds_tp["time.month"]
month_next = ((month_t % 12) + 1)
clim_tp_next = clim_tp.sel(month=month_next)
idx_t = month_index(pd.DatetimeIndex(ds_tp.time.values[:n_time]))
max_lag_needed = max(MAX_LAG, max(ONI_LAGS))
assert idx_t.min() - max_lag_needed >= 0, "YEAR_START cedo demais para os lags escolhidos"

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

FEATURES_FINAL = [c for c in df.columns if c not in ("y_true", "y_resid")]


### CÉLULA 3.5 — validação rápida (holdout dos últimos 60 meses, grid COMPLETO)
# Só pra ter um RMSE de referência real (não do grid reduzido do teste local)
# antes de gastar uma submissão. Treina com tudo antes do holdout, valida
# nele, e descarta esse modelo -- o modelo de verdade (Célula 4) é treinado
# com TODOS os dados, incluindo esse período.
VAL_MONTHS_HOLDOUT = 60
time_idx_arr = np.arange(n_time)
val_time_idx = time_idx_arr[-VAL_MONTHS_HOLDOUT:]
train_time_idx = time_idx_arr[:-VAL_MONTHS_HOLDOUT]

train_mask = np.repeat(np.isin(time_idx_arr, train_time_idx), n_grid)
valid_mask = np.repeat(np.isin(time_idx_arr, val_time_idx), n_grid)

true_va = df.loc[valid_mask, "y_true"].values
clim_va = df.loc[valid_mask, "clim_tp_next"].values
X_tr_h = df.loc[train_mask, FEATURES_FINAL]
y_tr_h = df.loc[train_mask, "y_resid"]
X_va_h = df.loc[valid_mask, FEATURES_FINAL]

print(f"\n=== Validação holdout (últimos {VAL_MONTHS_HOLDOUT} meses, grid completo) ===", flush=True)
params_holdout = dict(tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                       reg_lambda=1.0, n_jobs=-1, random_state=42)
params_holdout.update(BEST_PARAMS)
params_holdout.update(n_estimators=N_ESTIMATORS_FINAL)
m_holdout = xgb.XGBRegressor(**params_holdout)
m_holdout.fit(X_tr_h, y_tr_h)
pred_resid_h = m_holdout.predict(X_va_h)
pred_h = np.clip(pred_resid_h + clim_va, 0, None)
rmse_holdout = np.sqrt(np.mean((pred_h - true_va) ** 2))
rmse_climatology_holdout = np.sqrt(np.mean((clim_va - true_va) ** 2))
print(f"RMSE climatologia (baseline):        {rmse_climatology_holdout:.4f}", flush=True)
print(f"RMSE XGBoost final (grid completo):  {rmse_holdout:.4f}", flush=True)
print(f"Ganho sobre climatologia:             {100 * (1 - rmse_holdout / rmse_climatology_holdout):.1f}%", flush=True)
print("(esse é o RMSE de referência local -- o resultado real no leaderboard "
      "costuma ficar um pouco pior que esse, ver README)", flush=True)

del X_tr_h, y_tr_h, X_va_h, m_holdout
gc.collect()


### CÉLULA 4 — treinar o ensemble final de XGBoost com todo o histórico
X_all = df[FEATURES_FINAL]
y_all = df["y_resid"]

final_models = []
for seed in ENSEMBLE_SEEDS:
    print(f"\n--- treinando XGBoost final, seed {seed} ---", flush=True)
    params = dict(tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, n_jobs=-1)
    params.update(BEST_PARAMS)
    params.update(random_state=seed, n_estimators=N_ESTIMATORS_FINAL)
    m = xgb.XGBRegressor(**params)
    m.fit(X_all, y_all)
    final_models.append(m)

imp = pd.Series(final_models[0].feature_importances_, index=FEATURES_FINAL).sort_values(ascending=False)
print("\nFeature importance (gain relativo, seed 0):", flush=True)
print((100 * imp / imp.sum()).round(1).to_string(), flush=True)

del X_all, y_all
gc.collect()


### CÉLULA 5 — montar features de teste e prever
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
for lag in ONI_LAGS:
    Xt[f"oni_lag{lag}"] = np.repeat(oni_lookup[idx_origin_test - lag].astype(np.float32), n_grid)

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

df_test = pd.DataFrame(Xt)[FEATURES_FINAL]

pred_resid_test = np.mean([m.predict(df_test) for m in final_models], axis=0)
pred_final = np.clip(pred_resid_test + Xt["clim_tp_next"], 0, None)


### CÉLULA 6 — montar o submission.csv
times = pd.to_datetime(ds_test["time"].values)
latlon_str = np.array([f"{la:.2f}_{lo:.2f}" for la, lo in zip(lat_flat, lon_flat)])
ids = np.concatenate([np.char.add(f"{t.year}_{t.month:02d}_", latlon_str) for t in times])

df_pred = pd.DataFrame({"id": ids, "tp_mm_day": pred_final})
sub_out = sub[["id"]].merge(df_pred, on="id", how="left")
assert sub_out["tp_mm_day"].isna().sum() == 0, "faltou id no merge"

out_path = os.path.join(WORK_DIR, "submission.csv")
sub_out.to_csv(out_path, index=False)
print("Salvo em:", out_path, flush=True)
sub_out.head()
