# ============================================================================
# WORCAP/INPE — Previsão climática de precipitação sobre a América do Sul
# Versão 6: junta as duas perguntas em aberto (v2: o ONI ajuda de verdade? /
# v3: existe uma tendência climática ao longo das décadas que a climatologia
# simples não captura?) numa validação walk-forward só (TimeSeriesSplit +
# OOF pooling, técnica do Rob Mulla), testando as 4 combinações possíveis:
# nenhuma feature extra, só ONI, só tendência, ou as duas juntas. Mais leve
# que testar 5 tipos de modelo (v5) -- usa só LightGBM, que já se mostrou
# rápido e confiável.
#
# Como usar no Kaggle:
#   1. Novo Notebook -> Add Input -> busque a competição e adicione o dataset
#   2. Internet: ON (precisa baixar o índice ONI da NOAA)
#   3. Cole cada bloco "### CÉLULA N" numa célula separada (ou tudo numa só)
#   4. No fim, "Submit to Competition" usando o submission.csv gerado
# ============================================================================

### CÉLULA 1 — imports, download dos dados da competição e do índice ONI
import os
import gc
import urllib.request
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
import kagglehub
from scipy.ndimage import uniform_filter
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Baixa os dados por código -- não precisa clicar em "Add Input" na tela
DATA_DIR = kagglehub.competition_download("previsao-climatica-de-precipitacao-sobre-a-america-do-sul")
print("Dados da competição baixados em:", DATA_DIR, flush=True)
WORK_DIR = "/kaggle/working"

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

# Validação walk-forward: Kaggle tem bastante RAM, então usamos janela
# EXPANSIVA de verdade (cada fold treina com tudo antes dele), sem precisar
# do truque de janela fixa que usamos localmente por falta de memória.
N_SPLITS = 5
TEST_SIZE_MONTHS = 60
GAP_MONTHS = MAX_LAG

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
n_time = ds_tp.sizes["time"] - 1   # última linha não tem alvo

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

# tendência temporal: ano do mês-alvo, centrado -- deixa o modelo aprender
# uma mudança gradual de regime ao longo das décadas, se ela existir
target_dates_all = pd.DatetimeIndex(ds_tp.time.values[:n_time]) + pd.DateOffset(months=1)
year_trend_arr = (target_dates_all.year.values - 1990).astype(np.float32)
X["year_trend"] = np.repeat(year_trend_arr, n_grid)

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
TREND_COLS = ["year_trend"]
ALL_FEATURES = [c for c in df.columns if c not in ("y_true", "y_resid")]
BASE_FEATURES = [c for c in ALL_FEATURES if c not in ONI_COLS + TREND_COLS]

FEATURE_SETS = {
    "base": BASE_FEATURES,
    "oni": BASE_FEATURES + ONI_COLS,
    "trend": BASE_FEATURES + TREND_COLS,
    "oni_trend": BASE_FEATURES + ONI_COLS + TREND_COLS,
}

LGB_PARAMS = dict(
    objective="regression", n_estimators=3000, learning_rate=0.03,
    num_leaves=127, max_bin=127, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, min_child_samples=200, reg_lambda=1.0,
    n_jobs=-1, random_state=42,
)


### CÉLULA 4 — validação walk-forward testando as 4 combinações (ONI x tendência)
tss = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)
time_idx_arr = np.arange(n_time)

CONFIGS = list(FEATURE_SETS.keys())
results = {c: [] for c in CONFIGS}
best_iters = {c: [] for c in CONFIGS}
oof_pred = {c: np.full(len(df), np.nan, dtype=np.float32) for c in CONFIGS}
oof_covered = np.zeros(len(df), dtype=bool)

for fold, (tr_time_idx, va_time_idx) in enumerate(tss.split(time_idx_arr)):
    va_start = pd.Timestamp(f"{YEAR_START}-01-01") + pd.DateOffset(months=int(va_time_idx.min()) + 1)
    va_end = pd.Timestamp(f"{YEAR_START}-01-01") + pd.DateOffset(months=int(va_time_idx.max()) + 1)
    print(f"\n=== Fold {fold+1}/{N_SPLITS}: valida {va_start:%Y-%m} a {va_end:%Y-%m} "
          f"(treina com {len(tr_time_idx)} meses) ===", flush=True)

    train_mask = np.repeat(np.isin(time_idx_arr, tr_time_idx), n_grid)
    valid_mask = np.repeat(np.isin(time_idx_arr, va_time_idx), n_grid)
    oof_covered |= valid_mask

    true_va = df.loc[valid_mask, "y_true"].values
    clim_va = df.loc[valid_mask, "clim_tp_next"].values

    for config_name in CONFIGS:
        feats = FEATURE_SETS[config_name]
        X_tr = df.loc[train_mask, feats]
        y_tr = df.loc[train_mask, "y_resid"]
        X_va = df.loc[valid_mask, feats]
        y_va = df.loc[valid_mask, "y_resid"]

        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        pred = np.clip(m.predict(X_va, num_iteration=m.best_iteration_) + clim_va, 0, None)
        rmse = np.sqrt(np.mean((pred - true_va) ** 2))
        results[config_name].append(rmse)
        best_iters[config_name].append(m.best_iteration_)
        oof_pred[config_name][valid_mask] = pred
        print(f"  {config_name:10s} RMSE = {rmse:.4f}  (best_iter={m.best_iteration_})", flush=True)
        del X_tr, y_tr, X_va, y_va, m
        gc.collect()

print("\n=== Resumo (média simples dos folds) ===", flush=True)
for config_name in CONFIGS:
    print(f"{config_name:10s}: {np.mean(results[config_name]):.4f}  "
          f"(por fold: {[round(x,4) for x in results[config_name]]})", flush=True)

true_all = df.loc[oof_covered, "y_true"].values
print("\n=== RMSE agregado (out-of-fold pooling) ===", flush=True)
rmse_oof = {}
for config_name in CONFIGS:
    p = oof_pred[config_name][oof_covered]
    rmse_oof[config_name] = np.sqrt(np.mean((p - true_all) ** 2))
    print(f"{config_name:10s}: {rmse_oof[config_name]:.4f}", flush=True)

WINNER = min(rmse_oof, key=rmse_oof.get)
FEATURES_FINAL = FEATURE_SETS[WINNER]
print(f"\n>>> Decisão: usar a combinação '{WINNER}' no modelo final "
      f"(ONI: {'sim' if WINNER in ('oni', 'oni_trend') else 'não'}, "
      f"tendência: {'sim' if WINNER in ('trend', 'oni_trend') else 'não'}) <<<", flush=True)

# nº de árvores pro modelo final = mediana das best_iteration dos folds do
# vencedor, com uma margem, já que o fit final não tem holdout próprio
N_ESTIMATORS_FINAL = int(np.median(best_iters[WINNER]) * 1.2)
print(f"n_estimators do modelo final: {N_ESTIMATORS_FINAL} "
      f"(mediana dos folds: {best_iters[WINNER]})", flush=True)


### CÉLULA 5 — testar blend com Ridge (regressão linear regularizada)
# Reaproveita os mesmos folds e as previsões OOF do LightGBM já calculadas
# na Célula 4. Ideia vinda da pesquisa sobre regressão em clima: Ridge lida
# bem com a multicolinearidade que já achamos entre nossas variáveis
# atmosféricas (ex: t2 e temperature_850 têm correlação de 0.98) e pode
# capturar sinal linear complementar ao que a árvore aprende.
oof_pred_ridge = np.full(len(df), np.nan, dtype=np.float32)

for fold, (tr_time_idx, va_time_idx) in enumerate(tss.split(time_idx_arr)):
    train_mask = np.repeat(np.isin(time_idx_arr, tr_time_idx), n_grid)
    valid_mask = np.repeat(np.isin(time_idx_arr, va_time_idx), n_grid)

    X_tr = df.loc[train_mask, FEATURES_FINAL]
    y_tr = df.loc[train_mask, "y_resid"]
    X_va = df.loc[valid_mask, FEATURES_FINAL]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_tr_s, y_tr)
    oof_pred_ridge[valid_mask] = ridge.predict(X_va_s)
    print(f"Ridge fold {fold+1}/{N_SPLITS} concluído", flush=True)
    del X_tr, y_tr, X_va, scaler, ridge, X_tr_s, X_va_s
    gc.collect()

clim_all = df.loc[oof_covered, "clim_tp_next"].values
pred_ridge_full = np.clip(oof_pred_ridge[oof_covered] + clim_all, 0, None)
rmse_ridge = np.sqrt(np.mean((pred_ridge_full - true_all) ** 2))

pred_blend_full = np.clip(
    0.5 * oof_pred[WINNER][oof_covered] + 0.5 * oof_pred_ridge[oof_covered] + clim_all, 0, None
)
rmse_blend = np.sqrt(np.mean((pred_blend_full - true_all) ** 2))

print(f"\nRMSE LightGBM ({WINNER}): {rmse_oof[WINNER]:.4f}", flush=True)
print(f"RMSE Ridge:              {rmse_ridge:.4f}", flush=True)
print(f"RMSE blend (50/50):      {rmse_blend:.4f}", flush=True)

USE_BLEND = rmse_blend < rmse_oof[WINNER]
print(f"\n>>> {'Usar' if USE_BLEND else 'NÃO usar'} blend com Ridge no modelo final <<<", flush=True)


### CÉLULA 6 — treinar o(s) modelo(s) final(is) com todo o histórico disponível
# Ensemble de 3 seeds pra reduzir ruído, treinado com TODOS os dados (sem
# reservar validação) já que a escolha de features foi decidida na Célula 4.
ENSEMBLE_SEEDS = (42, 7, 123)
X_all = df[FEATURES_FINAL]
y_all = df["y_resid"]

final_models = []
for seed in ENSEMBLE_SEEDS:
    print(f"\n--- treinando modelo final, seed {seed} ---", flush=True)
    params = dict(LGB_PARAMS)
    params["random_state"] = seed
    params["n_estimators"] = N_ESTIMATORS_FINAL
    m = lgb.LGBMRegressor(**params, importance_type="gain")
    m.fit(X_all, y_all)
    final_models.append(m)

imp = pd.Series(final_models[0].feature_importances_, index=FEATURES_FINAL).sort_values(ascending=False)
print("\nFeature importance (gain relativo, seed 0):", flush=True)
print((100 * imp / imp.sum()).round(1).to_string(), flush=True)

final_ridge, final_scaler = None, None
if USE_BLEND:
    print("\n--- treinando Ridge final (todos os dados) ---", flush=True)
    final_scaler = StandardScaler()
    X_all_s = final_scaler.fit_transform(X_all)
    final_ridge = Ridge(alpha=10.0, random_state=42)
    final_ridge.fit(X_all_s, y_all)
    del X_all_s
    gc.collect()

del X_all, y_all
gc.collect()


### CÉLULA 7 — montar features de teste e prever
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

year_trend_test = (pd.DatetimeIndex(ds_test["time"].values).year.values - 1990).astype(np.float32)
Xt["year_trend"] = np.repeat(year_trend_test, n_grid)

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
if USE_BLEND:
    df_test_s = final_scaler.transform(df_test)
    pred_resid_ridge_test = final_ridge.predict(df_test_s)
    pred_resid_test = 0.5 * pred_resid_test + 0.5 * pred_resid_ridge_test
pred_final = np.clip(pred_resid_test + Xt["clim_tp_next"], 0, None)


### CÉLULA 8 — montar o submission.csv
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
