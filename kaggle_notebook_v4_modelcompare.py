# ============================================================================
# WORCAP/INPE — Previsão climática de precipitação sobre a América do Sul
# Versão 4 (variante da v2): compara LightGBM, LightGBM-RF (aproximação de
# Random Forest com a mesma velocidade de histograma) e XGBoost, todos com a
# MESMA validação walk-forward + OOF pooling. Random Forest e Gradient
# Boosting "puros" do sklearn foram deixados de fora de propósito: nesse
# volume de dados (30-50 milhões de linhas por fold) eles não têm os atalhos
# de histograma do LightGBM/XGBoost nem early stopping real, e travariam ou
# levariam muitas horas (mesmo problema que tivemos com o CatBoost).
#
# Depois de decidir o melhor tipo de modelo, faz uma busca de hiperparâmetros
# com Optuna (mais eficiente que grid search exaustivo) -- primeiro uma
# triagem rápida num fold só, depois confirma o melhor candidato com a
# validação completa (5 folds).
#
# Como usar no Kaggle:
#   1. Novo Notebook -> Add Input -> busque a competição e adicione o dataset
#   2. Internet: ON (precisa baixar o índice ONI da NOAA e instalar optuna)
#   3. Cole cada bloco "### CÉLULA N" numa célula separada (ou tudo numa só)
#   4. No fim, "Submit to Competition" usando o submission.csv gerado
# ============================================================================

### CÉLULA 1 — imports, config e download do índice ONI
import os
import gc
import urllib.request
import subprocess
import sys
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
import xgboost as xgb
from scipy.ndimage import uniform_filter
from sklearn.model_selection import TimeSeriesSplit

try:
    import optuna
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna"], check=True)
    import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = "/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul"
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

FEATURES_WITH_ONI = [c for c in df.columns if c not in ("y_true", "y_resid")]


### CÉLULA 4 — comparar LightGBM, LightGBM-RF e XGBoost (mesma validação walk-forward)
# Usa o conjunto completo de features (com ONI) -- a pergunta aqui é sobre o
# MODELO, não sobre feature. A pergunta do ONI é respondida em paralelo pelo
# notebook v2.
FEATURES_ALL = FEATURES_WITH_ONI
MODEL_TYPES = ["lightgbm", "lightgbm_rf", "xgboost"]


def fit_predict(model_type, X_tr, y_tr, X_va, y_va, seed=42, overrides=None):
    overrides = overrides or {}
    if model_type == "lightgbm":
        params = dict(objective="regression", n_estimators=3000, learning_rate=0.03,
                      num_leaves=127, max_bin=127, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, min_child_samples=200, reg_lambda=1.0,
                      n_jobs=-1, random_state=seed)
        params.update(overrides)
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        return m, m.predict(X_va, num_iteration=m.best_iteration_), m.best_iteration_
    if model_type == "lightgbm_rf":
        # aproximação de Random Forest com a mesma velocidade de histograma
        # do LightGBM -- RF de verdade (sklearn) travaria nesse volume de dados
        params = dict(boosting_type="rf", n_estimators=1000, num_leaves=127, max_bin=127,
                      bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8,
                      min_child_samples=200, n_jobs=-1, random_state=seed)
        params.update(overrides)
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        return m, m.predict(X_va, num_iteration=m.best_iteration_), m.best_iteration_
    if model_type == "xgboost":
        params = dict(tree_method="hist", n_estimators=3000, learning_rate=0.03,
                      max_depth=8, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=50, reg_lambda=1.0, n_jobs=-1, random_state=seed,
                      early_stopping_rounds=50)
        params.update(overrides)
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_iter = getattr(m, "best_iteration", None) or params["n_estimators"]
        return m, m.predict(X_va), best_iter
    raise ValueError(f"model_type desconhecido: {model_type}")


tss = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE_MONTHS, gap=GAP_MONTHS)
time_idx_arr = np.arange(n_time)

results = {t: [] for t in MODEL_TYPES}
best_iters = {t: [] for t in MODEL_TYPES}
oof_pred = {t: np.full(len(df), np.nan, dtype=np.float32) for t in MODEL_TYPES}
oof_covered = np.zeros(len(df), dtype=bool)
last_fold_time_idx = None   # guarda o último fold pra triagem rápida do Optuna na Célula 5

for fold, (tr_time_idx, va_time_idx) in enumerate(tss.split(time_idx_arr)):
    va_start = pd.Timestamp(f"{YEAR_START}-01-01") + pd.DateOffset(months=int(va_time_idx.min()) + 1)
    va_end = pd.Timestamp(f"{YEAR_START}-01-01") + pd.DateOffset(months=int(va_time_idx.max()) + 1)
    print(f"\n=== Fold {fold+1}/{N_SPLITS}: valida {va_start:%Y-%m} a {va_end:%Y-%m} "
          f"(treina com {len(tr_time_idx)} meses) ===", flush=True)
    last_fold_time_idx = (tr_time_idx, va_time_idx)

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
        print(f"  {model_type:12s} RMSE = {rmse:.4f}  (best_iter={best_iter})", flush=True)
        del m
        gc.collect()
    del X_tr, y_tr, X_va, y_va
    gc.collect()

print("\n=== Resumo (média simples dos folds) ===", flush=True)
for t in MODEL_TYPES:
    print(f"{t:12s}: {np.mean(results[t]):.4f}  (por fold: {[round(x,4) for x in results[t]]})", flush=True)

true_all = df.loc[oof_covered, "y_true"].values
print("\n=== RMSE agregado (out-of-fold pooling) ===", flush=True)
rmse_oof = {}
for t in MODEL_TYPES:
    p = oof_pred[t][oof_covered]
    rmse_oof[t] = np.sqrt(np.mean((p - true_all) ** 2))
    print(f"{t:12s}: {rmse_oof[t]:.4f}", flush=True)

BEST_MODEL_TYPE = min(rmse_oof, key=rmse_oof.get)
print(f"\n>>> Melhor tipo de modelo: {BEST_MODEL_TYPE} <<<", flush=True)


### CÉLULA 5 — tuning de hiperparâmetros com Optuna
# "Hyperparameter tuning é inútil sem uma validação adequada" (lição do
# notebook de Optuna do Rob Mulla) -- por isso só afinamos hiperparâmetro
# DEPOIS de escolher o tipo de modelo com validação walk-forward de verdade.
# Triagem rápida no último fold (o mais recente/parecido com o teste real),
# depois confirma o melhor candidato com a validação completa dos 5 folds.
N_OPTUNA_TRIALS = 20

tr_time_idx_last, va_time_idx_last = last_fold_time_idx
train_mask_last = np.repeat(np.isin(time_idx_arr, tr_time_idx_last), n_grid)
valid_mask_last = np.repeat(np.isin(time_idx_arr, va_time_idx_last), n_grid)
X_tr_last = df.loc[train_mask_last, FEATURES_ALL]
y_tr_last = df.loc[train_mask_last, "y_resid"]
X_va_last = df.loc[valid_mask_last, FEATURES_ALL]
y_va_last = df.loc[valid_mask_last, "y_resid"]
true_va_last = df.loc[valid_mask_last, "y_true"].values
clim_va_last = df.loc[valid_mask_last, "clim_tp_next"].values


def suggest_overrides(trial, model_type):
    if model_type == "lightgbm":
        return dict(
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            max_bin=trial.suggest_int("max_bin", 63, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 500),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        )
    if model_type == "lightgbm_rf":
        return dict(
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 0.95),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 0.95),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 500),
        )
    return dict(  # xgboost
        max_depth=trial.suggest_int("max_depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 300),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    )


def objective(trial):
    overrides = suggest_overrides(trial, BEST_MODEL_TYPE)
    _, pred_resid, _ = fit_predict(BEST_MODEL_TYPE, X_tr_last, y_tr_last, X_va_last, y_va_last,
                                    overrides=overrides)
    pred = np.clip(pred_resid + clim_va_last, 0, None)
    return np.sqrt(np.mean((pred - true_va_last) ** 2))


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

baseline_rmse_last_fold = results[BEST_MODEL_TYPE][-1]
print(f"\nMelhor RMSE do Optuna (triagem, 1 fold): {study.best_value:.4f}", flush=True)
print(f"RMSE do baseline (params default) nesse mesmo fold: {baseline_rmse_last_fold:.4f}", flush=True)
print(f"Melhores hiperparâmetros encontrados: {study.best_params}", flush=True)

BEST_PARAMS = study.best_params if study.best_value < baseline_rmse_last_fold else {}
if not BEST_PARAMS:
    print("Optuna não superou os parâmetros default -- mantendo o baseline.", flush=True)

del X_tr_last, y_tr_last, X_va_last, y_va_last
gc.collect()

# nº de árvores pro modelo final = mediana das best_iteration dos folds do
# tipo de modelo vencedor, com uma margem, já que o fit final não tem holdout
N_ESTIMATORS_FINAL = int(np.median(best_iters[BEST_MODEL_TYPE]) * 1.2)
print(f"n_estimators do modelo final ({BEST_MODEL_TYPE}): {N_ESTIMATORS_FINAL} "
      f"(mediana dos folds: {best_iters[BEST_MODEL_TYPE]})", flush=True)


### CÉLULA 6 — treinar o(s) modelo(s) final(is) com todo o histórico disponível
# Ensemble de 3 seeds pra reduzir ruído, treinado com TODOS os dados (sem
# reservar validação) já que tipo de modelo e hiperparâmetros já foram
# decididos nas Células 4 e 5.
ENSEMBLE_SEEDS = (42, 7, 123)
X_all = df[FEATURES_ALL]
y_all = df["y_resid"]


def build_final_model(model_type, seed, overrides, n_estimators):
    if model_type == "lightgbm":
        params = dict(objective="regression", learning_rate=0.03, num_leaves=127, max_bin=127,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      min_child_samples=200, reg_lambda=1.0, n_jobs=-1)
        params.update(overrides)
        params.update(random_state=seed, n_estimators=n_estimators, importance_type="gain")
        return lgb.LGBMRegressor(**params)
    if model_type == "lightgbm_rf":
        params = dict(boosting_type="rf", num_leaves=127, max_bin=127, bagging_fraction=0.8,
                      bagging_freq=1, feature_fraction=0.8, min_child_samples=200, n_jobs=-1)
        params.update(overrides)
        params.update(random_state=seed, n_estimators=n_estimators, importance_type="gain")
        return lgb.LGBMRegressor(**params)
    params = dict(tree_method="hist", learning_rate=0.03, max_depth=8, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=50, reg_lambda=1.0, n_jobs=-1)
    params.update(overrides)
    params.update(random_state=seed, n_estimators=n_estimators)
    return xgb.XGBRegressor(**params)


final_models = []
for seed in ENSEMBLE_SEEDS:
    print(f"\n--- treinando modelo final ({BEST_MODEL_TYPE}), seed {seed} ---", flush=True)
    m = build_final_model(BEST_MODEL_TYPE, seed, BEST_PARAMS, N_ESTIMATORS_FINAL)
    m.fit(X_all, y_all)
    final_models.append(m)

imp = pd.Series(final_models[0].feature_importances_, index=FEATURES_ALL).sort_values(ascending=False)
print("\nFeature importance (gain relativo, seed 0):", flush=True)
print((100 * imp / imp.sum()).round(1).to_string(), flush=True)

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

for v in FEATURE_VARS:
    print(f"Processando teste: {v}...", flush=True)
    da_full = load(f"treino_{v}.nc")[v].sel(time=slice(f"{YEAR_START}-01-01", None))
    clim_v = da_full.groupby("time.month").mean("time")
    anom_test_v = ds_test[v] - clim_v.sel(month=month_origin_test)
    Xt[f"anom_{v}"] = to2d(anom_test_v, n_time_test).reshape(-1)
    del da_full, clim_v, anom_test_v
    gc.collect()

df_test = pd.DataFrame(Xt)[FEATURES_ALL]

pred_resid_test = np.mean([m.predict(df_test) for m in final_models], axis=0)
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
