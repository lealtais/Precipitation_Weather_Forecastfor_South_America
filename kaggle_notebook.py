# ============================================================================
# WORCAP/INPE — Previsão climática de precipitação sobre a América do Sul
#
# DESATUALIZADO: esta versão usa objective="tweedie" sobre o valor absoluto,
# que testamos e piorou o resultado (ver README, "Tentativas descartadas").
# A versão que realmente evoluiu (resíduo + lags + suavização espacial +
# índice ONI + validação em anos de El Niño) está em `local_run.py`. Mantido
# aqui só como referência de como rodar via notebook do Kaggle em vez de
# localmente — precisaria ser atualizado com as mesmas features pra refletir
# o resultado atual (RMSE 1.8905 / leaderboard 1.9557).
#
# Estratégia (desta versão): prever a precipitação absoluta do mês seguinte
# com LightGBM (objective="tweedie"), usando como features a climatologia do
# mês alvo + anomalias das variáveis atmosféricas do mês atual + persistência
# (lags 0/1/2 meses).
#
# Como usar no Kaggle:
#   1. Novo Notebook -> Add Input -> busque a competição e adicione o dataset
#   2. Cole cada bloco "### CÉLULA N" numa célula separada (ou tudo numa só)
#   3. Accelerator: CPU básico já é suficiente
#   4. No fim, "Submit to Competition" usando o submission.csv gerado
# ============================================================================

### CÉLULA 1 — imports e paths
import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb

DATA_DIR = "/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul"

FEATURE_VARS = [
    "t2", "cloud_cover", "shum_850", "surface_pressure",
    "u_850", "v_850", "temperature_850", "rel_hum_850", "geopotential_850",
]

# Restringe o treino à era de satélite (ERA5 é mais confiável a partir daqui).
# Ajuste para None se quiser usar todo o histórico desde 1940 — mas nesse caso
# reduza também MAX_LAG abaixo, pois os 2 primeiros meses de 1940 não têm lag.
YEAR_START = 1979
MAX_LAG = 2   # quantos meses de persistência (lag) de anomalia de tp usar


def load(name):
    return xr.open_dataset(os.path.join(DATA_DIR, name))


ds_tp_all = load("treino_tp.nc")["tp"]          # série completa (sem corte), usada p/ montar os lags
ds_alvo = load("treino_tp_alvo.nc")["tp_alvo"]
ds_feats = {v: load(f"treino_{v}.nc")[v] for v in FEATURE_VARS}
ds_test = load("teste_features.nc")
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

ds_tp = ds_tp_all
if YEAR_START is not None:
    ds_tp = ds_tp.sel(time=slice(f"{YEAR_START}-01-01", None))
    ds_alvo = ds_alvo.sel(time=slice(f"{YEAR_START}-01-01", None))
    ds_feats = {v: da.sel(time=slice(f"{YEAR_START}-01-01", None)) for v, da in ds_feats.items()}

print("Treino:", ds_tp.sizes)
print("Teste:", ds_test.sizes)


### CÉLULA 2 — climatologia (média histórica por mês-calendário e ponto de grade)
clim_tp = ds_tp.groupby("time.month").mean("time")
clim_feats = {v: ds_feats[v].groupby("time.month").mean("time") for v in FEATURE_VARS}

# array (12, lat, lon) ordenado por mês 1..12, para indexação rápida por posição
clim_tp_arr = clim_tp.transpose("month", "lat", "lon").values


### CÉLULA 3 — série contínua de anomalia de tp (1940-01 até 2024-11) p/ montar os lags
# concatena o histórico de treino com as observações do "mês de origem" do teste
# (tp_ultima_obs), formando uma série mensal sem lacunas do início ao fim.
tp_full = ds_tp_all.values   # (996, lat, lon) — 1940-01 .. 2022-12, sempre sem corte de YEAR_START
tp_test_obs = ds_test["tp_ultima_obs"].values   # (24, lat, lon), meses 2022-12 .. 2024-11
tp_full_series = np.concatenate([tp_full, tp_test_obs[1:]], axis=0)   # 2022-12 já está em tp_full

month_pos = np.arange(tp_full_series.shape[0]) % 12   # 1940-01 é janeiro -> índice 0 = mês 1
anom_full_series = tp_full_series - clim_tp_arr[month_pos]

EPOCH_YEAR = 1940


def month_index(dates):
    dates = pd.to_datetime(dates)
    return (dates.year.values - EPOCH_YEAR) * 12 + (dates.month.values - 1)


### CÉLULA 4 — anomalias e lags no treino
month_t = ds_tp["time.month"]
month_next = ((month_t % 12) + 1)

anom_feats = {v: ds_feats[v] - clim_feats[v].sel(month=month_t) for v in FEATURE_VARS}
clim_tp_next = clim_tp.sel(month=month_next)   # baseline/climatologia do mês alvo

n_time = ds_tp.sizes["time"] - 1   # a última linha não tem alvo (mês seguinte não existe no arquivo)
assert bool(np.isnan(ds_alvo.isel(time=-1).values).all())

idx_t = month_index(pd.DatetimeIndex(ds_tp.time.values[:n_time]))
assert idx_t.min() - MAX_LAG >= 0, "YEAR_START cedo demais para o MAX_LAG escolhido"

lags_train = {lag: anom_full_series[idx_t - lag] for lag in range(0, MAX_LAG + 1)}


### CÉLULA 5 — achatar (flatten) treino em tabela para o LightGBM
lat = ds_tp.lat.values
lon = ds_tp.lon.values
LON2D, LAT2D = np.meshgrid(lon, lat)     # shape (lat, lon), mesma ordem das dims do xarray
lat_flat = LAT2D.reshape(-1).astype(np.float32)
lon_flat = LON2D.reshape(-1).astype(np.float32)
n_grid = lat_flat.shape[0]


def to2d(arr, n=n_time):
    values = arr.values if hasattr(arr, "values") else arr
    return values.reshape(values.shape[0], -1).astype(np.float32)[:n]


X = {}
for v in FEATURE_VARS:
    X[f"anom_{v}"] = to2d(anom_feats[v]).reshape(-1)
for lag, arr in lags_train.items():
    X[f"anom_tp_lag{lag}"] = to2d(arr).reshape(-1)

months_next_arr = month_next.values[:n_time].astype(np.float32)
X["lat"] = np.tile(lat_flat, n_time)
X["lon"] = np.tile(lon_flat, n_time)
X["month_sin"] = np.repeat(np.sin(2 * np.pi * months_next_arr / 12).astype(np.float32), n_grid)
X["month_cos"] = np.repeat(np.cos(2 * np.pi * months_next_arr / 12).astype(np.float32), n_grid)
X["clim_tp_next"] = to2d(clim_tp_next).reshape(-1)

time_idx = np.repeat(np.arange(n_time), n_grid)   # índice do mês de cada linha (para split temporal)

df_train = pd.DataFrame(X)
df_train["y_true"] = to2d(ds_alvo).reshape(-1)   # alvo em valor absoluto (mm/dia), não mais resíduo

del X, anom_feats, lags_train
gc.collect()

print(df_train.shape)
df_train.head()


### CÉLULA 6 — split temporal (últimos 5 anos como validação) e treino do LightGBM
FEATURES = [c for c in df_train.columns if c != "y_true"]

cutoff = n_time - 60   # últimos 60 meses (~5 anos) para validação
train_mask = time_idx < cutoff
valid_mask = ~train_mask

X_tr, y_tr = df_train.loc[train_mask, FEATURES], df_train.loc[train_mask, "y_true"]
X_va, y_va = df_train.loc[valid_mask, FEATURES], df_train.loc[valid_mask, "y_true"]

model = lgb.LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.3,   # entre 1 (Poisson) e 2 (Gamma); 1.2-1.5 costuma ir bem p/ chuva
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=255,
    max_depth=-1,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=200,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)


### CÉLULA 7 — validar: modelo bate a climatologia?
pred_full_va = np.clip(model.predict(X_va, num_iteration=model.best_iteration_), 0, None)
true_full_va = df_train.loc[valid_mask, "y_true"].values
clim_full_va = df_train.loc[valid_mask, "clim_tp_next"].values

rmse_model = np.sqrt(np.mean((pred_full_va - true_full_va) ** 2))
rmse_clim = np.sqrt(np.mean((clim_full_va - true_full_va) ** 2))
print(f"RMSE modelo:       {rmse_model:.4f} mm/dia")
print(f"RMSE climatologia: {rmse_clim:.4f} mm/dia")
print(f"Ganho sobre a climatologia: {100 * (1 - rmse_model / rmse_clim):.1f}%")


### CÉLULA 8 — montar features de teste (mesma lógica, alinhada ao mês de origem)
month_origin_test = ds_test["time_origem"].dt.month
month_target_test = ds_test["time"].dt.month

Xt = {}
for v in FEATURE_VARS:
    anom = ds_test[v] - clim_feats[v].sel(month=month_origin_test)
    Xt[f"anom_{v}"] = to2d(anom, n=ds_test.sizes["time"]).reshape(-1)

idx_origin_test = month_index(pd.DatetimeIndex(ds_test["time_origem"].values))
for lag in range(0, MAX_LAG + 1):
    Xt[f"anom_tp_lag{lag}"] = anom_full_series[idx_origin_test - lag].reshape(
        anom_full_series.shape[0] if False else len(idx_origin_test), -1
    ).astype(np.float32).reshape(-1)

n_time_test = ds_test.sizes["time"]
Xt["lat"] = np.tile(lat_flat, n_time_test)
Xt["lon"] = np.tile(lon_flat, n_time_test)

months_target_arr = month_target_test.values.astype(np.float32)
Xt["month_sin"] = np.repeat(np.sin(2 * np.pi * months_target_arr / 12).astype(np.float32), n_grid)
Xt["month_cos"] = np.repeat(np.cos(2 * np.pi * months_target_arr / 12).astype(np.float32), n_grid)

clim_tp_next_test = clim_tp.sel(month=month_target_test)
Xt["clim_tp_next"] = to2d(clim_tp_next_test, n=n_time_test).reshape(-1)

df_test = pd.DataFrame(Xt)[FEATURES]


### CÉLULA 9 — prever e montar o submission.csv
pred_final = np.clip(model.predict(df_test, num_iteration=model.best_iteration_), 0, None)

times = pd.to_datetime(ds_test["time"].values)
latlon_str = np.array([f"{la:.2f}_{lo:.2f}" for la, lo in zip(lat_flat, lon_flat)])
ids = np.concatenate([
    np.char.add(f"{t.year}_{t.month:02d}_", latlon_str) for t in times
])

df_pred = pd.DataFrame({"id": ids, "tp_mm_day": pred_final})

sub_out = sub[["id"]].merge(df_pred, on="id", how="left")
assert sub_out["tp_mm_day"].isna().sum() == 0, "faltou id no merge"

sub_out.to_csv("submission.csv", index=False)
sub_out.head()
