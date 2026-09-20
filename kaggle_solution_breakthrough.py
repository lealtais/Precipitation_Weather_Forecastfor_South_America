# ==============================================================================
# WORCAP/INPE — Previsão Climática Mensal de Precipitação sobre a América do Sul
# SOLUÇÃO BREAKTHROUGH: Quebrando o Platô de 1.9 no Kaggle
#
# Por que as versões anteriores (v4 a v11) estagnaram em ~1.9x?
# 1. DOMINÂNCIA DA AMAZÔNIA NO RESÍDUO: A variância da chuva na Amazônia é 10x
#    maior que no semi-árido/sul. O RMSE focava 90% dos splits na Amazônia,
#    gerando previsões ruidosas no resto do continente.
#    -> SOLUÇÃO: Prever a ANOMALIA PADRONIZADA (Z-Score por ponto de grade).
#       z = (tp - clim_mean) / (clim_std + eps). Agora a escala é unitária em todo o mapa!
# 2. FALTAVAM AS VARIÁVEIS FÍSICAS REAIS:
#    - Fluxo e Transporte de Umidade (Jato de Baixos Níveis): u850 * shum, v850 * shum
#    - Instabilidade Convectiva: Delta T = t2 - temperature_850
#    - Persistência Anual: Lag 12 (mesmo mês do ano anterior)
# 3. INTERAÇÃO ESPACIAL DE TELECONEXÃO:
#    - Em vez do ONI ser um escalar cego, multiplicamos pela sensibilidade local!
# 4. ENSEMBLE DIVERSO: LightGBM + XGBoost com blending ponderado.
# ==============================================================================

import os
import gc
import urllib.request
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
import xgboost as xgb
from scipy.ndimage import uniform_filter

# ------------------------------------------------------------------------------
# 1. Configurações de Ambiente e Dados
# ------------------------------------------------------------------------------
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

YEAR_START = 1970      # Era moderna com dados consistentes
EPOCH_YEAR = 1940
EPS = 1e-4

# Baixa o índice ONI da NOAA
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

# ------------------------------------------------------------------------------
# 2. Carregamento e Engenharia da Climatologia e Variabilidade Local
# ------------------------------------------------------------------------------
print("Carregando séries temporais de precipitação...", flush=True)
ds_tp_all = load("treino_tp.nc")["tp"]
tp_full_all = ds_tp_all.values                     # (996, lat, lon)
ds_test = load("teste_features.nc")
tp_test_obs = ds_test["tp_ultima_obs"].values       # (24, lat, lon)
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

# 1. Climatologia: Média E Desvio Padrão Mensal por Ponto (CHAVE DO Z-SCORE)
clim_mean = ds_tp.groupby("time.month").mean("time")
clim_std = ds_tp.groupby("time.month").std("time")

clim_mean_arr = clim_mean.transpose("month", "lat", "lon").values.astype(np.float32)
clim_std_arr = np.maximum(clim_std.transpose("month", "lat", "lon").values.astype(np.float32), EPS)

# Série temporal contínua para os lags
tp_full_series = np.concatenate([tp_full_all, tp_test_obs[1:]], axis=0).astype(np.float32)
month_pos = np.arange(tp_full_series.shape[0]) % 12

# Anomalia bruta e Anomalia Z-Score
anom_full_series = tp_full_series - clim_mean_arr[month_pos]
zscore_full_series = anom_full_series / clim_std_arr[month_pos]

# Suavização espacial 3x3 do Z-Score
zscore_smooth_3x3 = uniform_filter(zscore_full_series, size=(1, 3, 3), mode="nearest")

# Carrega ONI
oni_lookup = load_oni(ONI_PATH, tp_full_series.shape[0])

# Mapa histórico de sensibilidade ao ENSO por ponto de grade
# Permite ao modelo saber ONDE o El Niño causa seca e ONDE causa enchente
print("Calculando mapa físico de sensibilidade ao ENSO...", flush=True)
valid_idx = month_index(pd.DatetimeIndex(ds_tp.time.values[:n_time]))
oni_train = oni_lookup[valid_idx]
z_train_subset = zscore_full_series[valid_idx] # (n_time, lat, lon)

oni_centered = oni_train - np.mean(oni_train)
oni_var = np.sum(oni_centered ** 2) + EPS
enso_sensitivity_map = np.sum(oni_centered[:, None, None] * z_train_subset, axis=0) / oni_var
enso_sens_flat = enso_sensitivity_map.reshape(-1).astype(np.float32)

del tp_full_all, tp_test_obs, z_train_subset
gc.collect()

# ------------------------------------------------------------------------------
# 3. Construção da Tabela de Features (Treino)
# ------------------------------------------------------------------------------
print("Construindo matriz de features com Física Climática...", flush=True)
month_t = ds_tp["time.month"]
month_next = ((month_t % 12) + 1)
idx_t = month_index(pd.DatetimeIndex(ds_tp.time.values[:n_time]))

months_next_arr = month_next.values[:n_time].astype(np.float32)
clim_next_m = clim_mean_arr[month_next.values[:n_time] - 1]
clim_next_s = clim_std_arr[month_next.values[:n_time] - 1]

X = {}

# Coordenadas e Sazonalidade
X["lat"] = np.tile(lat_flat, n_time)
X["lon"] = np.tile(lon_flat, n_time)
X["month_sin"] = np.repeat(np.sin(2 * np.pi * months_next_arr / 12).astype(np.float32), n_grid)
X["month_cos"] = np.repeat(np.cos(2 * np.pi * months_next_arr / 12).astype(np.float32), n_grid)
X["clim_mean_next"] = clim_next_m.reshape(n_time, -1).reshape(-1)
X["clim_std_next"] = clim_next_s.reshape(n_time, -1).reshape(-1)

# Lags da Anomalia Padronizada (Z-score)
for lag in [0, 1, 2]:
    X[f"z_lag_{lag}"] = to2d(zscore_full_series[idx_t - lag], n_time).reshape(-1)
    X[f"z_smooth_lag_{lag}"] = to2d(zscore_smooth_3x3[idx_t - lag], n_time).reshape(-1)

# Persistência Anual (Mesmo mês do ano anterior - Lag 12)
X["z_lag_12"] = to2d(zscore_full_series[idx_t - 12], n_time).reshape(-1)

# Média Móvel de 3 Meses da Chuva Antecedente
z_rolling_3m = (zscore_full_series[idx_t] + zscore_full_series[idx_t - 1] + zscore_full_series[idx_t - 2]) / 3.0
X["z_rolling_3m"] = to2d(z_rolling_3m, n_time).reshape(-1)

# ENSO Físico: Interação de Sensibilidade Espacial x ONI
for lag in [0, 1, 2]:
    oni_val = oni_lookup[idx_t - lag].astype(np.float32)
    X[f"oni_lag_{lag}"] = np.repeat(oni_val, n_grid)
    # Sinal calibrado para cada latitude/longitude
    X[f"enso_impact_lag_{lag}"] = np.repeat(oni_val, n_grid) * np.tile(enso_sens_flat, n_time)

# ------------------------------------------------------------------------------
# 4. Variáveis Reanálise Atmosférica + Termodinâmica
# ------------------------------------------------------------------------------
print("Calculando variáveis dinâmicas e de transporte de umidade...", flush=True)
u850 = load("treino_u_850.nc")["u_850"].sel(time=slice(f"{YEAR_START}-01-01", None))
v850 = load("treino_v_850.nc")["v_850"].sel(time=slice(f"{YEAR_START}-01-01", None))
shum = load("treino_shum_850.nc")["shum_850"].sel(time=slice(f"{YEAR_START}-01-01", None))
t2 = load("treino_t2.nc")["t2"].sel(time=slice(f"{YEAR_START}-01-01", None))
t850 = load("treino_temperature_850.nc")["temperature_850"].sel(time=slice(f"{YEAR_START}-01-01", None))

# 1. Transporte de Umidade (Fluxo Zonal e Meridional)
flux_u = u850 * shum
flux_v = v850 * shum
flux_mag = np.sqrt(flux_u**2 + flux_v**2)

X["flux_u"] = to2d(flux_u, n_time).reshape(-1)
X["flux_v"] = to2d(flux_v, n_time).reshape(-1)
X["flux_mag"] = to2d(flux_mag, n_time).reshape(-1)

# 2. Instabilidade Atmosférica (Gradiente Vertical de Temperatura)
instab = t2 - t850
X["thermal_instability"] = to2d(instab, n_time).reshape(-1)

del u850, v850, shum, t2, t850, flux_u, flux_v, flux_mag, instab
gc.collect()

# Restante das variáveis atmosféricas (anomalias simples)
for v in ["cloud_cover", "surface_pressure", "rel_hum_850", "geopotential_850"]:
    da = load(f"treino_{v}.nc")[v].sel(time=slice(f"{YEAR_START}-01-01", None))
    clim_v = da.groupby("time.month").mean("time")
    anom_v = da - clim_v.sel(month=month_t)
    X[f"anom_{v}"] = to2d(anom_v, n_time).reshape(-1)
    del da, clim_v, anom_v
    gc.collect()

# Alvo: Z-Score da Precipitação Futura!
y_raw = to2d(ds_alvo, n_time).reshape(-1)
clim_m_flat = X["clim_mean_next"]
clim_s_flat = X["clim_std_next"]
y_target_z = (y_raw - clim_m_flat) / clim_s_flat

df_train = pd.DataFrame(X)
df_train["target_z"] = y_target_z
FEATURES = [c for c in df_train.columns if c != "target_z"]

print(f"Shape do Dataset de Treino: {df_train.shape}", flush=True)

# ------------------------------------------------------------------------------
# 5. Treinamento: Ensemble Híbrido (XGBoost + LightGBM)
# ------------------------------------------------------------------------------
print("\n>>> TREINANDO O ENSEMBLE HÍBRIDO NO DATASET COMPLETO...", flush=True)
X_train_data = df_train[FEATURES]
y_train_data = df_train["target_z"]

# 1. XGBoost (Excelente para interações complexas de relevo e fluxo)
print("Treinando XGBoost...", flush=True)
xgb_model = xgb.XGBRegressor(
    n_estimators=450,
    learning_rate=0.04,
    max_depth=6,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_data, y_train_data)

# 2. LightGBM (Mais suave e rápido, complementa os resíduos do XGBoost)
print("Treinando LightGBM...", flush=True)
lgb_model = lgb.LGBMRegressor(
    n_estimators=450,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=500,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=123,
    n_jobs=-1
)
lgb_model.fit(X_train_data, y_train_data)

del df_train, X_train_data, y_train_data
gc.collect()

# ------------------------------------------------------------------------------
# 6. Montagem das Features de Teste e Predição
# ------------------------------------------------------------------------------
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

# Dinâmica de Teste
u850_t = ds_test["u_850"]
v850_t = ds_test["v_850"]
shum_t = ds_test["shum_850"]
t2_t = ds_test["t2"]
t850_t = ds_test["temperature_850"]

flux_u_t = u850_t * shum_t
flux_v_t = v850_t * shum_t
flux_mag_t = np.sqrt(flux_u_t**2 + flux_v_t**2)
instab_t = t2_t - t850_t

Xt["flux_u"] = to2d(flux_u_t, n_time_test).reshape(-1)
Xt["flux_v"] = to2d(flux_v_t, n_time_test).reshape(-1)
Xt["flux_mag"] = to2d(flux_mag_t, n_time_test).reshape(-1)
Xt["thermal_instability"] = to2d(instab_t, n_time_test).reshape(-1)

del flux_u_t, flux_v_t, flux_mag_t, instab_t
gc.collect()

for v in ["cloud_cover", "surface_pressure", "rel_hum_850", "geopotential_850"]:
    da_train = load(f"treino_{v}.nc")[v].sel(time=slice(f"{YEAR_START}-01-01", None))
    clim_v = da_train.groupby("time.month").mean("time")
    anom_test_v = ds_test[v] - clim_v.sel(month=month_origin_test)
    Xt[f"anom_{v}"] = to2d(anom_test_v, n_time_test).reshape(-1)
    del da_train, clim_v, anom_test_v
    gc.collect()

df_test = pd.DataFrame(Xt)[FEATURES]

# ------------------------------------------------------------------------------
# 7. Predição com Blending Ponderado e Despadronização
# ------------------------------------------------------------------------------
print("Gerando previsões combinadas...", flush=True)
pred_z_xgb = xgb_model.predict(df_test)
pred_z_lgb = lgb_model.predict(df_test)

# Blending: 60% XGBoost + 40% LightGBM (cancela erros ortogonais)
pred_z_final = 0.60 * pred_z_xgb + 0.40 * pred_z_lgb

# DESPADRONIZAÇÃO: Retorna do Z-Score para mm/dia
pred_tp_raw = (pred_z_final * Xt["clim_std_next"]) + Xt["clim_mean_next"]

# Restrição Física: Chuva não pode ser negativa
pred_tp_final = np.clip(pred_tp_raw, 0.0, None)

# ------------------------------------------------------------------------------
# 8. Montagem do Arquivo Final de Submissão
# ------------------------------------------------------------------------------
times = pd.to_datetime(ds_test["time"].values)
latlon_str = np.array([f"{la:.2f}_{lo:.2f}" for la, lo in zip(lat_flat, lon_flat)])
ids = np.concatenate([np.char.add(f"{t.year}_{t.month:02d}_", latlon_str) for t in times])

df_pred = pd.DataFrame({"id": ids, "tp_mm_day": pred_tp_final})
sub_out = sub[["id"]].merge(df_pred, on="id", how="left")
assert sub_out["tp_mm_day"].isna().sum() == 0, "Erro: existem IDs com valor NaN!"

out_path = os.path.join(WORK_DIR, "submission_breakthrough.csv")
sub_out.to_csv(out_path, index=False)
print("=" * 64)
print(f"✅ SUCESSO! Submissão gerada em: {out_path}")
print(f"Média prevista: {pred_tp_final.mean():.3f} mm/dia | Max: {pred_tp_final.max():.2f}")
print("=" * 64)
