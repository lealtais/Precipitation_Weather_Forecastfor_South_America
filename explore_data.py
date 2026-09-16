# Exploração mais profunda dos dados: mapa de correlação ONI x chuva (pra
# visualizar o dipolo Norte/Sul do El Niño) e correlação entre as 9 variáveis
# atmosféricas (pra achar redundância).

import os
import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = r"C:\Users\China Link\.cache\kagglehub\competitions\previsao-climatica-de-precipitacao-sobre-a-america-do-sul"
OUT_DIR = r"C:\Users\China Link\Desktop\worcap-precipitacao"

FEATURE_VARS = [
    "t2", "cloud_cover", "shum_850", "surface_pressure",
    "u_850", "v_850", "temperature_850", "rel_hum_850", "geopotential_850",
]

EPOCH_YEAR = 1940


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


print("Carregando tp...", flush=True)
ds_tp = load("treino_tp.nc")["tp"]
lat = ds_tp.lat.values
lon = ds_tp.lon.values

clim_tp = ds_tp.groupby("time.month").mean("time")
anom_tp = (ds_tp.groupby("time.month") - clim_tp).values   # (996, lat, lon)

idx = month_index(pd.DatetimeIndex(ds_tp.time.values))
oni_lookup = load_oni(os.path.join(OUT_DIR, "oni.ascii.txt"), idx.max() + 1)
oni_series = oni_lookup[idx]   # (996,)

valid = ~np.isnan(oni_series)
oni_v = oni_series[valid]
anom_v = anom_tp[valid]   # (n_valid, lat, lon)

print(f"Meses válidos p/ correlação: {valid.sum()}", flush=True)

# --- correlação ONI x anomalia de chuva, ponto a ponto ---------------------
oni_c = oni_v - oni_v.mean()
anom_c = anom_v - anom_v.mean(axis=0, keepdims=True)
num = np.tensordot(oni_c, anom_c, axes=([0], [0]))
den = np.sqrt((oni_c ** 2).sum()) * np.sqrt((anom_c ** 2).sum(axis=0))
corr_map = num / np.where(den == 0, np.nan, den)   # (lat, lon)

print("\nCorrelação ONI x anomalia de chuva por faixa de latitude (média):", flush=True)
LAT_BANDS = [(-60, -30, "Sul"), (-30, -10, "Central"), (-10, 15, "Norte")]
for lo, hi, label in LAT_BANDS:
    m = (lat >= lo) & (lat < hi)
    print(f"  {label:10s} (lat {lo} a {hi}): corr média = {np.nanmean(corr_map[m, :]):+.3f}", flush=True)

print(f"\nCorrelação mínima (mais negativa): {np.nanmin(corr_map):+.3f} em "
      f"lat={lat[np.unravel_index(np.nanargmin(corr_map), corr_map.shape)[0]]:.2f}, "
      f"lon={lon[np.unravel_index(np.nanargmin(corr_map), corr_map.shape)[1]]:.2f}", flush=True)
print(f"Correlação máxima (mais positiva): {np.nanmax(corr_map):+.3f} em "
      f"lat={lat[np.unravel_index(np.nanargmax(corr_map), corr_map.shape)[0]]:.2f}, "
      f"lon={lon[np.unravel_index(np.nanargmax(corr_map), corr_map.shape)[1]]:.2f}", flush=True)

del anom_tp, anom_v, anom_c

# --- correlação entre as 9 variáveis atmosféricas (achar redundância) ------
print("\nCarregando variáveis atmosféricas p/ matriz de correlação (amostra)...", flush=True)
SAMPLE_TIME = slice(0, 996, 4)   # 1 a cada 4 meses, só p/ estimar correlação rápido
data_sample = {}
for v in FEATURE_VARS:
    da = load(f"treino_{v}.nc")[v].isel(time=SAMPLE_TIME)
    data_sample[v] = da.values.reshape(da.shape[0], -1).mean(axis=1)   # média espacial por mês, só p/ visão geral

df_sample = pd.DataFrame(data_sample)
print("\nCorrelação entre variáveis atmosféricas (média espacial mensal):", flush=True)
print(df_sample.corr().round(2).to_string(), flush=True)
