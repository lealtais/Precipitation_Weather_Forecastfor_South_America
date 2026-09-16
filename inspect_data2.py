import xarray as xr
import numpy as np
import os

DATA_DIR = r"C:\Users\China Link\.cache\kagglehub\competitions\previsao-climatica-de-precipitacao-sobre-a-america-do-sul"

print("### teste_features.nc — time / time_origem / lag_meses ###")
ds_test = xr.open_dataset(os.path.join(DATA_DIR, "teste_features.nc"))
print(ds_test[["time", "time_origem", "lag_meses"]].to_dataframe().reset_index()[["time","time_origem","lag_meses"]].drop_duplicates())

print()
print("### tp_alvo in test: any non-nan? ###")
print("all nan:", bool(np.isnan(ds_test["tp_alvo"].values).all()))
print("nan fraction:", np.isnan(ds_test["tp_alvo"].values).mean())

print()
print("### tp_ultima_obs in test: any non-nan? ###")
print("nan fraction:", np.isnan(ds_test["tp_ultima_obs"].values).mean())
print(ds_test["tp_ultima_obs"].isel(time=0).values[150,150], ds_test["tp_ultima_obs"].isel(time=1).values[150,150])

print()
print("### relationship tp(t+1) vs tp_alvo(t) in train ###")
ds_tp = xr.open_dataset(os.path.join(DATA_DIR, "treino_tp.nc"))
ds_alvo = xr.open_dataset(os.path.join(DATA_DIR, "treino_tp_alvo.nc"))
tp = ds_tp["tp"].values
alvo = ds_alvo["tp_alvo"].values
print("tp shape", tp.shape, "alvo shape", alvo.shape)
diff = tp[1:] - alvo[:-1]
print("max abs diff tp[t+1] vs alvo[t]:", np.nanmax(np.abs(diff)))
print("tp[0,150,150]:", tp[0,150,150], "alvo[0,150,150] (should ~ tp[1,150,150]):", alvo[0,150,150], "tp[1,150,150]:", tp[1,150,150])

print()
print("### units sanity: tp value range (train) ###")
print("min/max/mean tp:", np.nanmin(tp), np.nanmax(tp), np.nanmean(tp))
print("min/max/mean tp_alvo:", np.nanmin(alvo), np.nanmax(alvo), np.nanmean(alvo))

print()
print("### lat/lon match between test grid and sample_submission ids ###")
print("lat range test:", ds_test.lat.min().item(), ds_test.lat.max().item())
print("lon range test:", ds_test.lon.min().item(), ds_test.lon.max().item())
