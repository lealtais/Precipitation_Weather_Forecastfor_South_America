import xarray as xr
import pandas as pd
import glob, os

DATA_DIR = r"C:\Users\China Link\.cache\kagglehub\competitions\previsao-climatica-de-precipitacao-sobre-a-america-do-sul"

for fname in sorted(os.listdir(DATA_DIR)):
    if fname.endswith(".nc"):
        path = os.path.join(DATA_DIR, fname)
        print("=" * 80)
        print(fname)
        ds = xr.open_dataset(path)
        print(ds)
        print()

print("=" * 80)
print("sample_submission.csv head")
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
print(sub.shape)
print(sub.head())
print(sub.tail())
