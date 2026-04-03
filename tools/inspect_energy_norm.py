import json
import pandas as pd
import numpy as np

p = json.load(open('figures/dwave_qbsolv_progress.json'))
df = pd.DataFrame(p)
print('columns:', df.columns.tolist())

def _norm(col):
    s = col.astype(float)
    mn, mx = s.min(), s.max()
    print('mn,mx =', mn, mx)
    if mx == mn:
        return np.zeros_like(s)
    return (s - mn) / (mx - mn)

if 'energy' in df.columns:
    energy_norm = _norm(df['energy'])
elif 'energy_norm' in df.columns:
    energy_norm = df['energy_norm'].astype(float)
else:
    energy_norm = np.zeros(len(df))

print('energy_norm sample (first 20):', [round(float(x),4) for x in energy_norm[:20]])
print('energy_norm min,max:', float(energy_norm.min()), float(energy_norm.max()))
print('count>0:', int((energy_norm>0).sum()))
