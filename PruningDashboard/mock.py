import json
import numpy as np

data = []
lams = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
accs = [85, 84, 82, 75, 60, 45, 20]
spars = [5, 15, 35, 60, 80, 95, 99]

for i, l in enumerate(lams):
    data.append({
        'lam': l,
        'accuracy': accs[i],
        'sparsity': spars[i],
        'layer_sparsities': [spars[i]*0.8, spars[i]*0.9, spars[i]*1.1, spars[i]*1.2],
        'hist_counts': np.random.randint(0, 100, 50).tolist(),
        'hist_bins': np.linspace(0, 1, 51).tolist()
    })

with open('dashboard_data.json', 'w') as f:
    json.dump(data, f)
