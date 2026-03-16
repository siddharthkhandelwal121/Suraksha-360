# make_quake_model.py
import pandas as pd, pickle, os
from sklearn.ensemble import RandomForestClassifier

BASE = os.path.abspath(os.path.dirname(__file__))
csv_path = os.path.join(BASE, "earthquake_dataset_basic.csv")
out_path = os.path.join(BASE, "earthquake_model.pkl")

# If CSV missing, create a tiny dummy dataset
if not os.path.exists(csv_path):
    import numpy as np
    df = pd.DataFrame({
        "magnitude": np.random.uniform(3.0,6.5,100),
        "depth": np.random.uniform(5,60,100),
    })
    df["event"] = (df["magnitude"] >= 5.0).astype(int)
    df.to_csv(csv_path, index=False)
else:
    df = pd.read_csv(csv_path)

X = df[["magnitude","depth"]].fillna(0)
y = df["event"].fillna(0).astype(int)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

pickle.dump({'model': model, 'feature_columns': X.columns.tolist()}, open(out_path, "wb"))
print("Saved", out_path)
