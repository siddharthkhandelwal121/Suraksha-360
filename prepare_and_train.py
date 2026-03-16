# prepare_and_train.py
import pandas as pd, numpy as np, pickle, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE = os.path.abspath(os.path.dirname(__file__))

# Input filenames (must be placed in same folder)
RAIN_CSV = os.path.join(BASE, "daily-rainfall-at-state-level.csv")
LANDS_CSV = os.path.join(BASE, "Global_Landslide_Catalog_Export_rows.csv")
QUAKE_CSV = os.path.join(BASE, "query.csv")

OUT_FOLDER = BASE  # save files here
os.makedirs(OUT_FOLDER, exist_ok=True)

# ---------- LOAD RAW FILES ----------
print("Loading raw files...")
rain = pd.read_csv(RAIN_CSV, low_memory=False)
lands = pd.read_csv(LANDS_CSV, low_memory=False)
eq = pd.read_csv(QUAKE_CSV, low_memory=False)

# ---------- PREPARE FLOOD DATASET ----------
print("Preparing flood dataset...")
rain['date'] = pd.to_datetime(rain['date'], errors='coerce').dt.date
flood = rain[['date','state_name','actual','normal','deviation']].copy()
flood.rename(columns={'actual':'rainfall_mm'}, inplace=True)
flood['rainfall_mm'] = flood['rainfall_mm'].fillna(0)
flood['normal'] = flood['normal'].fillna(0)
# crude flood label: rainfall >=100 mm OR rainfall > normal*5
flood['flood_event'] = ((flood['rainfall_mm'] >= 100) | (flood['rainfall_mm'] > (flood['normal']*5))).astype(int)
flood.to_csv(os.path.join(OUT_FOLDER, "flood_dataset.csv"), index=False)
print("Saved flood_dataset.csv")

# ---------- PREPARE LANDSLIDE DATASET ----------
print("Preparing landslide dataset (India subset + mapping)...")
lands['event_date_parsed'] = pd.to_datetime(lands.get('event_date', pd.Series()), errors='coerce')
lands['event_date_only'] = lands['event_date_parsed'].dt.date

# normalize state names helper
import unicodedata, re
def norm(s):
    if pd.isna(s): return ""
    s = str(s)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

rain_states = sorted(rain['state_name'].dropna().unique())
norm_rain = {norm(s): s for s in rain_states}
lands_ind = lands[lands['country_name'].str.lower() == 'india'].copy()
lands_ind['admin_norm'] = lands_ind['admin_division_name'].apply(norm)

from difflib import get_close_matches
mapping = {}
for a in lands_ind['admin_norm'].dropna().unique():
    if a in norm_rain:
        mapping[a] = norm_rain[a]
    else:
        match = get_close_matches(a, list(norm_rain.keys()), n=1, cutoff=0.6)
        mapping[a] = norm_rain[match[0]] if match else None
lands_ind['state_name_mapped'] = lands_ind['admin_norm'].map(mapping)
lands_events = lands_ind.dropna(subset=['state_name_mapped']).groupby(['event_date_only','state_name_mapped']).size().reset_index(name='n_events')
lands_events.rename(columns={'event_date_only':'date','state_name_mapped':'state_name'}, inplace=True)

merged = pd.merge(rain, lands_events, how='left', left_on=['date','state_name'], right_on=['date','state_name'])
merged['n_events'] = merged['n_events'].fillna(0).astype(int)
merged['landslide'] = (merged['n_events']>0).astype(int)
merged['rainfall_mm'] = merged['actual'].fillna(0)
merged = merged.sort_values(['state_name','date'])
# 3-day and 7-day antecedent rainfall
merged['rainfall_3d'] = merged.groupby('state_name')['rainfall_mm'].transform(lambda s: s.rolling(3, min_periods=1).sum())
merged['rainfall_7d'] = merged.groupby('state_name')['rainfall_mm'].transform(lambda s: s.rolling(7, min_periods=1).sum())

# synthesize slope/soil/veg proxies per state (so we can train)
soil_types = ['clay','loam','sand','silt']
np.random.seed(42)
state_slope = {s: (10 + np.random.rand()*20) for s in rain_states}
state_veg = {s: (0.3 + np.random.rand()*0.6) for s in rain_states}
def sample_soil(s):
    probs = np.array([0.25,0.25,0.25,0.25])
    return np.random.choice(soil_types, p=probs)
merged['soil'] = merged['state_name'].apply(lambda s: sample_soil(s) if pd.notna(s) else 'clay')
merged['slope'] = merged['state_name'].apply(lambda s: state_slope.get(s, 12))
merged['vegetation_index'] = merged['state_name'].apply(lambda s: state_veg.get(s, 0.5))

lands_out = merged[['date','state_name','rainfall_mm','rainfall_3d','rainfall_7d','slope','soil','vegetation_index','landslide']].copy()
lands_out.to_csv(os.path.join(OUT_FOLDER, "processed_landslide_dataset.csv"), index=False)
print("Saved processed_landslide_dataset.csv")

# ---------- PREPARE EARTHQUAKE DATA ----------
print("Preparing earthquake dataset...")
eq['time_parsed'] = pd.to_datetime(eq.get('time', pd.Series()), errors='coerce')
eq['date'] = eq['time_parsed'].dt.date
eq_out = eq[['time_parsed','latitude','longitude','depth','mag']].copy()
eq_out.rename(columns={'time_parsed':'time','mag':'magnitude'}, inplace=True)
eq_out['distance_to_fault_km'] = np.nan
eq_out['event'] = (eq_out['magnitude'].fillna(0) >= 5.0).astype(int)
eq_out.to_csv(os.path.join(OUT_FOLDER, "earthquake_dataset_basic.csv"), index=False)
print("Saved earthquake_dataset_basic.csv")

# ---------- TRAIN MODELS ----------
print("Training models...")

# Landslide model
L = lands_out.dropna(subset=['landslide']).copy()
# Build X for landslide: numeric + soil one-hot
Xl_num = L[['rainfall_mm','rainfall_3d','rainfall_7d','slope','vegetation_index']].fillna(0)
Xl = pd.concat([Xl_num, pd.get_dummies(L['soil'], prefix='soil')], axis=1)
yl = L['landslide'].astype(int)

print("Landslide shapes:", Xl.shape, yl.shape)
if len(yl) == 0:
    print("No positive landslide samples found; creating a trivial model")
    rf_l = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_l.fit(Xl.fillna(0), np.zeros(len(Xl)))
else:
    # if heavily imbalanced, do simple stratified split; you can later do SMOTE or class_weight
    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(Xl, yl, test_size=0.2, random_state=42, stratify=yl)
    rf_l = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_l.fit(Xl_tr, yl_tr)

pickle.dump({'model': rf_l, 'feature_columns': Xl.columns.tolist()}, open(os.path.join(OUT_FOLDER, "landslide_model.pkl"), "wb"))
pickle.dump({'model': rf_l, 'feature_columns': Xl.columns.tolist()}, open(os.path.join(OUT_FOLDER, "landslide_model_balanced.pkl"), "wb"))
print("Saved landslide_model.pkl and landslide_model_balanced.pkl")

# Flood model
# -----------------------
# Flood model (safe version)
# -----------------------
F = flood.copy()

# OPTIONAL: if you want to sample to speed up, do it here on F (keep X and y aligned).
# If you don't want sampling, leave this out.
# Example safe sampling (uncomment if needed):
# if len(F) > 200000:
#     F = F.sample(n=200000, random_state=42)

# Build X and y from the SAME DataFrame F so lengths always match
Xf = F[['rainfall_mm']].fillna(0)
yf = F['flood_event'].astype(int)

# Debug prints right before training — these will show in terminal
print("DEBUG: total flood rows (F):", len(F))
print("DEBUG: Xf.shape:", Xf.shape)
print("DEBUG: yf.shape:", yf.shape)

# Defensive assert to stop if lengths differ
assert len(Xf) == len(yf), f"Length mismatch: Xf {len(Xf)} vs yf {len(yf)}"

from sklearn.ensemble import RandomForestClassifier
rf_f = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

print("Training flood RandomForest on", Xf.shape[0], "samples ...")
rf_f.fit(Xf, yf)
print("Flood model trained.")

pickle.dump({'model': rf_f, 'feature_columns': Xf.columns.tolist()},
            open(os.path.join(OUT_FOLDER, "flood_model.pkl"), "wb"))
print("Saved flood_model.pkl")
