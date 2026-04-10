import pandas as pd

# -----------------------------
# 1. LOAD RAW DATA
# -----------------------------
df_full = pd.read_csv("data/MI.data", header=None)

# -----------------------------
# 2. MAP ADMISSION / HISTORY VARIABLES + OUTCOMES
# -----------------------------
column_map = {
    0: "ID",
    1: "AGE",
    2: "SEX",
    3: "INF_ANAM",
    4: "STENOK_AN",
    5: "FK_STENOK",
    6: "IBS_POST",
    7: "IBS_NASL",
    8: "GB",
    9: "SIM_GIPERT",
    10: "DLIT_AG",
    11: "ZSN_A",
    12: "nr11",        # Observing of arrhythmia in anamnesis
    13: "nr01",        # Premature atrial contractions in anamnesis
    14: "nr02",        # Premature ventricular contractions in anamnesis
    15: "nr03",        # Paroxysms of atrial fibrillation in anamnesis
    16: "nr04",        # Persistent atrial fibrillation in anamnesis
    17: "nr07",
    18: "nr08",
    19: "np01",
    20: "np04",
    21: "np05",
    22: "np07",
    23: "np08",
    24: "np09",
    25: "np10",
    26: "endocr_01",   # Diabetes mellitus in anamnesis
    27: "endocr_02",   # Obesity in anamnesis
    28: "endocr_03",   # Thyrotoxicosis in anamnesis
    29: "zab_leg_01",
    30: "zab_leg_02",
    31: "zab_leg_03",
    32: "zab_leg_04",
    33: "zab_leg_06",
    34: "S_AD_KBRIG",  # Systolic BP by emergency team
    35: "D_AD_KBRIG",  # Diastolic BP by emergency team

    # Outcomes / complications
    112: "FIBR_PREDS",   # Atrial fibrillation
    113: "PREDS_TAH",
    114: "JELUD_TAH",
    115: "FIBR_JELUD",
    116: "A_V_BLOK",
    117: "OTEK_LANC",
    118: "RAZRIV",
    119: "DRESSLER",
    120: "ZSN",
    121: "REC_IM",
    122: "P_IM_STEN",
    123: "LET_IS"
}

df_full = df_full.rename(columns=column_map)

# -----------------------------
# 3. BASIC CHECKS ON FULL DATA
# -----------------------------
print("FULL DATA SHAPE:", df_full.shape)

print("\nFIBR_PREDS COUNTS:")
print(df_full["FIBR_PREDS"].value_counts())

af_rate = df_full["FIBR_PREDS"].mean()
print(f"\nAF RATE: {af_rate:.4f}")
print(f"AF PERCENT: {af_rate * 100:.2f}%")

# -----------------------------
# 4. CREATE MODEL DATASET
#    (admission/history-focused model)
# -----------------------------
model_cols = [
    "AGE",
    "SEX",
    "INF_ANAM",
    "STENOK_AN",
    "FK_STENOK",
    "IBS_POST",
    "IBS_NASL",
    "GB",
    "SIM_GIPERT",
    "DLIT_AG",
    "ZSN_A",
    "nr11",
    "nr01",
    "nr02",
    "nr03",
    "nr04",
    "nr07",
    "nr08",
    "np01",
    "np04",
    "np05",
    "np07",
    "np08",
    "np09",
    "np10",
    "endocr_01",
    "endocr_02",
    "endocr_03",
    "zab_leg_01",
    "zab_leg_02",
    "zab_leg_03",
    "zab_leg_04",
    "zab_leg_06",
    "S_AD_KBRIG",
    "D_AD_KBRIG",
    "FIBR_PREDS"
]

df_model = df_full[model_cols].copy()

# -----------------------------
# 5. CLEAN MODEL DATASET
# -----------------------------
# Replace '?' with missing
df_model = df_model.replace("?", pd.NA)

# Convert everything possible to numeric
df_model = df_model.apply(pd.to_numeric, errors="coerce")

# Impute:
# - categorical/binary-ish vars -> mode
# - more continuous vars -> median
for col in df_model.columns:
    if col != "FIBR_PREDS":
        unique_non_missing = df_model[col].dropna().nunique()

        if unique_non_missing <= 5:
            mode_val = df_model[col].mode(dropna=True)
            if not mode_val.empty:
                df_model[col] = df_model[col].fillna(mode_val.iloc[0])
        else:
            df_model[col] = df_model[col].fillna(df_model[col].median())

# -----------------------------
# 6. FINAL CHECKS
# -----------------------------
print("\nMODEL DATA SHAPE:", df_model.shape)
print("TOTAL MISSING VALUES IN MODEL DATA:", df_model.isna().sum().sum())

print("\nMODEL DATA HEAD:")
print(df_model.head())

print("\nMISSING VALUES BY COLUMN (TOP 10):")
print(df_model.isna().sum().sort_values(ascending=False).head(10))

# -----------------------------
# 7. SPLIT X AND y
# -----------------------------
X = df_model.drop(columns=["FIBR_PREDS"])
y = df_model["FIBR_PREDS"]

print("\nX SHAPE:", X.shape)
print("y SHAPE:", y.shape)

# -----------------------------
# 8. TRAIN / TEST SPLIT
# -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# -----------------------------
# 9. SCALING
# -----------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled train shape:", X_train_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)