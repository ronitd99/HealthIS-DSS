"""
build_sqlite.py
One-time script to build database/patients.db from data/MI.data.
Run from the repo root: python database/build_sqlite.py
"""

import sqlite3
import pandas as pd
import os
import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "patients.db")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "MI.data")

# ── Load raw data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, header=None)
df = df.replace("?", pd.NA)
df = df.apply(pd.to_numeric, errors="coerce")

col_map = {
    0: "patient_id",
    1: "AGE", 2: "SEX",
    3: "INF_ANAM", 4: "STENOK_AN", 5: "FK_STENOK", 6: "IBS_POST",
    7: "IBS_NASL", 8: "GB", 9: "SIM_GIPERT", 10: "DLIT_AG", 11: "ZSN_A",
    12: "nr11", 13: "nr01", 14: "nr02", 15: "nr03", 16: "nr04",
    17: "nr07", 18: "nr08",
    19: "np01", 20: "np04", 21: "np05", 22: "np07", 23: "np08",
    24: "np09", 25: "np10",
    26: "endocr_01", 27: "endocr_02", 28: "endocr_03",
    29: "zab_leg_01", 30: "zab_leg_02", 31: "zab_leg_03",
    32: "zab_leg_04", 33: "zab_leg_06",
    34: "S_AD_KBRIG", 35: "D_AD_KBRIG",
    112: "fibr_preds",
}
df = df[list(col_map.keys())].rename(columns=col_map)
df["patient_id"] = range(1, len(df) + 1)

# ── Build SQLite ───────────────────────────────────────────────────────────────
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    age        REAL,
    sex        REAL
);
CREATE TABLE cv_history (
    patient_id INTEGER PRIMARY KEY,
    inf_anam INTEGER, stenok_an INTEGER, fk_stenok INTEGER,
    ibs_post INTEGER, ibs_nasl INTEGER, gb INTEGER,
    sim_gipert INTEGER, dlit_ag INTEGER, zsn_a INTEGER
);
CREATE TABLE arrhythmia_history (
    patient_id INTEGER PRIMARY KEY,
    nr11 INTEGER, nr01 INTEGER, nr02 INTEGER,
    nr03 INTEGER, nr04 INTEGER, nr07 INTEGER, nr08 INTEGER
);
CREATE TABLE conduction_history (
    patient_id INTEGER PRIMARY KEY,
    np01 INTEGER, np04 INTEGER, np05 INTEGER, np07 INTEGER,
    np08 INTEGER, np09 INTEGER, np10 INTEGER
);
CREATE TABLE endocrine_history (
    patient_id INTEGER PRIMARY KEY,
    endocr_01 INTEGER, endocr_02 INTEGER, endocr_03 INTEGER
);
CREATE TABLE lung_history (
    patient_id INTEGER PRIMARY KEY,
    zab_leg_01 INTEGER, zab_leg_02 INTEGER, zab_leg_03 INTEGER,
    zab_leg_04 INTEGER, zab_leg_06 INTEGER
);
CREATE TABLE admission_vitals (
    patient_id INTEGER PRIMARY KEY,
    s_ad_kbrig REAL, d_ad_kbrig REAL
);
CREATE TABLE outcomes (
    patient_id INTEGER PRIMARY KEY,
    fibr_preds INTEGER
);
""")

def insert(table, cols):
    sub = df[["patient_id"] + cols].copy()
    sub.columns = ["patient_id"] + [c.lower() for c in cols]
    sub = sub.where(pd.notnull(sub), None)
    rows = [tuple(r) for r in sub.itertuples(index=False)]
    placeholders = ",".join(["?"] * len(sub.columns))
    cur.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)

insert("patients",           ["AGE", "SEX"])
insert("cv_history",         ["INF_ANAM","STENOK_AN","FK_STENOK","IBS_POST","IBS_NASL","GB","SIM_GIPERT","DLIT_AG","ZSN_A"])
insert("arrhythmia_history", ["nr11","nr01","nr02","nr03","nr04","nr07","nr08"])
insert("conduction_history", ["np01","np04","np05","np07","np08","np09","np10"])
insert("endocrine_history",  ["endocr_01","endocr_02","endocr_03"])
insert("lung_history",       ["zab_leg_01","zab_leg_02","zab_leg_03","zab_leg_04","zab_leg_06"])
insert("admission_vitals",   ["S_AD_KBRIG","D_AD_KBRIG"])

# outcomes — use patient_id and fibr_preds
out = df[["patient_id", "fibr_preds"]].copy()
out = out.where(pd.notnull(out), None)
cur.executemany("INSERT INTO outcomes VALUES (?,?)",
                [tuple(r) for r in out.itertuples(index=False)])

conn.commit()
conn.close()
print(f"Done — {len(df)} patients written to {DB_PATH}")
