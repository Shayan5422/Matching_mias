# -*- coding: utf-8 -*-
"""
Script: compare_columns.py
Goal  : برای هر ستون از ALLOWED_COLUMNS نشان می‌دهد چه درصدی از ردیف‌های
        تطابق‌یافته (بر اساس فایل لاگ) در دو فایل مقدار یکسان دارند.
"""

import pandas as pd

# ------------------------------------------------------------------ ثابت‌ها
ENCODING        = "utf-8"
DATE_FORMAT     = "%m/%d/%y"
UPLOADED_ID_COL = "id"

ALLOWED_COLUMNS = [
    "decennie_naiss",
    "ghm_prefix",
    "sexe",
    "age_gestationnel_weeks",
    "entree_mode",
    "entree_provenance",
    "sortie_mode",
    "sortie_destination",
    "nb_rea",
    "nb_si",
    "delta_days",
]

# ------------------------------------------------------------ توابع کمکی قبلی
import numpy as np
import pandas as pd

def calculate_annee_naiss(annee_series: pd.Series,
                             age_series: pd.Series,
                             *,
                             floor_age: bool = True) -> pd.Series:
    """
    برمی‌گرداند دههٔ تولد = ⌊(سال − سن) / 10⌋ × 10
    مثال: 1997 ← 1990
    """
    year_num = pd.to_numeric(annee_series, errors="coerce")
    age_num  = pd.to_numeric(age_series,  errors="coerce")
    if floor_age:
        age_num = np.floor(age_num)

    birth_year = year_num - age_num
    return (birth_year // 10) * 10


def calculate_delta_days(entree, sortie):
    ent_dt = pd.to_datetime(entree, errors="coerce")
    sor_dt = pd.to_datetime(sortie, errors="coerce")
    return (sor_dt - ent_dt).dt.days

def extract_gestational_weeks(series):
    def get_w(val):
        if pd.isna(val):
            return pd.NA
        s = str(val).split("+")[0].split()[0]
        digits = "".join(filter(str.isdigit, s))
        return int(digits) if digits else pd.NA
    return series.apply(get_w)

# ------------------------------------------------------- پیش‌پردازش دیتا فریم
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ستون‌های پایه که ممکن است موجود نباشند
    base_cols = [
        "annee","age","ghm","entree_date","sortie_date","age_gestationnel",
        "sexe","entree_mode","entree_provenance","sortie_mode",
        "sortie_destination","nb_rea","nb_si",UPLOADED_ID_COL
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # مشتق‌ها
    df["decennie_naiss"]            = calculate_annee_naiss(df["annee"], df["age"])
    df["ghm_prefix"]             = df["ghm"].astype(str).str[:3].replace("nan", None)
    df["age_gestationnel_weeks"] = extract_gestational_weeks(df["age_gestationnel"])
    df["entree_date"]            = pd.to_datetime(df["entree_date"], format=DATE_FORMAT,
                                                  errors="coerce")
    df["sortie_date"]            = pd.to_datetime(df["sortie_date"], format=DATE_FORMAT,
                                                  errors="coerce")
    df["delta_days"]             = calculate_delta_days(df["entree_date"], df["sortie_date"])

    df[UPLOADED_ID_COL] = df[UPLOADED_ID_COL].astype(str)

    # اطمینان از وجود همه‌ی ستون‌های مجاز
    for c in ALLOWED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    return df

# ------------------------------------------------------------- تابع اصلی کار
def column_agreement_stats(
    comparison_csv: str,
    entree_csv: str,
    log_csv: str,
    *,
    print_table: bool = True,
) -> pd.DataFrame:
    """
    بر اساس جدول log (دارای ستون‌های sortie_id و entre_id) درصد برابری
    هر ستونِ ALLOWED_COLUMNS را حساب می‌کند.
    خروجی: DataFrame با ستون‌های total_pairs، equal_pairs و percent_equal
    """

    # ۱) بارگذاری فایل‌ها
    comp_raw = pd.read_csv(comparison_csv, encoding=ENCODING, low_memory=False)
    ent_raw  = pd.read_csv(entree_csv,     encoding=ENCODING, low_memory=False)
    log_df   = pd.read_csv(log_csv,        encoding=ENCODING, low_memory=False)

    if {"sortie_id", "entre_id"}.issubset(log_df.columns):
        log_df = log_df.rename(columns={"sortie_id": "comparison_id",
                                        "entre_id":  "entree_id"})
    else:
        raise ValueError("log file needs columns sortie_id and entre_id")

    # ۲) پیش‌پردازش
    comp = preprocess(comp_raw)
    ent  = preprocess(ent_raw)

    comp_ids = comp.set_index(UPLOADED_ID_COL)
    ent_ids  = ent.set_index(UPLOADED_ID_COL)

    # ۳) محاسبه درصد تطابق در هر ستون
    stats = []
    for col in ALLOWED_COLUMNS:
        equal_count  = 0
        total_count  = 0

        for _, row in log_df.iterrows():
            cid = str(row["comparison_id"])
            eid = str(row["entree_id"])

            if cid not in comp_ids.index or eid not in ent_ids.index:
                # در صورتی که رکورد در یکی از فایل‌ها پیدا نشود، صرف‌نظر می‌کنیم
                continue

            c_val = comp_ids.at[cid, col]
            e_val = ent_ids.at[eid, col]

            # شرط «هر دو NaN یا هردو مساوی»
            if (pd.isna(c_val) and pd.isna(e_val)) or (c_val == e_val):
                equal_count += 1

            total_count += 1

        percent = 0 if total_count == 0 else (equal_count / total_count) * 100
        stats.append({
            "column": col,
            "total_pairs": total_count,
            "equal_pairs": equal_count,
            "percent_equal": round(percent, 2)
        })

    stats_df = pd.DataFrame(stats).sort_values("percent_equal", ascending=False)

    if print_table:
        print("\n=== Column Agreement Statistics ===")
        print(stats_df.to_string(index=False))
        print("====================================\n")

    return stats_df

# --------------------------------------------------------- اجرای مستقیم فایل
if __name__ == "__main__":
    # <-- مسیر فایل‌های خود را اینجا تنظیم کنید -->
    comparison_csv = "export_sortie1.csv"
    entree_csv     = "export_entree.csv"
    log_csv        = "log_sortie1.csv"

    column_agreement_stats(comparison_csv, entree_csv, log_csv)
