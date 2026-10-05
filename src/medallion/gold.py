"""
Gold Medallion Layer for MediNexus AI.
Produces business-ready, AI-ready, and role-specific analytical data models:
- gold_patient_360
- gold_admissions
- gold_clinical
- gold_laboratory
- gold_pharmacy
- gold_prescriptions
- gold_appointments
- gold_billing
- gold_hospital_operations
- gold_patient_risk_features
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.constants import (
    SILVER_DATA_DIR,
    GOLD_DATA_DIR,
    GOLD_TABLES,
)
from src.utils.database import execute_query, save_df_to_table, query_df
from src.utils.helpers import generate_uuid
from src.utils.logging_utils import get_logger

logger = get_logger("gold_transformation")


def load_silver_table(table_name: str, silver_dir: Path = SILVER_DATA_DIR) -> pd.DataFrame:
    """Helper to load a silver parquet table."""
    path = silver_dir / f"{table_name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Fallback to DuckDB silver table
    return query_df(f"SELECT * FROM silver_{table_name}")


def generate_gold_models(
    silver_dir: Path = SILVER_DATA_DIR,
    gold_dir: Path = GOLD_DATA_DIR,
    run_id: str = None,
) -> Dict[str, Any]:
    """
    Generate and persist all 10 Gold analytical models.
    """
    run_id = run_id or generate_uuid("run_gld")
    gold_dir.mkdir(parents=True, exist_ok=True)
    created_models = {}

    logger.info(f"Starting Gold Layer generation (Run ID: {run_id})")

    # Load required Silver datasets
    df_hospitals = load_silver_table("hospitals", silver_dir)
    df_departments = load_silver_table("departments", silver_dir)
    df_doctors = load_silver_table("doctors", silver_dir)
    df_patients = load_silver_table("patients", silver_dir)
    df_admissions = load_silver_table("admissions", silver_dir)
    df_diagnoses = load_silver_table("diagnoses", silver_dir)
    df_labs = load_silver_table("laboratory_results", silver_dir)
    df_meds = load_silver_table("medications", silver_dir)
    df_rx = load_silver_table("prescriptions", silver_dir)
    df_inv = load_silver_table("pharmacy_inventory", silver_dir)
    df_apts = load_silver_table("appointments", silver_dir)
    df_billing = load_silver_table("billing", silver_dir)

    # -------------------------------------------------------------
    # 1. GOLD_ADMISSIONS (with 30-day readmission calculation)
    # -------------------------------------------------------------
    df_adm = df_admissions.sort_values(by=["patient_id", "admission_date"]).copy()
    df_adm["admission_dt"] = pd.to_datetime(df_adm["admission_date"])
    df_adm["discharge_dt"] = pd.to_datetime(df_adm["discharge_date"])

    # Calculate gap to next admission for each patient
    df_adm["next_admission_dt"] = df_adm.groupby("patient_id")["admission_dt"].shift(-1)
    df_adm["days_to_next_admission"] = (df_adm["next_admission_dt"] - df_adm["discharge_dt"]).dt.days
    df_adm["readmitted_30d"] = np.where(
        (df_adm["days_to_next_admission"] >= 0) & (df_adm["days_to_next_admission"] <= 30),
        1,
        0,
    )
    df_adm.drop(columns=["next_admission_dt", "days_to_next_admission"], inplace=True)

    # Join with patient info, doctor, hospital, dept, billing
    df_gold_admissions = df_adm.merge(
        df_patients[["patient_id", "name", "age", "gender", "blood_group", "insurance_provider"]],
        on="patient_id",
        how="left",
    ).rename(columns={"name": "patient_name"})

    df_gold_admissions = df_gold_admissions.merge(
        df_doctors[["doctor_id", "name", "specialty"]],
        on="doctor_id",
        how="left",
    ).rename(columns={"name": "attending_doctor", "specialty": "doctor_specialty"})

    df_gold_admissions = df_gold_admissions.merge(
        df_departments[["department_id", "department_name"]],
        on="department_id",
        how="left",
    )

    df_gold_admissions = df_gold_admissions.merge(
        df_hospitals[["hospital_id", "hospital_name", "location"]],
        on="hospital_id",
        how="left",
    )

    df_gold_admissions = df_gold_admissions.merge(
        df_billing[["admission_id", "total_amount", "payment_status"]],
        on="admission_id",
        how="left",
    ).rename(columns={"total_amount": "billing_amount"})

    df_gold_admissions["_gold_created_at"] = datetime.now().isoformat()
    df_gold_admissions.to_parquet(gold_dir / "gold_admissions.parquet", index=False)
    save_df_to_table("gold_admissions", df_gold_admissions, overwrite=True)
    created_models["gold_admissions"] = len(df_gold_admissions)

    # -------------------------------------------------------------
    # 2. GOLD_CLINICAL
    # -------------------------------------------------------------
    df_gold_clinical = df_diagnoses.merge(
        df_gold_admissions[[
            "admission_id", "patient_id", "patient_name", "age", "gender",
            "admission_date", "discharge_date", "length_of_stay", "admission_type",
            "department_name", "attending_doctor"
        ]],
        on=["admission_id", "patient_id"],
        how="left",
    )
    df_gold_clinical["_gold_created_at"] = datetime.now().isoformat()
    df_gold_clinical.to_parquet(gold_dir / "gold_clinical.parquet", index=False)
    save_df_to_table("gold_clinical", df_gold_clinical, overwrite=True)
    created_models["gold_clinical"] = len(df_gold_clinical)

    # -------------------------------------------------------------
    # 3. GOLD_LABORATORY
    # -------------------------------------------------------------
    df_gold_laboratory = df_labs.merge(
        df_patients[["patient_id", "name", "age", "gender"]],
        on="patient_id",
        how="left",
    ).rename(columns={"name": "patient_name"})

    df_gold_laboratory = df_gold_laboratory.merge(
        df_gold_admissions[["admission_id", "department_name", "attending_doctor"]],
        on="admission_id",
        how="left",
    )
    df_gold_laboratory["is_abnormal"] = df_gold_laboratory["abnormal_flag"].isin(["High", "Low", "Critical"]).astype(int)
    df_gold_laboratory["is_critical"] = (df_gold_laboratory["abnormal_flag"] == "Critical").astype(int)
    df_gold_laboratory["_gold_created_at"] = datetime.now().isoformat()
    df_gold_laboratory.to_parquet(gold_dir / "gold_laboratory.parquet", index=False)
    save_df_to_table("gold_laboratory", df_gold_laboratory, overwrite=True)
    created_models["gold_laboratory"] = len(df_gold_laboratory)

    # -------------------------------------------------------------
    # 4. GOLD_PRESCRIPTIONS
    # -------------------------------------------------------------
    df_gold_prescriptions = df_rx.merge(
        df_meds[["medication_id", "medication_name", "category", "unit_cost"]],
        on="medication_id",
        how="left",
    ).rename(columns={"category": "medication_category"})

    df_gold_prescriptions = df_gold_prescriptions.merge(
        df_patients[["patient_id", "name", "age", "gender"]],
        on="patient_id",
        how="left",
    ).rename(columns={"name": "patient_name"})

    df_gold_prescriptions["estimated_cost"] = df_gold_prescriptions["quantity"] * df_gold_prescriptions["unit_cost"]
    df_gold_prescriptions["_gold_created_at"] = datetime.now().isoformat()
    df_gold_prescriptions.to_parquet(gold_dir / "gold_prescriptions.parquet", index=False)
    save_df_to_table("gold_prescriptions", df_gold_prescriptions, overwrite=True)
    created_models["gold_prescriptions"] = len(df_gold_prescriptions)

    # -------------------------------------------------------------
    # 5. GOLD_PHARMACY (Inventory & Demand aggregation)
    # -------------------------------------------------------------
    rx_agg = df_rx.groupby("medication_id").agg(
        total_dispensed_qty=("quantity", "sum"),
        prescription_count=("prescription_id", "count"),
    ).reset_index()

    df_gold_pharmacy = df_inv.merge(
        df_meds[["medication_id", "medication_name", "category", "unit_cost", "reorder_threshold"]],
        on="medication_id",
        how="left",
    )

    df_gold_pharmacy = df_gold_pharmacy.merge(
        df_hospitals[["hospital_id", "hospital_name"]],
        on="hospital_id",
        how="left",
    )

    df_gold_pharmacy = df_gold_pharmacy.merge(rx_agg, on="medication_id", how="left")
    df_gold_pharmacy["total_dispensed_qty"] = df_gold_pharmacy["total_dispensed_qty"].fillna(0)
    df_gold_pharmacy["prescription_count"] = df_gold_pharmacy["prescription_count"].fillna(0)

    # Calculate average daily consumption rate (over simulated ~180 days)
    df_gold_pharmacy["estimated_daily_burn"] = (df_gold_pharmacy["total_dispensed_qty"] / 180.0).clip(lower=0.5).round(2)
    df_gold_pharmacy["days_of_supply_remaining"] = (
        df_gold_pharmacy["stock_quantity"] / df_gold_pharmacy["estimated_daily_burn"]
    ).round(1)

    df_gold_pharmacy["stockout_risk"] = np.where(
        df_gold_pharmacy["stock_quantity"] <= df_gold_pharmacy["reorder_level"],
        "High Risk",
        np.where(df_gold_pharmacy["days_of_supply_remaining"] < 14, "Moderate Risk", "Safe"),
    )

    suggested_qty = np.where(
        df_gold_pharmacy["stockout_risk"] != "Safe",
        (df_gold_pharmacy["reorder_level"] * 2) - df_gold_pharmacy["stock_quantity"],
        0,
    )
    df_gold_pharmacy["suggested_reorder_qty"] = np.maximum(suggested_qty, 0)

    df_gold_pharmacy["_gold_created_at"] = datetime.now().isoformat()
    df_gold_pharmacy.to_parquet(gold_dir / "gold_pharmacy.parquet", index=False)
    save_df_to_table("gold_pharmacy", df_gold_pharmacy, overwrite=True)
    created_models["gold_pharmacy"] = len(df_gold_pharmacy)

    # -------------------------------------------------------------
    # 6. GOLD_APPOINTMENTS
    # -------------------------------------------------------------
    df_gold_appointments = df_apts.merge(
        df_patients[["patient_id", "name", "age", "gender", "contact"]],
        on="patient_id",
        how="left",
    ).rename(columns={"name": "patient_name"})

    df_gold_appointments = df_gold_appointments.merge(
        df_doctors[["doctor_id", "name", "specialty"]],
        on="doctor_id",
        how="left",
    ).rename(columns={"name": "doctor_name"})

    df_gold_appointments = df_gold_appointments.merge(
        df_departments[["department_id", "department_name"]],
        on="department_id",
        how="left",
    )

    df_gold_appointments["is_completed"] = (df_gold_appointments["status"] == "Completed").astype(int)
    df_gold_appointments["is_no_show"] = (df_gold_appointments["status"] == "No-Show").astype(int)
    df_gold_appointments["is_cancelled"] = (df_gold_appointments["status"] == "Cancelled").astype(int)
    df_gold_appointments["_gold_created_at"] = datetime.now().isoformat()
    df_gold_appointments.to_parquet(gold_dir / "gold_appointments.parquet", index=False)
    save_df_to_table("gold_appointments", df_gold_appointments, overwrite=True)
    created_models["gold_appointments"] = len(df_gold_appointments)

    # -------------------------------------------------------------
    # 7. GOLD_BILLING
    # -------------------------------------------------------------
    df_gold_billing = df_billing.merge(
        df_gold_admissions[[
            "admission_id", "patient_id", "patient_name", "admission_date",
            "discharge_date", "length_of_stay", "department_name", "hospital_name"
        ]],
        on=["admission_id", "patient_id"],
        how="left",
    )
    df_gold_billing["collection_rate"] = (
        (df_gold_billing["total_amount"] - df_gold_billing["patient_payable"]) / df_gold_billing["total_amount"]
    ).round(3)
    df_gold_billing["_gold_created_at"] = datetime.now().isoformat()
    df_gold_billing.to_parquet(gold_dir / "gold_billing.parquet", index=False)
    save_df_to_table("gold_billing", df_gold_billing, overwrite=True)
    created_models["gold_billing"] = len(df_gold_billing)

    # -------------------------------------------------------------
    # 8. GOLD_PATIENT_360 (Comprehensive Patient Profile)
    # -------------------------------------------------------------
    # Aggregated admission metrics per patient
    adm_agg = df_adm.groupby("patient_id").agg(
        total_admissions=("admission_id", "count"),
        total_los=("length_of_stay", "sum"),
        avg_los=("length_of_stay", "mean"),
        last_admission_date=("admission_date", "max"),
        prior_readmissions=("readmitted_30d", "sum"),
    ).reset_index()
    adm_agg["avg_los"] = adm_agg["avg_los"].round(1)

    # Aggregated diagnoses per patient
    dx_agg = df_diagnoses.groupby("patient_id").agg(
        total_diagnoses=("diagnosis_id", "count"),
        comorbidities_list=("diagnosis_description", lambda s: ", ".join(sorted(set(s))[:3])),
    ).reset_index()

    # Comorbidity indicator flags
    dx_codes = df_diagnoses.groupby("patient_id")["icd10_code"].apply(list).reset_index()
    dx_codes["has_hypertension"] = dx_codes["icd10_code"].apply(lambda codes: int(any(c.startswith("I10") for c in codes)))
    dx_codes["has_diabetes"] = dx_codes["icd10_code"].apply(lambda codes: int(any(c.startswith("E11") for c in codes)))
    dx_codes["has_copd"] = dx_codes["icd10_code"].apply(lambda codes: int(any(c.startswith("J44") for c in codes)))
    dx_codes["has_heart_failure"] = dx_codes["icd10_code"].apply(lambda codes: int(any(c.startswith("I50") for c in codes)))
    dx_codes.drop(columns=["icd10_code"], inplace=True)

    # Aggregated lab metrics per patient
    lab_agg = df_gold_laboratory.groupby("patient_id").agg(
        total_lab_tests=("lab_id", "count"),
        abnormal_lab_count=("is_abnormal", "sum"),
        critical_lab_count=("is_critical", "sum"),
        latest_lab_date=("test_date", "max"),
    ).reset_index()

    # Aggregated prescription metrics per patient
    rx_patient_agg = df_rx.groupby("patient_id").agg(
        total_prescriptions=("prescription_id", "count"),
        active_prescriptions_count=("status", lambda s: (s == "Active").sum()),
    ).reset_index()

    # Aggregated billing metrics per patient
    bill_agg = df_billing.groupby("patient_id").agg(
        lifetime_billing=("total_amount", "sum"),
        outstanding_due=("patient_payable", "sum"),
    ).reset_index()

    # Merge into gold_patient_360
    df_p360 = df_patients.copy()
    df_p360 = df_p360.merge(adm_agg, on="patient_id", how="left")
    df_p360 = df_p360.merge(dx_agg, on="patient_id", how="left")
    df_p360 = df_p360.merge(dx_codes, on="patient_id", how="left")
    df_p360 = df_p360.merge(lab_agg, on="patient_id", how="left")
    df_p360 = df_p360.merge(rx_patient_agg, on="patient_id", how="left")
    df_p360 = df_p360.merge(bill_agg, on="patient_id", how="left")

    # Impute missing aggregates for patients without admissions yet
    df_p360["total_admissions"] = df_p360["total_admissions"].fillna(0).astype(int)
    df_p360["total_los"] = df_p360["total_los"].fillna(0).astype(int)
    df_p360["avg_los"] = df_p360["avg_los"].fillna(0.0)
    df_p360["prior_readmissions"] = df_p360["prior_readmissions"].fillna(0).astype(int)
    df_p360["total_diagnoses"] = df_p360["total_diagnoses"].fillna(0).astype(int)
    df_p360["comorbidities_list"] = df_p360["comorbidities_list"].fillna("None recorded")
    df_p360["has_hypertension"] = df_p360["has_hypertension"].fillna(0).astype(int)
    df_p360["has_diabetes"] = df_p360["has_diabetes"].fillna(0).astype(int)
    df_p360["has_copd"] = df_p360["has_copd"].fillna(0).astype(int)
    df_p360["has_heart_failure"] = df_p360["has_heart_failure"].fillna(0).astype(int)
    df_p360["total_lab_tests"] = df_p360["total_lab_tests"].fillna(0).astype(int)
    df_p360["abnormal_lab_count"] = df_p360["abnormal_lab_count"].fillna(0).astype(int)
    df_p360["critical_lab_count"] = df_p360["critical_lab_count"].fillna(0).astype(int)
    df_p360["total_prescriptions"] = df_p360["total_prescriptions"].fillna(0).astype(int)
    df_p360["active_prescriptions_count"] = df_p360["active_prescriptions_count"].fillna(0).astype(int)
    df_p360["lifetime_billing"] = df_p360["lifetime_billing"].fillna(0.0).round(2)
    df_p360["outstanding_due"] = df_p360["outstanding_due"].fillna(0.0).round(2)

    # Clinical risk classification
    def calculate_risk(row):
        score = 0
        if row["age"] >= 65:
            score += 2
        elif row["age"] >= 50:
            score += 1
        if row["total_admissions"] >= 3:
            score += 3
        elif row["total_admissions"] >= 2:
            score += 1
        if row["prior_readmissions"] >= 1:
            score += 3
        if (row["has_hypertension"] + row["has_diabetes"] + row["has_copd"] + row["has_heart_failure"]) >= 2:
            score += 2
        if row["critical_lab_count"] >= 1:
            score += 2
        elif row["abnormal_lab_count"] >= 3:
            score += 1

        if score >= 6:
            return "High"
        elif score >= 3:
            return "Moderate"
        return "Low"

    df_p360["clinical_risk_tier"] = df_p360.apply(calculate_risk, axis=1)
    df_p360["_gold_created_at"] = datetime.now().isoformat()
    df_p360.to_parquet(gold_dir / "gold_patient_360.parquet", index=False)
    save_df_to_table("gold_patient_360", df_p360, overwrite=True)
    created_models["gold_patient_360"] = len(df_p360)

    # -------------------------------------------------------------
    # 9. GOLD_HOSPITAL_OPERATIONS
    # -------------------------------------------------------------
    ops = df_gold_admissions.groupby(["admission_date", "hospital_id", "hospital_name", "department_name"]).agg(
        daily_admissions=("admission_id", "count"),
        avg_stay_days=("length_of_stay", "mean"),
        emergency_admissions=("admission_type", lambda s: (s == "Emergency").sum()),
        daily_billed_sum=("billing_amount", "sum"),
    ).reset_index()

    ops["avg_stay_days"] = ops["avg_stay_days"].round(1)
    ops["daily_billed_sum"] = ops["daily_billed_sum"].round(2)
    # Estimated department bed occupancy
    ops["estimated_occupancy_rate"] = ((ops["daily_admissions"] * ops["avg_stay_days"]) / 45.0).clip(lower=0.45, upper=0.98).round(3)
    ops["_gold_created_at"] = datetime.now().isoformat()
    ops.to_parquet(gold_dir / "gold_hospital_operations.parquet", index=False)
    save_df_to_table("gold_hospital_operations", ops, overwrite=True)
    created_models["gold_hospital_operations"] = len(ops)

    # -------------------------------------------------------------
    # 10. GOLD_PATIENT_RISK_FEATURES (Curated Feature Set for ML)
    # -------------------------------------------------------------
    df_features = df_gold_admissions[[
        "admission_id", "patient_id", "age", "gender", "admission_type",
        "length_of_stay", "readmitted_30d"
    ]].copy()

    # Join patient 360 attributes
    df_features = df_features.merge(
        df_p360[[
            "patient_id", "total_admissions", "total_los", "avg_los",
            "has_hypertension", "has_diabetes", "has_copd", "has_heart_failure",
            "total_diagnoses", "abnormal_lab_count", "critical_lab_count",
            "active_prescriptions_count", "lifetime_billing"
        ]],
        on="patient_id",
        how="left",
    )

    # Encode categoricals cleanly
    df_features["gender_encoded"] = np.where(df_features["gender"] == "Male", 1, 0)
    df_features["is_emergency"] = np.where(df_features["admission_type"] == "Emergency", 1, 0)
    df_features["is_urgent"] = np.where(df_features["admission_type"] == "Urgent", 1, 0)

    # Feature column selection
    df_features = df_features.fillna(0)
    df_features["_gold_created_at"] = datetime.now().isoformat()
    df_features.to_parquet(gold_dir / "gold_patient_risk_features.parquet", index=False)
    save_df_to_table("gold_patient_risk_features", df_features, overwrite=True)
    created_models["gold_patient_risk_features"] = len(df_features)

    logger.info(f"Gold Layer Generation Complete. Models created: {list(created_models.keys())}")

    return {
        "run_id": run_id,
        "status": "SUCCESS",
        "models": created_models,
        "total_gold_tables": len(created_models),
        "generated_at": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    generate_gold_models()
