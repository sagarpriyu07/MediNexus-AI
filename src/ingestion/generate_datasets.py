"""
Synthetic Healthcare Dataset Generator for MediNexus AI.
Generates 12 interconnected datasets with realistic clinical patterns and intentional data quality defects.
"""

import sys
import os
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from faker import Faker

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.constants import (
    RAW_DATA_DIR,
    METADATA_DIR,
    DEFAULT_COUNTS,
    BLOOD_GROUPS,
    GENDERS,
    ADMISSION_TYPES,
    DISCHARGE_DISPOSITIONS,
    PAYMENT_STATUSES,
    APPOINTMENT_STATUSES,
    DIAGNOSIS_TYPES,
    DIAGNOSIS_SEVERITIES,
    ICD10_CATALOG,
    LAB_CATALOG,
    MEDICATION_CATALOG,
    HOSPITAL_DEPARTMENTS,
)
from src.ingestion.ingestion_utils import inject_realistic_imperfections


def generate_healthcare_ecosystem(
    counts: dict = None,
    seed: int = 42,
    output_dir: Path = RAW_DATA_DIR,
) -> dict:
    """
    Generate 12 interconnected healthcare datasets with realistic data quality defects.
    """
    counts = counts or DEFAULT_COUNTS
    random.seed(seed)
    np.random.seed(seed)
    fake = Faker("en_US")
    Faker.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic healthcare ecosystem (Seed: {seed})...")

    # 1. Hospitals (5 records)
    num_hospitals = counts.get("hospitals", 5)
    hospitals = []
    hospital_names = [
        "Metropolitan General Hospital",
        "Saint Jude Health Pavilion",
        "Apex Memorial Medical Center",
        "Riverside University Hospital",
        "Beacon Hill Community Hospital",
    ]
    for i in range(num_hospitals):
        h_id = f"HOSP_{i+1:03d}"
        hospitals.append({
            "hospital_id": h_id,
            "hospital_name": hospital_names[i % len(hospital_names)],
            "location": fake.city() + ", " + fake.state_abbr(),
            "total_beds": random.choice([250, 400, 600, 750, 500]),
            "icu_beds": random.choice([30, 50, 80, 100, 60]),
            "operational_status": "Active",
        })
    df_hospitals = pd.DataFrame(hospitals)

    # 2. Departments
    departments = []
    dept_id_counter = 1
    for h in hospitals:
        for d_name in HOSPITAL_DEPARTMENTS:
            departments.append({
                "department_id": f"DEPT_{dept_id_counter:03d}",
                "hospital_id": h["hospital_id"],
                "department_name": d_name,
                "bed_capacity": random.randint(25, 80),
            })
            dept_id_counter += 1
    df_departments = pd.DataFrame(departments)

    # 3. Doctors (40 records)
    num_doctors = counts.get("doctors", 40)
    doctors = []
    specialties = [
        "Cardiology", "Internal Medicine", "Pulmonology", "Emergency Medicine",
        "General Surgery", "Oncology", "Endocrinology", "Nephrology"
    ]
    for i in range(num_doctors):
        doc_id = f"DOC_{i+1:03d}"
        assigned_dept = random.choice(departments)
        doctors.append({
            "doctor_id": doc_id,
            "hospital_id": assigned_dept["hospital_id"],
            "department_id": assigned_dept["department_id"],
            "name": f"Dr. {fake.first_name()} {fake.last_name()}, MD",
            "specialty": random.choice(specialties),
            "qualification": random.choice(["MD, FACC", "MD, FACP", "MD, FCCP", "MBBS, MD"]),
            "experience_years": random.randint(3, 35),
            "contact": fake.phone_number(),
        })
    df_doctors = pd.DataFrame(doctors)

    # 4. Patients
    num_patients = counts.get("patients", 3500)
    patients = []
    base_date = datetime(2023, 1, 1)
    insurance_providers = [
        "Blue Cross Blue Shield", "Aetna Health", "UnitedHealthcare",
        "Medicare", "Medicaid", "Cigna", "Humana"
    ]

    for i in range(num_patients):
        p_id = f"P_{i+1:05d}"
        age = random.choices([
            random.randint(18, 35),
            random.randint(36, 50),
            random.randint(51, 65),
            random.randint(66, 88),
        ], weights=[0.2, 0.25, 0.3, 0.25])[0]
        dob = (datetime.now() - timedelta(days=int(age * 365.25))).strftime("%Y-%m-%d")
        created_at = (base_date + timedelta(days=random.randint(0, 700))).strftime("%Y-%m-%d")
        gender = random.choices(GENDERS, weights=[0.48, 0.49, 0.03])[0]

        patients.append({
            "patient_id": p_id,
            "name": fake.name_male() if gender == "Male" else fake.name_female(),
            "dob": dob,
            "age": age,
            "gender": gender,
            "blood_group": random.choice(BLOOD_GROUPS),
            "contact": fake.phone_number(),
            "address": fake.street_address() + ", " + fake.city(),
            "insurance_provider": random.choice(insurance_providers),
            "emergency_contact": fake.phone_number(),
            "created_at": created_at,
        })
    df_patients = pd.DataFrame(patients)

    # 5. Medications Catalog (50 items)
    medications = []
    for idx, med_meta in enumerate(MEDICATION_CATALOG):
        med_id = f"MED_{idx+1:03d}"
        medications.append({
            "medication_id": med_id,
            "medication_name": med_meta["name"],
            "category": med_meta["category"],
            "dosage_form": med_meta["form"],
            "unit_cost": med_meta["unit_cost"],
            "reorder_threshold": med_meta["reorder"],
            "standard_daily_dose": med_meta["daily_dose"],
        })
    df_medications = pd.DataFrame(medications)

    # 6. Pharmacy Inventory
    pharmacy_inventory = []
    inv_id_counter = 1
    for h in hospitals:
        for med in medications:
            stock = random.randint(20, 1200)
            reorder = med["reorder_threshold"]
            # Occasional low stock scenario for demonstration
            if random.random() < 0.20:
                stock = random.randint(5, int(reorder * 0.4))

            exp_date = (datetime.now() + timedelta(days=random.randint(30, 730))).strftime("%Y-%m-%d")
            restock_date = (datetime.now() - timedelta(days=random.randint(5, 60))).strftime("%Y-%m-%d")

            pharmacy_inventory.append({
                "inventory_id": f"INV_{inv_id_counter:04d}",
                "medication_id": med["medication_id"],
                "hospital_id": h["hospital_id"],
                "stock_quantity": stock,
                "batch_number": f"BATCH_{fake.bothify(text='??###').upper()}",
                "expiry_date": exp_date,
                "reorder_level": reorder,
                "last_restocked_date": restock_date,
            })
            inv_id_counter += 1
    df_inventory = pd.DataFrame(pharmacy_inventory)

    # 7. Admissions (Generate correlated readmissions for ML)
    num_admissions = counts.get("admissions", 5000)
    admissions = []
    adm_counter = 1

    # Map patients to their admission history
    patient_ids = [p["patient_id"] for p in patients]
    # Some patients will have multiple admissions to simulate 30-day readmissions
    frequent_flyers = set(random.sample(patient_ids, int(len(patient_ids) * 0.25)))

    while adm_counter <= num_admissions:
        p_id = random.choice(patient_ids)
        p_info = next(p for p in patients if p["patient_id"] == p_id)
        doctor = random.choice(doctors)
        hosp_id = doctor["hospital_id"]
        dept_id = doctor["department_id"]

        adm_date_dt = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 500))
        # Length of stay between 1 and 21 days, skewed towards 3-7 days
        los_days = int(np.random.gamma(shape=3.0, scale=1.5)) + 1
        los_days = min(max(los_days, 1), 28)
        discharge_date_dt = adm_date_dt + timedelta(days=los_days)

        adm_type = random.choices(ADMISSION_TYPES, weights=[0.45, 0.35, 0.20])[0]
        disposition = random.choices(DISCHARGE_DISPOSITIONS, weights=[0.75, 0.12, 0.08, 0.03, 0.02])[0]

        adm_id = f"ADM_{adm_counter:05d}"
        admissions.append({
            "admission_id": adm_id,
            "patient_id": p_id,
            "hospital_id": hosp_id,
            "doctor_id": doctor["doctor_id"],
            "department_id": dept_id,
            "admission_date": adm_date_dt.strftime("%Y-%m-%d"),
            "discharge_date": discharge_date_dt.strftime("%Y-%m-%d"),
            "length_of_stay": los_days,
            "admission_type": adm_type,
            "room_number": f"{random.randint(100, 599)}",
            "discharge_disposition": disposition,
        })
        adm_counter += 1

        # If patient is a frequent flyer and not deceased, generate a secondary readmission within 30 days
        if p_id in frequent_flyers and disposition != "Deceased" and adm_counter <= num_admissions and random.random() < 0.65:
            gap_days = random.randint(3, 29)  # 30-day readmission!
            readm_date_dt = discharge_date_dt + timedelta(days=gap_days)
            readm_los = min(max(int(np.random.gamma(shape=3.5, scale=1.8)) + 1, 1), 30)
            readm_discharge = readm_date_dt + timedelta(days=readm_los)

            admissions.append({
                "admission_id": f"ADM_{adm_counter:05d}",
                "patient_id": p_id,
                "hospital_id": hosp_id,
                "doctor_id": doctor["doctor_id"],
                "department_id": dept_id,
                "admission_date": readm_date_dt.strftime("%Y-%m-%d"),
                "discharge_date": readm_discharge.strftime("%Y-%m-%d"),
                "length_of_stay": readm_los,
                "admission_type": "Emergency",
                "room_number": f"{random.randint(100, 599)}",
                "discharge_disposition": "Home",
            })
            adm_counter += 1

    df_admissions = pd.DataFrame(admissions)

    # 8. Diagnoses (Linked to admissions and patients)
    diagnoses = []
    dx_counter = 1
    for adm in admissions:
        # Every admission has at least 1 primary diagnosis
        primary_dx = random.choice(ICD10_CATALOG)
        diagnoses.append({
            "diagnosis_id": f"DX_{dx_counter:06d}",
            "admission_id": adm["admission_id"],
            "patient_id": adm["patient_id"],
            "icd10_code": primary_dx["code"],
            "diagnosis_description": primary_dx["description"],
            "diagnosis_type": "Primary",
            "diagnosis_date": adm["admission_date"],
            "severity": primary_dx["severity"],
        })
        dx_counter += 1

        # 60% probability of secondary comorbidities (Hypertension, Diabetes, COPD)
        if random.random() < 0.60:
            secondary_dx = random.choice([d for d in ICD10_CATALOG if d["code"] != primary_dx["code"]])
            diagnoses.append({
                "diagnosis_id": f"DX_{dx_counter:06d}",
                "admission_id": adm["admission_id"],
                "patient_id": adm["patient_id"],
                "icd10_code": secondary_dx["code"],
                "diagnosis_description": secondary_dx["description"],
                "diagnosis_type": "Secondary",
                "diagnosis_date": adm["admission_date"],
                "severity": secondary_dx["severity"],
            })
            dx_counter += 1

    df_diagnoses = pd.DataFrame(diagnoses)

    # 9. Laboratory Results (15,000 records)
    num_labs = counts.get("laboratory_results", 15000)
    lab_results = []
    for i in range(num_labs):
        adm = random.choice(admissions)
        test_meta = random.choice(LAB_CATALOG)
        val_type = random.choices(["Normal", "High", "Low", "Critical"], weights=[0.68, 0.18, 0.09, 0.05])[0]

        # Generate realistic value based on classification
        low_b = test_meta["low"]
        high_b = test_meta["high"]
        if val_type == "Normal":
            val = round(random.uniform(low_b, high_b), 2)
            flag = "Normal"
        elif val_type == "High":
            val = round(random.uniform(high_b + 0.1, test_meta["crit_high"] * 0.9), 2)
            flag = "High"
        elif val_type == "Low":
            val = round(random.uniform(test_meta["crit_low"] * 1.1, low_b - 0.1), 2)
            flag = "Low"
        else:
            val = round(random.uniform(test_meta["crit_high"], test_meta["crit_high"] * 1.4), 2)
            flag = "Critical"

        lab_date = adm["admission_date"]

        lab_results.append({
            "lab_id": f"LAB_{i+1:06d}",
            "patient_id": adm["patient_id"],
            "admission_id": adm["admission_id"],
            "test_name": test_meta["name"],
            "test_category": test_meta["category"],
            "test_value": float(val),
            "reference_range": f"{low_b} - {high_b}",
            "unit": test_meta["unit"],
            "abnormal_flag": flag,
            "test_date": lab_date,
        })
    df_labs = pd.DataFrame(lab_results)

    # 10. Prescriptions (12,000 records)
    num_rx = counts.get("prescriptions", 12000)
    prescriptions = []
    frequencies = ["Once daily", "Twice daily", "Every 8 hours", "At bedtime", "As needed"]
    for i in range(num_rx):
        adm = random.choice(admissions)
        med = random.choice(medications)
        duration = random.choice([7, 10, 14, 30, 90])
        prescriptions.append({
            "prescription_id": f"RX_{i+1:06d}",
            "patient_id": adm["patient_id"],
            "doctor_id": adm["doctor_id"],
            "admission_id": adm["admission_id"],
            "medication_id": med["medication_id"],
            "dosage": f"{random.choice([10, 20, 40, 500, 1000])} mg",
            "frequency": random.choice(frequencies),
            "duration_days": duration,
            "quantity": random.choice([30, 60, 90, 10]),
            "prescription_date": adm["admission_date"],
            "status": random.choice(["Active", "Completed", "Discontinued"]),
        })
    df_prescriptions = pd.DataFrame(prescriptions)

    # 11. Appointments (8,000 records)
    num_apts = counts.get("appointments", 8000)
    appointments = []
    reasons = [
        "Routine Follow-up", "Annual Health Checkup", "Chronic Disease Review",
        "Post-Operative Follow-up", "Medication Adjustment", "Chest Discomfort",
        "Respiratory Symptoms", "Diabetes Monitoring"
    ]
    for i in range(num_apts):
        p_id = random.choice(patient_ids)
        doc = random.choice(doctors)
        apt_dt = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 480))
        status = random.choices(APPOINTMENT_STATUSES, weights=[0.72, 0.14, 0.08, 0.06])[0]
        # Wait time between 5 and 75 minutes, higher for completed/walk-ins
        wait_time = random.randint(8, 65) if status == "Completed" else 0

        appointments.append({
            "appointment_id": f"APT_{i+1:06d}",
            "patient_id": p_id,
            "doctor_id": doc["doctor_id"],
            "department_id": doc["department_id"],
            "appointment_date": apt_dt.strftime("%Y-%m-%d"),
            "appointment_time": f"{random.randint(8, 17):02d}:{random.choice(['00', '15', '30', '45'])}",
            "status": status,
            "reason_for_visit": random.choice(reasons),
            "waiting_time_minutes": wait_time,
        })
    df_appointments = pd.DataFrame(appointments)

    # 12. Billing (One per admission)
    billing = []
    payment_methods = ["Commercial Insurance", "Medicare", "Credit Card", "Direct Debit", "Cash"]
    for idx, adm in enumerate(admissions):
        los = adm["length_of_stay"]
        base_charge = los * random.uniform(1100, 2400)
        lab_charge = random.uniform(300, 1800)
        pharmacy_charge = random.uniform(150, 1200)
        total_bill = round(base_charge + lab_charge + pharmacy_charge, 2)
        ins_coverage_ratio = random.uniform(0.65, 0.95)
        ins_paid = round(total_bill * ins_coverage_ratio, 2)
        patient_due = round(total_bill - ins_paid, 2)
        status = random.choices(PAYMENT_STATUSES, weights=[0.82, 0.14, 0.04])[0]

        billing.append({
            "bill_id": f"BILL_{idx+1:06d}",
            "patient_id": adm["patient_id"],
            "admission_id": adm["admission_id"],
            "bill_date": adm["discharge_date"],
            "total_amount": total_bill,
            "insurance_covered": ins_paid,
            "patient_payable": patient_due,
            "payment_status": status,
            "payment_method": random.choice(payment_methods),
        })
    df_billing = pd.DataFrame(billing)

    # Apply realistic imperfection injection for Bronze/Silver testing!
    print("Injecting realistic data quality issues (duplicates, nulls, inconsistent casing, corrupt dates)...")

    # Patients: dirty gender casing, null contacts, duplicates
    df_patients_raw = inject_realistic_imperfections(
        df_patients,
        date_columns=["dob", "created_at"],
        categorical_columns=["gender", "blood_group"],
        nullable_columns=["contact", "emergency_contact", "address"],
        numeric_columns=["age"],
        duplicate_ratio=0.015,
        seed=seed,
    )

    # Admissions: dirty admission_type, room_number nulls, duplicates
    df_admissions_raw = inject_realistic_imperfections(
        df_admissions,
        date_columns=["admission_date", "discharge_date"],
        categorical_columns=["admission_type", "discharge_disposition"],
        nullable_columns=["room_number"],
        numeric_columns=["length_of_stay"],
        duplicate_ratio=0.012,
        seed=seed + 1,
    )

    # Diagnoses: casing in severity
    df_diagnoses_raw = inject_realistic_imperfections(
        df_diagnoses,
        date_columns=["diagnosis_date"],
        categorical_columns=["severity", "diagnosis_type"],
        nullable_columns=[],
        duplicate_ratio=0.01,
        seed=seed + 2,
    )

    # Lab Results: save as JSON to demonstrate multi-format ingestion
    df_labs_raw = inject_realistic_imperfections(
        df_labs,
        date_columns=["test_date"],
        categorical_columns=["abnormal_flag"],
        nullable_columns=["unit"],
        numeric_columns=["test_value"],
        duplicate_ratio=0.012,
        seed=seed + 3,
    )

    # Medications: save as JSON to demonstrate multi-format ingestion
    df_medications_raw = inject_realistic_imperfections(
        df_medications,
        categorical_columns=["category"],
        nullable_columns=["dosage_form"],
        duplicate_ratio=0.0,
        seed=seed + 4,
    )

    # Prescriptions: CSV
    df_prescriptions_raw = inject_realistic_imperfections(
        df_prescriptions,
        date_columns=["prescription_date"],
        categorical_columns=["status"],
        nullable_columns=["frequency"],
        duplicate_ratio=0.01,
        seed=seed + 5,
    )

    # Appointments: CSV
    df_appointments_raw = inject_realistic_imperfections(
        df_appointments,
        date_columns=["appointment_date"],
        categorical_columns=["status"],
        nullable_columns=["reason_for_visit"],
        duplicate_ratio=0.01,
        seed=seed + 6,
    )

    # Billing: CSV
    df_billing_raw = inject_realistic_imperfections(
        df_billing,
        date_columns=["bill_date"],
        categorical_columns=["payment_status", "payment_method"],
        nullable_columns=["payment_method"],
        duplicate_ratio=0.01,
        seed=seed + 7,
    )

    # Inventory: CSV
    df_inventory_raw = inject_realistic_imperfections(
        df_inventory,
        date_columns=["expiry_date", "last_restocked_date"],
        categorical_columns=[],
        nullable_columns=["batch_number"],
        duplicate_ratio=0.01,
        seed=seed + 8,
    )

    # Save to disk in RAW directory
    print(f"Persisting raw files to {output_dir}...")
    df_hospitals.to_csv(output_dir / "hospitals.csv", index=False)
    df_departments.to_csv(output_dir / "departments.csv", index=False)
    df_doctors.to_csv(output_dir / "doctors.csv", index=False)
    df_patients_raw.to_csv(output_dir / "patients.csv", index=False)
    df_admissions_raw.to_csv(output_dir / "admissions.csv", index=False)
    df_diagnoses_raw.to_csv(output_dir / "diagnoses.csv", index=False)
    df_prescriptions_raw.to_csv(output_dir / "prescriptions.csv", index=False)
    df_appointments_raw.to_csv(output_dir / "appointments.csv", index=False)
    df_billing_raw.to_csv(output_dir / "billing.csv", index=False)
    df_inventory_raw.to_csv(output_dir / "pharmacy_inventory.csv", index=False)

    # JSON formatted datasets to satisfy multi-format requirement
    df_labs_raw.to_json(output_dir / "laboratory_results.json", orient="records", indent=2)
    df_medications_raw.to_json(output_dir / "medications.json", orient="records", indent=2)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "seed": seed,
        "counts": {
            "hospitals": len(df_hospitals),
            "departments": len(df_departments),
            "doctors": len(df_doctors),
            "patients": len(df_patients_raw),
            "admissions": len(df_admissions_raw),
            "diagnoses": len(df_diagnoses_raw),
            "laboratory_results": len(df_labs_raw),
            "medications": len(df_medications_raw),
            "prescriptions": len(df_prescriptions_raw),
            "appointments": len(df_appointments_raw),
            "billing": len(df_billing_raw),
            "pharmacy_inventory": len(df_inventory_raw),
        },
        "quality_defects_injected": [
            "Intentional duplicate rows (~1-1.5%)",
            "Inconsistent date representations (MM/DD/YYYY, DD-MM-YYYY, ISO)",
            "Irregular casing (UPPER, lower, leading/trailing whitespace)",
            "Missing values in optional fields (contact, room_number, units)",
            "Invalid categorical token anomalies ('UNKNOWN_VAL')",
            "Occasional invalid numerical values for outlier detection",
        ],
    }

    with open(METADATA_DIR / "generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Synthetic healthcare dataset generation complete.")
    return summary


if __name__ == "__main__":
    generate_healthcare_ecosystem()
