import streamlit as st
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MIMIC-IV Patient Explorer",
    page_icon="🏥",
    layout="wide"
)

DATA_DIR = Path(
    r"C:\Users\fadhlina\Downloads\mimic_iv_csv"
)

HOSP_DIR = DATA_DIR / "hosp"
ICU_DIR = DATA_DIR / "icu"


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_csv(folder, filename):
    path = folder / filename

    if path.exists():
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception as e:
            st.warning(f"Could not read {filename}: {e}")
            return pd.DataFrame()

    return pd.DataFrame()


# Core patient information
patients = load_csv(HOSP_DIR, "patients.csv")
admissions = load_csv(HOSP_DIR, "admissions.csv")

# Diagnoses
diagnoses = load_csv(HOSP_DIR, "diagnoses_icd.csv")
diagnosis_dictionary = load_csv(
    HOSP_DIR,
    "d_icd_diagnoses.csv"
)

# Laboratory
labevents = load_csv(HOSP_DIR, "labevents.csv")
lab_dictionary = load_csv(
    HOSP_DIR,
    "d_labitems.csv"
)

# Procedures
procedures = load_csv(
    HOSP_DIR,
    "procedures_icd.csv"
)

procedure_dictionary = load_csv(
    HOSP_DIR,
    "d_icd_procedures.csv"
)

# Medications
prescriptions = load_csv(
    HOSP_DIR,
    "prescriptions.csv"
)

# ICU
icustays = load_csv(
    ICU_DIR,
    "icustays.csv"
)

chartevents = load_csv(
    ICU_DIR,
    "chartevents.csv"
)

item_dictionary = load_csv(
    ICU_DIR,
    "d_items.csv"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏥 MIMIC-IV Patient Explorer")

st.caption(
    "Patient-level exploration of the MIMIC-IV Clinical Database"
)

st.divider()


# ============================================================
# CHECK PATIENT TABLE
# ============================================================

if patients.empty:
    st.error(
        "patients.csv could not be found. "
        "Check DATA_DIR."
    )
    st.stop()


# ============================================================
# SIDEBAR PATIENT SELECTION
# ============================================================

st.sidebar.header("Patient Selection")

patient_ids = sorted(
    patients["subject_id"].dropna().unique()
)

selected_patient = st.sidebar.selectbox(
    "Select Subject ID",
    patient_ids
)

st.sidebar.write(
    f"Patients available: {len(patient_ids):,}"
)


# ============================================================
# FILTER SELECTED PATIENT
# ============================================================

patient = patients[
    patients["subject_id"] == selected_patient
].copy()

patient_admissions = admissions[
    admissions["subject_id"] == selected_patient
].copy() if not admissions.empty else pd.DataFrame()

patient_diagnoses = diagnoses[
    diagnoses["subject_id"] == selected_patient
].copy() if not diagnoses.empty else pd.DataFrame()

patient_labs = labevents[
    labevents["subject_id"] == selected_patient
].copy() if not labevents.empty else pd.DataFrame()

patient_procedures = procedures[
    procedures["subject_id"] == selected_patient
].copy() if not procedures.empty else pd.DataFrame()

patient_prescriptions = prescriptions[
    prescriptions["subject_id"] == selected_patient
].copy() if not prescriptions.empty else pd.DataFrame()

patient_icu = icustays[
    icustays["subject_id"] == selected_patient
].copy() if not icustays.empty else pd.DataFrame()

patient_chart = chartevents[
    chartevents["subject_id"] == selected_patient
].copy() if not chartevents.empty else pd.DataFrame()


# ============================================================
# PATIENT SUMMARY
# ============================================================

st.header(f"Patient {selected_patient}")

col1, col2, col3, col4 = st.columns(4)

row = patient.iloc[0]

with col1:
    st.metric(
        "Gender",
        row.get("gender", "N/A")
    )

with col2:
    st.metric(
        "Anchor Age",
        row.get("anchor_age", "N/A")
    )

with col3:
    st.metric(
        "Admissions",
        len(patient_admissions)
    )

with col4:
    st.metric(
        "ICU Stays",
        len(patient_icu)
    )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "👤 Demographics",
    "🏥 Admissions",
    "🧠 Diagnoses",
    "🧪 Laboratory",
    "💊 Medications",
    "🩺 Procedures",
    "❤️ ICU",
    "📈 ICU Measurements"
])


# ============================================================
# TAB 1 - DEMOGRAPHICS
# ============================================================

with tabs[0]:

    st.subheader("Patient Demographics")

    st.dataframe(
        patient,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2 - ADMISSIONS
# ============================================================

with tabs[1]:

    st.subheader("Hospital Admissions")

    if patient_admissions.empty:

        st.info("No admissions found.")

    else:

        preferred_columns = [
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admit_provider_id",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "hospital_expire_flag"
        ]

        available_columns = [
            c for c in preferred_columns
            if c in patient_admissions.columns
        ]

        st.dataframe(
            patient_admissions[available_columns],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 3 - DIAGNOSES
# ============================================================

with tabs[2]:

    st.subheader("Diagnoses")

    if patient_diagnoses.empty:

        st.info("No diagnoses found.")

    else:

        dx = patient_diagnoses.copy()

        # Add human-readable diagnosis title
        if not diagnosis_dictionary.empty:

            dx = dx.merge(
                diagnosis_dictionary,
                on=["icd_code", "icd_version"],
                how="left"
            )

        preferred_columns = [
            "hadm_id",
            "seq_num",
            "icd_code",
            "icd_version",
            "long_title"
        ]

        available_columns = [
            c for c in preferred_columns
            if c in dx.columns
        ]

        st.dataframe(
            dx[available_columns],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 4 - LABORATORY
# ============================================================

with tabs[3]:

    st.subheader("Laboratory Results")

    if patient_labs.empty:

        st.info("No laboratory results found.")

    else:

        labs = patient_labs.copy()

        if not lab_dictionary.empty:

            lab_columns = [
                c for c in [
                    "itemid",
                    "label",
                    "fluid",
                    "category"
                ]
                if c in lab_dictionary.columns
            ]

            labs = labs.merge(
                lab_dictionary[lab_columns],
                on="itemid",
                how="left"
            )

        preferred_columns = [
            "hadm_id",
            "charttime",
            "label",
            "value",
            "valuenum",
            "valueuom",
            "ref_range_lower",
            "ref_range_upper",
            "flag"
        ]

        available_columns = [
            c for c in preferred_columns
            if c in labs.columns
        ]

        st.write(
            f"Laboratory records: {len(labs):,}"
        )

        st.dataframe(
            labs[available_columns],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 5 - MEDICATIONS
# ============================================================

with tabs[4]:

    st.subheader("Prescriptions / Medications")

    if patient_prescriptions.empty:

        st.info("No prescriptions found.")

    else:

        preferred_columns = [
            "hadm_id",
            "starttime",
            "stoptime",
            "drug",
            "formulary_drug_cd",
            "dose_val_rx",
            "dose_unit_rx",
            "route"
        ]

        available_columns = [
            c for c in preferred_columns
            if c in patient_prescriptions.columns
        ]

        st.dataframe(
            patient_prescriptions[available_columns],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 6 - PROCEDURES
# ============================================================

with tabs[5]:

    st.subheader("Procedures")

    if patient_procedures.empty:

        st.info("No procedures found.")

    else:

        proc = patient_procedures.copy()

        if not procedure_dictionary.empty:

            proc = proc.merge(
                procedure_dictionary,
                on=["icd_code", "icd_version"],
                how="left"
            )

        preferred_columns = [
            "hadm_id",
            "seq_num",
            "chartdate",
            "icd_code",
            "icd_version",
            "long_title"
        ]

        available_columns = [
            c for c in preferred_columns
            if c in proc.columns
        ]

        st.dataframe(
            proc[available_columns],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 7 - ICU STAYS
# ============================================================

with tabs[6]:

    st.subheader("ICU Stays")

    if patient_icu.empty:

        st.info(
            "This patient has no ICU stay."
        )

    else:

        preferred_columns = [
            "hadm_id",
            "stay_id",
            "first_careunit",
            "last_careunit",
            "intime",
            "outtime",
            "los"
        ]

        available_columns = [
            c for c in preferred_columns
            if c in patient_icu.columns
        ]

        st.dataframe(
            patient_icu[available_columns],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 8 - ICU MEASUREMENTS
# ============================================================

with tabs[7]:

    st.subheader("ICU Measurements")

    if patient_chart.empty:

        st.info(
            "No ICU chart events found."
        )

    else:

        chart = patient_chart.copy()

        if not item_dictionary.empty:

            item_columns = [
                c for c in [
                    "itemid",
                    "label",
                    "category",
                    "unitname"
                ]
                if c in item_dictionary.columns
            ]

            chart = chart.merge(
                item_dictionary[item_columns],
                on="itemid",
                how="left"
            )

        preferred_columns = [
            "stay_id",
            "charttime",
            "label",
            "value",
            "valuenum",
            "valueuom"
        ]

        available_columns = [
            c for c in preferred_columns
            if c in chart.columns
        ]

        st.write(
            f"ICU measurements: {len(chart):,}"
        )

        st.dataframe(
            chart[available_columns],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MIMIC-IV Patient Explorer | Research use"
)