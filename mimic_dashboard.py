import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stroke AI Decision Support",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "mimic_iv_csv"

HOSP_DIR = DATA_DIR / "hosp"
ICU_DIR = DATA_DIR / "icu"

SYNTHETIC_FILE = (
    BASE_DIR
    / "synthetic_data"
    / "synthetic_asean_stroke_v2.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "stroke_models"
)


# ============================================================
# LOAD CSV
# ============================================================

@st.cache_data
def load_csv(folder, filename):

    path = folder / filename

    if not path.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(
            path,
            low_memory=False
        )

    except Exception as e:

        st.warning(
            f"Could not load {filename}: {e}"
        )

        return pd.DataFrame()


# ============================================================
# LOAD MIMIC DATA
# ============================================================

patients = load_csv(
    HOSP_DIR,
    "patients.csv"
)

admissions = load_csv(
    HOSP_DIR,
    "admissions.csv"
)

diagnoses = load_csv(
    HOSP_DIR,
    "diagnoses_icd.csv"
)

diagnosis_dictionary = load_csv(
    HOSP_DIR,
    "d_icd_diagnoses.csv"
)

labevents = load_csv(
    HOSP_DIR,
    "labevents.csv"
)

lab_dictionary = load_csv(
    HOSP_DIR,
    "d_labitems.csv"
)

procedures = load_csv(
    HOSP_DIR,
    "procedures_icd.csv"
)

procedure_dictionary = load_csv(
    HOSP_DIR,
    "d_icd_procedures.csv"
)

prescriptions = load_csv(
    HOSP_DIR,
    "prescriptions.csv"
)

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
# LOAD SYNTHETIC DATA
# ============================================================

@st.cache_data
def load_synthetic_data():

    if not SYNTHETIC_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(
        SYNTHETIC_FILE
    )


synthetic_df = load_synthetic_data()


# ============================================================
# LOAD AI MODELS
# ============================================================

@st.cache_resource
def load_model(filename):

    path = MODEL_DIR / filename

    if not path.exists():
        return None

    return joblib.load(
        path
    )


M1 = load_model(
    "M1_HT_Risk.joblib"
)

M2 = load_model(
    "M2_Early_Discharge_mRS.joblib"
)

M3 = load_model(
    "M3_Updated_Discharge_mRS.joblib"
)

M4 = load_model(
    "M4_90Day_mRS.joblib"
)

M5 = load_model(
    "M5_90Day_Mortality.joblib"
)


# ============================================================
# MODEL PREDICTION FUNCTIONS
# ============================================================

def prepare_patient_for_model(
    patient_row,
    model_object
):

    features = model_object[
        "features"
    ]

    data = {}

    for feature in features:

        if feature in patient_row.index:

            data[feature] = [
                patient_row[feature]
            ]

        else:

            data[feature] = [
                np.nan
            ]

    return pd.DataFrame(
        data
    )


def predict_binary(
    patient_row,
    model_object
):

    if model_object is None:
        return None

    X = prepare_patient_for_model(
        patient_row,
        model_object
    )

    pipeline = model_object[
        "pipeline"
    ]

    probability = (
        pipeline.predict_proba(
            X
        )[0, 1]
    )

    return float(
        probability
    )


def predict_mrs(
    patient_row,
    model_object
):

    if model_object is None:
        return None

    X = prepare_patient_for_model(
        patient_row,
        model_object
    )

    pipeline = model_object[
        "pipeline"
    ]

    prediction = pipeline.predict(
        X
    )[0]

    return int(
        prediction
    )


# ============================================================
# RISK LABEL
# ============================================================

def risk_label(probability):

    if probability is None:
        return "Unavailable"

    if probability < 0.10:
        return "Lower model-estimated risk"

    elif probability < 0.20:
        return "Intermediate model-estimated risk"

    else:
        return "Higher model-estimated risk"


# ============================================================
# YES / NO DISPLAY
# ============================================================

def yes_no(value):

    try:

        return (
            "Yes"
            if int(value) == 1
            else "No"
        )

    except:

        return str(
            value
        )


# ============================================================
# BUILD MIMIC STROKE COHORT
# ============================================================

@st.cache_data
def build_stroke_cohort(
    diagnoses,
    diagnosis_dictionary,
    admissions,
    patients
):

    if diagnoses.empty:
        return pd.DataFrame()

    dx = diagnoses.copy()

    if not diagnosis_dictionary.empty:

        dx = dx.merge(
            diagnosis_dictionary,
            on=[
                "icd_code",
                "icd_version"
            ],
            how="left"
        )


    dx["icd_code_clean"] = (
        dx["icd_code"]
        .astype(str)
        .str.upper()
        .str.replace(
            ".",
            "",
            regex=False
        )
        .str.strip()
    )


    dx["icd_version"] = pd.to_numeric(
        dx["icd_version"],
        errors="coerce"
    )


    def classify_stroke(row):

        code = row[
            "icd_code_clean"
        ]

        version = row[
            "icd_version"
        ]


        # ICD-10

        if version == 10:

            if code.startswith(
                (
                    "I60",
                    "I61",
                    "I62"
                )
            ):

                return (
                    "Hemorrhagic stroke"
                )


            if code.startswith(
                "I63"
            ):

                return (
                    "Ischemic stroke"
                )


            if code.startswith(
                "I64"
            ):

                return (
                    "Unspecified stroke"
                )


        # ICD-9

        elif version == 9:

            if code.startswith(
                (
                    "430",
                    "431",
                    "432"
                )
            ):

                return (
                    "Hemorrhagic stroke"
                )


            if code.startswith(
                "434"
            ):

                return (
                    "Ischemic stroke"
                )


            if code.startswith(
                "436"
            ):

                return (
                    "Unspecified stroke"
                )


        return None


    dx[
        "stroke_type"
    ] = dx.apply(
        classify_stroke,
        axis=1
    )


    stroke = dx[
        dx[
            "stroke_type"
        ].notna()
    ].copy()


    if not admissions.empty:

        admission_columns = [
            c
            for c in [
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "admission_type",
                "admission_location",
                "discharge_location",
                "race",
                "hospital_expire_flag"
            ]
            if c in admissions.columns
        ]


        stroke = stroke.merge(
            admissions[
                admission_columns
            ],
            on=[
                "subject_id",
                "hadm_id"
            ],
            how="left"
        )


    if not patients.empty:

        patient_columns = [
            c
            for c in [
                "subject_id",
                "gender",
                "anchor_age",
                "anchor_year",
                "dod"
            ]
            if c in patients.columns
        ]


        stroke = stroke.merge(
            patients[
                patient_columns
            ],
            on="subject_id",
            how="left"
        )


    return stroke


stroke_cohort = build_stroke_cohort(
    diagnoses,
    diagnosis_dictionary,
    admissions,
    patients
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧠 Stroke AI"
)

st.sidebar.caption(
    "Research Prototype"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "MIMIC Patient Explorer",
        "MIMIC Stroke Cohort",
        "Stroke AI Simulator"
    ]
)


# ============================================================
# PAGE 1
#
# MIMIC PATIENT EXPLORER
# ============================================================

if page == "MIMIC Patient Explorer":

    st.title(
        "🏥 MIMIC-IV Patient Explorer"
    )

    st.caption(
        "Explore patient-level information "
        "from the locally stored MIMIC-IV dataset."
    )


    if patients.empty:

        st.error(
            "patients.csv was not found."
        )

        st.write(
            f"Expected folder: {HOSP_DIR}"
        )

        st.stop()


    patient_ids = sorted(
        patients[
            "subject_id"
        ]
        .dropna()
        .unique()
    )


    selected_patient = st.selectbox(
        "Select patient",
        patient_ids
    )


    patient = patients[
        patients[
            "subject_id"
        ] == selected_patient
    ].copy()


    patient_admissions = (
        admissions[
            admissions[
                "subject_id"
            ] == selected_patient
        ].copy()
        if not admissions.empty
        else pd.DataFrame()
    )


    patient_diagnoses = (
        diagnoses[
            diagnoses[
                "subject_id"
            ] == selected_patient
        ].copy()
        if not diagnoses.empty
        else pd.DataFrame()
    )


    patient_labs = (
        labevents[
            labevents[
                "subject_id"
            ] == selected_patient
        ].copy()
        if not labevents.empty
        else pd.DataFrame()
    )


    patient_prescriptions = (
        prescriptions[
            prescriptions[
                "subject_id"
            ] == selected_patient
        ].copy()
        if not prescriptions.empty
        else pd.DataFrame()
    )


    patient_procedures = (
        procedures[
            procedures[
                "subject_id"
            ] == selected_patient
        ].copy()
        if not procedures.empty
        else pd.DataFrame()
    )


    patient_icu = (
        icustays[
            icustays[
                "subject_id"
            ] == selected_patient
        ].copy()
        if not icustays.empty
        else pd.DataFrame()
    )


    row = patient.iloc[0]


    st.divider()


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "Patient",
        selected_patient
    )


    c2.metric(
        "Gender",
        row.get(
            "gender",
            "N/A"
        )
    )


    c3.metric(
        "Anchor age",
        row.get(
            "anchor_age",
            "N/A"
        )
    )


    c4.metric(
        "Admissions",
        len(
            patient_admissions
        )
    )


    tabs = st.tabs(
        [
            "Demographics",
            "Admissions",
            "Diagnoses",
            "Laboratory",
            "Medications",
            "Procedures",
            "ICU"
        ]
    )


    # --------------------------------------------------------
    # DEMOGRAPHICS
    # --------------------------------------------------------

    with tabs[0]:

        st.dataframe(
            patient,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # ADMISSIONS
    # --------------------------------------------------------

    with tabs[1]:

        if patient_admissions.empty:

            st.info(
                "No admissions found."
            )

        else:

            st.dataframe(
                patient_admissions,
                use_container_width=True,
                hide_index=True
            )


    # --------------------------------------------------------
    # DIAGNOSES
    # --------------------------------------------------------

    with tabs[2]:

        if patient_diagnoses.empty:

            st.info(
                "No diagnoses found."
            )

        else:

            dx = (
                patient_diagnoses.copy()
            )


            if not diagnosis_dictionary.empty:

                dx = dx.merge(
                    diagnosis_dictionary,
                    on=[
                        "icd_code",
                        "icd_version"
                    ],
                    how="left"
                )


            st.dataframe(
                dx,
                use_container_width=True,
                hide_index=True
            )


    # --------------------------------------------------------
    # LABS
    # --------------------------------------------------------

    with tabs[3]:

        if patient_labs.empty:

            st.info(
                "No laboratory results."
            )

        else:

            labs = (
                patient_labs.copy()
            )


            if not lab_dictionary.empty:

                lab_cols = [
                    c
                    for c in [
                        "itemid",
                        "label",
                        "fluid",
                        "category"
                    ]
                    if c in lab_dictionary.columns
                ]


                labs = labs.merge(
                    lab_dictionary[
                        lab_cols
                    ],
                    on="itemid",
                    how="left"
                )


            st.dataframe(
                labs,
                use_container_width=True,
                hide_index=True
            )


    # --------------------------------------------------------
    # MEDICATIONS
    # --------------------------------------------------------

    with tabs[4]:

        if patient_prescriptions.empty:

            st.info(
                "No medication records."
            )

        else:

            st.dataframe(
                patient_prescriptions,
                use_container_width=True,
                hide_index=True
            )


    # --------------------------------------------------------
    # PROCEDURES
    # --------------------------------------------------------

    with tabs[5]:

        if patient_procedures.empty:

            st.info(
                "No procedure records."
            )

        else:

            proc = (
                patient_procedures.copy()
            )


            if not procedure_dictionary.empty:

                proc = proc.merge(
                    procedure_dictionary,
                    on=[
                        "icd_code",
                        "icd_version"
                    ],
                    how="left"
                )


            st.dataframe(
                proc,
                use_container_width=True,
                hide_index=True
            )


    # --------------------------------------------------------
    # ICU
    # --------------------------------------------------------

    with tabs[6]:

        if patient_icu.empty:

            st.info(
                "No ICU stay."
            )

        else:

            st.dataframe(
                patient_icu,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# PAGE 2
#
# MIMIC STROKE COHORT
# ============================================================

elif page == "MIMIC Stroke Cohort":

    st.title(
        "🧠 MIMIC-IV Stroke Cohort"
    )

    st.caption(
        "Stroke diagnoses identified from "
        "MIMIC-IV ICD diagnosis records."
    )


    if stroke_cohort.empty:

        st.warning(
            "No stroke records were identified."
        )

        st.stop()


    selected_type = st.selectbox(
        "Stroke type",
        [
            "All",
            "Ischemic stroke",
            "Hemorrhagic stroke",
            "Unspecified stroke"
        ]
    )


    if selected_type == "All":

        filtered = (
            stroke_cohort.copy()
        )

    else:

        filtered = (
            stroke_cohort[
                stroke_cohort[
                    "stroke_type"
                ] == selected_type
            ].copy()
        )


    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(
        "Patients",
        filtered[
            "subject_id"
        ].nunique()
    )


    c2.metric(
        "Admissions",
        filtered[
            "hadm_id"
        ].nunique()
    )


    c3.metric(
        "Diagnosis records",
        len(
            filtered
        )
    )


    st.divider()


    display_columns = [
        c
        for c in [
            "subject_id",
            "hadm_id",
            "gender",
            "anchor_age",
            "stroke_type",
            "icd_code",
            "icd_version",
            "long_title",
            "admittime",
            "dischtime",
            "hospital_expire_flag"
        ]
        if c in filtered.columns
    ]


    st.dataframe(
        filtered[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    st.subheader(
        "Inspect Stroke Patient"
    )


    stroke_ids = sorted(
        filtered[
            "subject_id"
        ]
        .dropna()
        .unique()
    )


    selected_stroke = st.selectbox(
        "Select patient",
        stroke_ids
    )


    selected_records = (
        filtered[
            filtered[
                "subject_id"
            ] == selected_stroke
        ]
    )


    st.dataframe(
        selected_records[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "NIHSS is not assumed to be available as a "
        "structured field for every MIMIC-IV patient. "
        "The synthetic Stroke AI cohort contains explicit "
        "baseline NIHSS for development and simulation."
    )


# ============================================================
# PAGE 3
#
# STROKE AI SIMULATOR
# ============================================================

elif page == "Stroke AI Simulator":

    st.title(
        "🧠 Stroke AI Decision-Support Simulator"
    )

    st.caption(
        "Synthetic ASEAN-oriented ischemic stroke "
        "research cohort"
    )


    st.warning(
        "Research prototype only. "
        "Patients beginning with S are synthetic. "
        "Predictions are generated from models trained "
        "on synthetic data and are not clinically validated."
    )


    if synthetic_df.empty:

        st.error(
            "Synthetic dataset was not found."
        )

        st.write(
            f"Expected file: {SYNTHETIC_FILE}"
        )

        st.stop()


    if any(
        model is None
        for model in [
            M1,
            M2,
            M3,
            M4,
            M5
        ]
    ):

        st.error(
            "One or more trained model files "
            "could not be found."
        )

        st.write(
            f"Expected folder: {MODEL_DIR}"
        )

        st.stop()


    # ========================================================
    # PATIENT SELECTOR
    # ========================================================

    synthetic_ids = (
        synthetic_df[
            "stroke_id"
        ]
        .astype(str)
        .tolist()
    )


    selected_id = st.selectbox(
        "Select synthetic stroke patient",
        synthetic_ids
    )


    patient = (
        synthetic_df[
            synthetic_df[
                "stroke_id"
            ] == selected_id
        ]
        .iloc[0]
        .copy()
    )


    # ========================================================
    # PATIENT PROFILE
    # ========================================================

    st.divider()

    st.header(
        f"Patient {selected_id}"
    )


    st.caption(
        "S = Synthetic stroke patient"
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "Age",
        int(
            patient[
                "age"
            ]
        )
    )


    c2.metric(
        "Sex",
        patient[
            "sex"
        ]
    )


    c3.metric(
        "Baseline NIHSS",
        int(
            patient[
                "baseline_nihss"
            ]
        )
    )


    c4.metric(
        "NIHSS severity",
        patient[
            "nihss_severity"
        ]
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "ASPECTS",
        int(
            patient[
                "aspects"
            ]
        )
    )


    c2.metric(
        "Pre-stroke mRS",
        int(
            patient[
                "prestroke_mrs"
            ]
        )
    )


    c3.metric(
        "Glucose",
        f"{patient['admission_glucose_mmol_l']:.1f} mmol/L"
    )


    c4.metric(
        "SBP",
        f"{int(patient['systolic_bp'])} mmHg"
    )


    # ========================================================
    # CLINICAL PROFILE
    # ========================================================

    with st.expander(
        "View complete clinical profile"
    ):

        profile = pd.DataFrame(
            {
                "Variable": [

                    "Country",
                    "Age",
                    "Sex",

                    "Hypertension",
                    "Diabetes",
                    "Atrial fibrillation",
                    "Prior stroke",
                    "Dyslipidemia",
                    "Smoking",

                    "Pre-stroke mRS",
                    "Baseline NIHSS",
                    "NIHSS severity",

                    "Stroke territory",
                    "TOAST subtype",

                    "Large vessel occlusion",
                    "ASPECTS",

                    "Systolic BP",
                    "Diastolic BP",

                    "Admission glucose",
                    "HbA1c",
                    "Platelets",
                    "INR",
                    "Creatinine",

                    "Onset-to-door",

                    "IV thrombolysis",
                    "Mechanical thrombectomy",

                    "Successful recanalisation",

                    "HT",
                    "HT subtype",
                    "sICH",

                    "Discharge NIHSS",
                    "Observed discharge mRS",
                    "Observed 90-day mRS",
                    "Observed 90-day mortality"
                ],

                "Value": [

                    patient[
                        "country"
                    ],

                    patient[
                        "age"
                    ],

                    patient[
                        "sex"
                    ],

                    yes_no(
                        patient[
                            "hypertension"
                        ]
                    ),

                    yes_no(
                        patient[
                            "diabetes"
                        ]
                    ),

                    yes_no(
                        patient[
                            "atrial_fibrillation"
                        ]
                    ),

                    yes_no(
                        patient[
                            "prior_stroke"
                        ]
                    ),

                    yes_no(
                        patient[
                            "dyslipidemia"
                        ]
                    ),

                    yes_no(
                        patient[
                            "smoking"
                        ]
                    ),

                    patient[
                        "prestroke_mrs"
                    ],

                    patient[
                        "baseline_nihss"
                    ],

                    patient[
                        "nihss_severity"
                    ],

                    patient[
                        "stroke_territory"
                    ],

                    patient[
                        "toast_subtype"
                    ],

                    yes_no(
                        patient[
                            "large_vessel_occlusion"
                        ]
                    ),

                    patient[
                        "aspects"
                    ],

                    patient[
                        "systolic_bp"
                    ],

                    patient[
                        "diastolic_bp"
                    ],

                    patient[
                        "admission_glucose_mmol_l"
                    ],

                    patient[
                        "hba1c"
                    ],

                    patient[
                        "platelets_10e9_l"
                    ],

                    patient[
                        "inr"
                    ],

                    patient[
                        "creatinine_umol_l"
                    ],

                    patient[
                        "onset_to_door_min"
                    ],

                    yes_no(
                        patient[
                            "iv_thrombolysis"
                        ]
                    ),

                    yes_no(
                        patient[
                            "mechanical_thrombectomy"
                        ]
                    ),

                    yes_no(
                        patient[
                            "successful_recanalisation"
                        ]
                    ),

                    yes_no(
                        patient[
                            "hemorrhagic_transformation"
                        ]
                    ),

                    patient[
                        "ht_subtype"
                    ],

                    yes_no(
                        patient[
                            "symptomatic_ich"
                        ]
                    ),

                    patient[
                        "discharge_nihss"
                    ],

                    patient[
                        "discharge_mrs"
                    ],

                    patient[
                        "mrs_90d"
                    ],

                    yes_no(
                        patient[
                            "mortality_90d"
                        ]
                    )
                ]
            }
        )


        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # RUN ORIGINAL PREDICTIONS
    # ========================================================

    ht_probability = predict_binary(
        patient,
        M1
    )


    early_mrs = predict_mrs(
        patient,
        M2
    )


    updated_mrs = predict_mrs(
        patient,
        M3
    )


    mrs_90 = predict_mrs(
        patient,
        M4
    )


    mortality_probability = predict_binary(
        patient,
        M5
    )


    # ========================================================
    # AI PREDICTIONS
    # ========================================================

    st.divider()

    st.header(
        "AI Predictions"
    )


    st.caption(
        "Predictions shown below are generated "
        "by the saved machine-learning models."
    )


    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(
        "HT risk",
        f"{ht_probability * 100:.1f}%"
    )


    c1.caption(
        risk_label(
            ht_probability
        )
    )


    c2.metric(
        "Early discharge mRS",
        early_mrs
    )


    c2.caption(
        "Prediction using admission information"
    )


    c3.metric(
        "Updated discharge mRS",
        updated_mrs
    )


    c3.caption(
        "Updated using hospitalization information"
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(
        "Predicted 90-day mRS",
        mrs_90
    )


    c2.metric(
        "90-day mortality risk",
        f"{mortality_probability * 100:.1f}%"
    )


    c2.caption(
        risk_label(
            mortality_probability
        )
    )


    # ========================================================
    # OBSERVED SYNTHETIC OUTCOMES
    # ========================================================

    with st.expander(
        "Compare predictions with synthetic outcomes"
    ):

        comparison = pd.DataFrame(
            {
                "Outcome": [

                    "Hemorrhagic transformation",

                    "Discharge mRS",

                    "90-day mRS",

                    "90-day mortality"
                ],

                "Synthetic outcome": [

                    yes_no(
                        patient[
                            "hemorrhagic_transformation"
                        ]
                    ),

                    int(
                        patient[
                            "discharge_mrs"
                        ]
                    ),

                    int(
                        patient[
                            "mrs_90d"
                        ]
                    ),

                    yes_no(
                        patient[
                            "mortality_90d"
                        ]
                    )
                ],

                "Model prediction": [

                    (
                        f"{ht_probability * 100:.1f}% risk"
                    ),

                    (
                        f"{early_mrs} early / "
                        f"{updated_mrs} updated"
                    ),

                    mrs_90,

                    (
                        f"{mortality_probability * 100:.1f}% risk"
                    )
                ]
            }
        )


        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # DYNAMIC PREDICTION
    # ========================================================

    st.divider()

    st.header(
        "Dynamic Prediction"
    )


    col1, col2, col3 = st.columns(
        [2, 1, 2]
    )


    with col1:

        st.subheader(
            "Admission"
        )

        st.metric(
            "Predicted discharge mRS",
            early_mrs
        )


    with col2:

        st.markdown(
            "### →"
        )


    with col3:

        st.subheader(
            "Updated"
        )

        delta = (
            updated_mrs
            - early_mrs
        )


        st.metric(
            "Predicted discharge mRS",
            updated_mrs,
            delta=delta,
            delta_color="inverse"
        )


    new_information = []


    if int(
        patient[
            "hemorrhagic_transformation"
        ]
    ) == 1:

        new_information.append(
            "Hemorrhagic transformation detected"
        )


    if int(
        patient[
            "symptomatic_ich"
        ]
    ) == 1:

        new_information.append(
            "Symptomatic intracranial hemorrhage detected"
        )


    if int(
        patient[
            "mechanical_thrombectomy"
        ]
    ) == 1:

        if int(
            patient[
                "successful_recanalisation"
            ]
        ) == 1:

            new_information.append(
                "Successful recanalisation"
            )

        else:

            new_information.append(
                "Recanalisation not successful"
            )


    if len(
        new_information
    ) > 0:

        st.write(
            "**New hospitalization information:**"
        )

        for item in new_information:

            st.write(
                f"• {item}"
            )

    else:

        st.write(
            "No major simulated complication/update "
            "recorded for this patient."
        )


    # ========================================================
    # COUNTERFACTUAL SIMULATOR
    # ========================================================

    st.divider()

    st.header(
        "Counterfactual Scenario Simulator"
    )


    st.write(
        "Change selected modifiable or scenario variables "
        "and rerun the HT model. This demonstrates how the "
        "model-estimated risk changes under an alternative "
        "patient scenario."
    )


    st.info(
        "A counterfactual model scenario is not the same "
        "as evidence that changing a variable will cause "
        "the predicted clinical benefit."
    )


    scenario_patient = (
        patient.copy()
    )


    c1, c2 = st.columns(
        2
    )


    with c1:

        scenario_glucose = st.slider(
            "Admission glucose (mmol/L)",
            min_value=3.0,
            max_value=20.0,
            value=float(
                patient[
                    "admission_glucose_mmol_l"
                ]
            ),
            step=0.1
        )


        scenario_sbp = st.slider(
            "Systolic BP (mmHg)",
            min_value=80,
            max_value=240,
            value=int(
                patient[
                    "systolic_bp"
                ]
            ),
            step=1
        )


    with c2:

        scenario_onset = st.slider(
            "Onset-to-door time (minutes)",
            min_value=20,
            max_value=720,
            value=int(
                patient[
                    "onset_to_door_min"
                ]
            ),
            step=5
        )


        current_dtn = patient[
            "door_to_needle_min"
        ]


        if pd.isna(
            current_dtn
        ):

            default_dtn = 60

        else:

            default_dtn = int(
                current_dtn
            )


        scenario_dtn = st.slider(
            "Door-to-needle time (minutes)",
            min_value=20,
            max_value=180,
            value=max(
                20,
                min(
                    180,
                    default_dtn
                )
            ),
            step=5,
            disabled=(
                int(
                    patient[
                        "iv_thrombolysis"
                    ]
                ) == 0
            )
        )


    scenario_patient[
        "admission_glucose_mmol_l"
    ] = scenario_glucose


    scenario_patient[
        "systolic_bp"
    ] = scenario_sbp


    scenario_patient[
        "onset_to_door_min"
    ] = scenario_onset


    if int(
        patient[
            "iv_thrombolysis"
        ]
    ) == 1:

        scenario_patient[
            "door_to_needle_min"
        ] = scenario_dtn


    scenario_ht = predict_binary(
        scenario_patient,
        M1
    )


    risk_difference = (
        scenario_ht
        - ht_probability
    )


    st.subheader(
        "Scenario Result"
    )


    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(
        "Current HT risk",
        f"{ht_probability * 100:.1f}%"
    )


    c2.metric(
        "Scenario HT risk",
        f"{scenario_ht * 100:.1f}%"
    )


    c3.metric(
        "Risk difference",
        f"{risk_difference * 100:+.1f} pp",
        delta_color="inverse"
    )


    # ========================================================
    # NON-MODIFIABLE / BASELINE FACTORS
    # ========================================================

    st.subheader(
        "Baseline Risk Context"
    )


    context = []


    if patient[
        "baseline_nihss"
    ] >= 16:

        context.append(
            "Higher baseline neurological severity "
            "(NIHSS ≥16)"
        )


    elif patient[
        "baseline_nihss"
    ] >= 6:

        context.append(
            "Moderate baseline neurological deficit"
        )


    if patient[
        "aspects"
    ] <= 6:

        context.append(
            "Lower ASPECTS"
        )


    if int(
        patient[
            "atrial_fibrillation"
        ]
    ) == 1:

        context.append(
            "Atrial fibrillation present"
        )


    if int(
        patient[
            "diabetes"
        ]
    ) == 1:

        context.append(
            "Diabetes present"
        )


    if int(
        patient[
            "prior_stroke"
        ]
    ) == 1:

        context.append(
            "History of previous stroke"
        )


    if int(
        patient[
            "large_vessel_occlusion"
        ]
    ) == 1:

        context.append(
            "Large vessel occlusion present"
        )


    if len(
        context
    ) == 0:

        st.write(
            "No predefined major baseline flags "
            "were triggered."
        )

    else:

        for item in context:

            st.write(
                f"• {item}"
            )


    # ========================================================
    # DECISION-SUPPORT INTERPRETATION
    # ========================================================

    st.divider()

    st.header(
        "Decision-Support Interpretation"
    )


    if scenario_ht < ht_probability:

        direction_text = (
            f"The alternative scenario produced a "
            f"{abs(risk_difference) * 100:.1f} "
            f"percentage-point reduction in the "
            f"model-estimated HT risk."
        )


    elif scenario_ht > ht_probability:

        direction_text = (
            f"The alternative scenario produced a "
            f"{abs(risk_difference) * 100:.1f} "
            f"percentage-point increase in the "
            f"model-estimated HT risk."
        )


    else:

        direction_text = (
            "The alternative scenario did not materially "
            "change the model-estimated HT risk."
        )


    st.write(
        f"""
The model estimates the current hemorrhagic transformation
risk at **{ht_probability * 100:.1f}%**.

The admission model predicts a discharge mRS of
**{early_mrs}**. After hospitalization information is
included, the updated prediction is **{updated_mrs}**.

The model predicts a 90-day mRS of **{mrs_90}** and a
90-day mortality risk of **{mortality_probability * 100:.1f}%**.

{direction_text}
"""
    )


    st.warning(
        "The scenario result represents an association "
        "learned from synthetic data. It must not be "
        "interpreted as a causal treatment effect."
    )


    # ========================================================
    # FUTURE LLM / RAG SECTION
    # ========================================================

    st.subheader(
        "Evidence-Grounded Recommendation Layer"
    )


    st.write(
        """
The next development stage can connect this section to a
retrieval-augmented LLM. The LLM should receive:

1. the patient profile,
2. prediction results,
3. model explanations,
4. constrained counterfactual scenarios, and
5. retrieved stroke-management guidance.

The LLM should explain the evidence and possible clinical
considerations rather than independently prescribe treatment.
"""
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with st.expander(
        "Model information"
    ):

        model_information = []


        for name, model in [

            (
                "M1 HT Risk",
                M1
            ),

            (
                "M2 Early Discharge mRS",
                M2
            ),

            (
                "M3 Updated Discharge mRS",
                M3
            ),

            (
                "M4 90-Day mRS",
                M4
            ),

            (
                "M5 90-Day Mortality",
                M5
            )
        ]:

            model_information.append(
                {
                    "Model":
                        name,

                    "Algorithm":
                        model.get(
                            "algorithm",
                            "Unknown"
                        ),

                    "Target":
                        model.get(
                            "target",
                            "Unknown"
                        ),

                    "Features":
                        len(
                            model.get(
                                "features",
                                []
                            )
                        )
                }
            )


        st.dataframe(
            pd.DataFrame(
                model_information
            ),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Stroke AI Decision-Support Research Prototype | "
    "Synthetic models are not clinically validated."
)