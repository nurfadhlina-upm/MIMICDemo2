import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AdaptStroke AI",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# PATHS
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

MODEL_DIR = BASE_DIR / "stroke_models"


# ============================================================
# ENVIRONMENT DETECTION
# ============================================================

MIMIC_AVAILABLE = (
    (HOSP_DIR / "patients.csv").exists()
    and
    (HOSP_DIR / "admissions.csv").exists()
    and
    (HOSP_DIR / "diagnoses_icd.csv").exists()
)


# ============================================================
# GENERAL LOADERS
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

    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_synthetic_data():

    if not SYNTHETIC_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(
        SYNTHETIC_FILE,
        low_memory=False
    )


@st.cache_resource
def load_model(filename):

    path = MODEL_DIR / filename

    if not path.exists():
        return None

    return joblib.load(
        path
    )


# ============================================================
# LOAD SYNTHETIC DATA + MODELS
# ============================================================

synthetic_df = load_synthetic_data()

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
# LOAD MIMIC ONLY WHEN AVAILABLE
# ============================================================

if MIMIC_AVAILABLE:

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

    prescriptions = load_csv(
        HOSP_DIR,
        "prescriptions.csv"
    )

    procedures = load_csv(
        HOSP_DIR,
        "procedures_icd.csv"
    )

    procedure_dictionary = load_csv(
        HOSP_DIR,
        "d_icd_procedures.csv"
    )

    icustays = load_csv(
        ICU_DIR,
        "icustays.csv"
    )

else:

    patients = pd.DataFrame()
    admissions = pd.DataFrame()
    diagnoses = pd.DataFrame()
    diagnosis_dictionary = pd.DataFrame()
    labevents = pd.DataFrame()
    lab_dictionary = pd.DataFrame()
    prescriptions = pd.DataFrame()
    procedures = pd.DataFrame()
    procedure_dictionary = pd.DataFrame()
    icustays = pd.DataFrame()


# ============================================================
# HELPERS
# ============================================================

def yes_no(value):

    try:
        return (
            "Yes"
            if int(value) == 1
            else "No"
        )

    except Exception:
        return str(value)


def safe_int(value, default=0):

    try:

        if pd.isna(value):
            return default

        return int(value)

    except Exception:
        return default


def safe_float(value, default=0.0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


# ============================================================
# MODEL INPUT
# ============================================================

def prepare_patient_for_model(
    patient_row,
    model_object
):

    features = model_object[
        "features"
    ]

    values = {}

    for feature in features:

        if feature in patient_row.index:

            values[feature] = [
                patient_row[feature]
            ]

        else:

            values[feature] = [
                np.nan
            ]

    return pd.DataFrame(
        values
    )


# ============================================================
# PREDICTION
# ============================================================

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
        pipeline
        .predict_proba(X)[0, 1]
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

    prediction = (
        model_object[
            "pipeline"
        ]
        .predict(X)[0]
    )

    return int(
        prediction
    )


# ============================================================
# RISK LABEL
#
# Descriptive only.
# These are prototype display bands,
# NOT validated clinical thresholds.
# ============================================================

def risk_label(probability):

    if probability is None:
        return "Unavailable"

    if probability < 0.10:
        return "Lower model-estimated risk"

    elif probability < 0.20:
        return "Intermediate model-estimated risk"

    return "Higher model-estimated risk"


# ============================================================
# APPROXIMATE LOCAL MODEL EXPLANATION
#
# Perturb one variable at a time and calculate change
# in prediction.
#
# This is model sensitivity / local perturbation,
# NOT causal importance.
# ============================================================

def local_binary_explanation(
    patient,
    model_object,
    reference_df,
    max_features=8
):

    if model_object is None:
        return pd.DataFrame()

    baseline = predict_binary(
        patient,
        model_object
    )

    results = []

    features = model_object.get(
        "features",
        []
    )

    for feature in features:

        if feature not in patient.index:
            continue

        if feature not in reference_df.columns:
            continue

        original = patient[
            feature
        ]

        modified = patient.copy()

        column = reference_df[
            feature
        ]


        # ----------------------------------------------------
        # NUMERIC VARIABLE
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            column
        ):

            reference_value = (
                column
                .dropna()
                .median()
            )

        else:

            mode = (
                column
                .dropna()
                .mode()
            )

            if len(mode) == 0:
                continue

            reference_value = (
                mode.iloc[0]
            )


        if pd.isna(
            reference_value
        ):
            continue


        modified[
            feature
        ] = reference_value


        try:

            alternative = predict_binary(
                modified,
                model_object
            )

        except Exception:
            continue


        effect = (
            baseline
            - alternative
        )


        results.append(
            {
                "Feature":
                    feature,

                "Patient value":
                    original,

                "Reference value":
                    reference_value,

                "Local contribution":
                    effect,

                "Absolute contribution":
                    abs(effect)
            }
        )


    if not results:
        return pd.DataFrame()


    result_df = pd.DataFrame(
        results
    )


    result_df = (
        result_df
        .sort_values(
            "Absolute contribution",
            ascending=False
        )
        .head(max_features)
    )


    return result_df


# ============================================================
# MIMIC STROKE COHORT
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


    dx[
        "icd_code_clean"
    ] = (
        dx[
            "icd_code"
        ]
        .astype(str)
        .str.upper()
        .str.replace(
            ".",
            "",
            regex=False
        )
        .str.strip()
    )


    dx[
        "icd_version"
    ] = pd.to_numeric(
        dx[
            "icd_version"
        ],
        errors="coerce"
    )


    def classify(row):

        code = row[
            "icd_code_clean"
        ]

        version = row[
            "icd_version"
        ]


        if version == 10:

            if code.startswith(
                (
                    "I60",
                    "I61",
                    "I62"
                )
            ):
                return "Hemorrhagic stroke"

            if code.startswith(
                "I63"
            ):
                return "Ischemic stroke"

            if code.startswith(
                "I64"
            ):
                return "Unspecified stroke"


        if version == 9:

            if code.startswith(
                (
                    "430",
                    "431",
                    "432"
                )
            ):
                return "Hemorrhagic stroke"

            if code.startswith(
                "434"
            ):
                return "Ischemic stroke"

            if code.startswith(
                "436"
            ):
                return "Unspecified stroke"


        return None


    dx[
        "stroke_type"
    ] = dx.apply(
        classify,
        axis=1
    )


    stroke = dx[
        dx[
            "stroke_type"
        ].notna()
    ].copy()


    if not admissions.empty:

        cols = [
            c
            for c in [
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "admission_type",
                "discharge_location",
                "race",
                "hospital_expire_flag"
            ]
            if c in admissions.columns
        ]


        stroke = stroke.merge(
            admissions[
                cols
            ],
            on=[
                "subject_id",
                "hadm_id"
            ],
            how="left"
        )


    if not patients.empty:

        cols = [
            c
            for c in [
                "subject_id",
                "gender",
                "anchor_age",
                "dod"
            ]
            if c in patients.columns
        ]


        stroke = stroke.merge(
            patients[
                cols
            ],
            on="subject_id",
            how="left"
        )


    return stroke


if MIMIC_AVAILABLE:

    stroke_cohort = build_stroke_cohort(
        diagnoses,
        diagnosis_dictionary,
        admissions,
        patients
    )

else:

    stroke_cohort = pd.DataFrame()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧠 AdaptStroke AI"
)

st.sidebar.caption(
    "Adaptive Stroke Decision-Support Research Prototype"
)


# ============================================================
# AUTOMATIC MENU
# ============================================================

if MIMIC_AVAILABLE:

    pages = [
        "Stroke AI Simulator",
        "MIMIC Patient Explorer",
        "MIMIC Stroke Cohort"
    ]

else:

    pages = [
        "Stroke AI Simulator"
    ]


page = st.sidebar.radio(
    "Navigation",
    pages
)


if not MIMIC_AVAILABLE:

    st.sidebar.success(
        "Public research mode"
    )

    st.sidebar.caption(
        "Restricted MIMIC-IV files are not "
        "loaded in this deployment."
    )

else:

    st.sidebar.success(
        "Local research mode"
    )


# ============================================================
# STROKE AI SIMULATOR
# ============================================================

if page == "Stroke AI Simulator":

    st.title(
        "🧠 AdaptStroke AI"
    )

    st.subheader(
        "Adaptive Stroke Decision-Support Simulator"
    )

    st.caption(
        "Prediction • Dynamic Updating • "
        "Explainability • Counterfactual Simulation"
    )


    st.warning(
        "Research prototype only. "
        "All S-prefixed patients are synthetic. "
        "The models are trained on synthetic data and "
        "have not been clinically validated."
    )


    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if synthetic_df.empty:

        st.error(
            "Synthetic stroke dataset is unavailable."
        )

        st.stop()


    models = [
        M1,
        M2,
        M3,
        M4,
        M5
    ]


    if any(
        model is None
        for model in models
    ):

        st.error(
            "One or more trained AI model files "
            "are unavailable."
        )

        st.stop()


    # ========================================================
    # PATIENT SELECTION
    # ========================================================

    st.header(
        "1. Patient Selection"
    )


    patient_ids = (
        synthetic_df[
            "stroke_id"
        ]
        .astype(str)
        .tolist()
    )


    selected_id = st.selectbox(
        "Synthetic patient",
        patient_ids
    )


    patient = (
        synthetic_df[
            synthetic_df[
                "stroke_id"
            ].astype(str)
            == selected_id
        ]
        .iloc[0]
        .copy()
    )


    # ========================================================
    # PATIENT PROFILE
    # ========================================================

    st.header(
        "2. Patient Profile"
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "Age",
        safe_int(
            patient["age"]
        )
    )

    c2.metric(
        "Sex",
        patient["sex"]
    )

    c3.metric(
        "Baseline NIHSS",
        safe_int(
            patient[
                "baseline_nihss"
            ]
        )
    )

    c4.metric(
        "ASPECTS",
        safe_int(
            patient[
                "aspects"
            ]
        )
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "Pre-stroke mRS",
        safe_int(
            patient[
                "prestroke_mrs"
            ]
        )
    )

    c2.metric(
        "Glucose",
        (
            f"{safe_float(patient['admission_glucose_mmol_l']):.1f} "
            f"mmol/L"
        )
    )

    c3.metric(
        "SBP",
        (
            f"{safe_int(patient['systolic_bp'])} "
            f"mmHg"
        )
    )

    c4.metric(
        "LVO",
        yes_no(
            patient[
                "large_vessel_occlusion"
            ]
        )
    )


    with st.expander(
        "Complete clinical profile"
    ):

        fields = [

            "country",
            "age",
            "sex",

            "hypertension",
            "diabetes",
            "atrial_fibrillation",
            "prior_stroke",
            "dyslipidemia",
            "smoking",

            "prestroke_mrs",

            "baseline_nihss",
            "nihss_severity",

            "stroke_territory",
            "toast_subtype",

            "large_vessel_occlusion",
            "aspects",

            "systolic_bp",
            "diastolic_bp",

            "admission_glucose_mmol_l",
            "hba1c",
            "platelets_10e9_l",
            "inr",
            "creatinine_umol_l",

            "onset_to_door_min",

            "iv_thrombolysis",
            "door_to_needle_min",

            "mechanical_thrombectomy",
            "tici_score",
            "successful_recanalisation",

            "hemorrhagic_transformation",
            "ht_subtype",
            "symptomatic_ich",

            "discharge_nihss",
            "discharge_mrs",

            "mrs_90d",
            "mortality_90d"
        ]


        fields = [
            field
            for field in fields
            if field in patient.index
        ]


        profile = pd.DataFrame(
            {
                "Variable":
                    fields,

                "Value":
                    [
                        patient[field]
                        for field in fields
                    ]
            }
        )


        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    ht_risk = predict_binary(
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

    mrs90 = predict_mrs(
        patient,
        M4
    )

    mortality = predict_binary(
        patient,
        M5
    )


    st.header(
        "3. AI Prediction"
    )


    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(
        "Hemorrhagic transformation",
        f"{ht_risk * 100:.1f}%"
    )

    c1.caption(
        risk_label(
            ht_risk
        )
    )


    c2.metric(
        "Early discharge mRS",
        early_mrs
    )

    c2.caption(
        "Admission-time prediction"
    )


    c3.metric(
        "Updated discharge mRS",
        updated_mrs
    )

    c3.caption(
        "Updated during hospitalization"
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(
        "Predicted 90-day mRS",
        mrs90
    )


    c2.metric(
        "90-day mortality",
        f"{mortality * 100:.1f}%"
    )

    c2.caption(
        risk_label(
            mortality
        )
    )


    st.caption(
        "Displayed risk bands are prototype "
        "visualisation categories and are not "
        "validated clinical thresholds."
    )


    # ========================================================
    # DYNAMIC / ADAPTIVE PREDICTION
    # ========================================================

    st.header(
        "4. Dynamic Prediction Update"
    )


    c1, c2, c3 = st.columns(
        [2, 1, 2]
    )


    with c1:

        st.markdown(
            "#### Admission"
        )

        st.metric(
            "Expected discharge mRS",
            early_mrs
        )


    with c2:

        st.markdown(
            "## →"
        )


    with c3:

        st.markdown(
            "#### Updated"
        )

        difference = (
            updated_mrs
            - early_mrs
        )


        st.metric(
            "Expected discharge mRS",
            updated_mrs,
            delta=difference,
            delta_color="inverse"
        )


    update_events = []


    if safe_int(
        patient[
            "hemorrhagic_transformation"
        ]
    ) == 1:

        update_events.append(
            "Hemorrhagic transformation became available."
        )


    if safe_int(
        patient[
            "symptomatic_ich"
        ]
    ) == 1:

        update_events.append(
            "Symptomatic intracranial hemorrhage became available."
        )


    if safe_int(
        patient[
            "mechanical_thrombectomy"
        ]
    ) == 1:

        if safe_int(
            patient[
                "successful_recanalisation"
            ]
        ) == 1:

            update_events.append(
                "Successful recanalisation was recorded."
            )

        else:

            update_events.append(
                "Successful recanalisation was not recorded."
            )


    if update_events:

        st.write(
            "**Information incorporated during updating:**"
        )

        for event in update_events:

            st.write(
                f"• {event}"
            )

    else:

        st.write(
            "No predefined major hospitalization event "
            "was recorded for this synthetic patient."
        )


    # ========================================================
    # MODEL EXPLAINABILITY
    # ========================================================

    st.header(
        "5. Model Explainability"
    )


    st.write(
        "The following analysis estimates which patient "
        "variables most influence this patient's HT prediction "
        "by replacing one variable at a time with the cohort "
        "reference value."
    )


    explanation = local_binary_explanation(
        patient,
        M1,
        synthetic_df,
        max_features=8
    )


    if not explanation.empty:

        display_explanation = (
            explanation.copy()
        )


        display_explanation[
            "Effect on HT risk (percentage points)"
        ] = (
            display_explanation[
                "Local contribution"
            ]
            * 100
        ).round(2)


        display_explanation[
            "Direction"
        ] = np.where(
            display_explanation[
                "Local contribution"
            ] > 0,
            "Associated with higher model estimate",
            "Associated with lower model estimate"
        )


        st.dataframe(
            display_explanation[
                [
                    "Feature",
                    "Patient value",
                    "Reference value",
                    "Effect on HT risk (percentage points)",
                    "Direction"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


        chart_data = (
            display_explanation[
                [
                    "Feature",
                    "Effect on HT risk (percentage points)"
                ]
            ]
            .set_index(
                "Feature"
            )
        )


        st.bar_chart(
            chart_data
        )


    st.info(
        "These values describe model sensitivity. "
        "They are not estimates of causal effects."
    )


    # ========================================================
    # COUNTERFACTUAL SIMULATION
    # ========================================================

    st.header(
        "6. Constrained Counterfactual Simulator"
    )


    st.write(
        "Explore alternative values for selected variables "
        "while keeping the remaining patient profile unchanged."
    )


    scenario = patient.copy()


    c1, c2 = st.columns(
        2
    )


    with c1:

        glucose = st.slider(
            "Admission glucose (mmol/L)",
            3.0,
            20.0,
            safe_float(
                patient[
                    "admission_glucose_mmol_l"
                ],
                7.0
            ),
            0.1
        )


        sbp = st.slider(
            "Systolic blood pressure (mmHg)",
            80,
            240,
            safe_int(
                patient[
                    "systolic_bp"
                ],
                140
            ),
            1
        )


    with c2:

        onset = st.slider(
            "Onset-to-door time (minutes)",
            20,
            720,
            max(
                20,
                min(
                    720,
                    safe_int(
                        patient[
                            "onset_to_door_min"
                        ],
                        120
                    )
                )
            ),
            5
        )


        current_dtn = (
            safe_int(
                patient[
                    "door_to_needle_min"
                ],
                60
            )
        )


        door_to_needle = st.slider(
            "Door-to-needle time (minutes)",
            20,
            180,
            max(
                20,
                min(
                    180,
                    current_dtn
                )
            ),
            5,
            disabled=(
                safe_int(
                    patient[
                        "iv_thrombolysis"
                    ]
                ) == 0
            )
        )


    scenario[
        "admission_glucose_mmol_l"
    ] = glucose

    scenario[
        "systolic_bp"
    ] = sbp

    scenario[
        "onset_to_door_min"
    ] = onset


    if safe_int(
        patient[
            "iv_thrombolysis"
        ]
    ) == 1:

        scenario[
            "door_to_needle_min"
        ] = door_to_needle


    scenario_ht = predict_binary(
        scenario,
        M1
    )


    risk_change = (
        scenario_ht
        - ht_risk
    )


    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(
        "Current model estimate",
        f"{ht_risk * 100:.1f}%"
    )


    c2.metric(
        "Alternative scenario",
        f"{scenario_ht * 100:.1f}%"
    )


    c3.metric(
        "Difference",
        f"{risk_change * 100:+.1f} pp",
        delta_color="inverse"
    )


    st.warning(
        "The difference is a model-based counterfactual "
        "scenario, not a treatment effect."
    )


    # ========================================================
    # COUNTERFACTUAL TRAJECTORY
    # ========================================================

    st.subheader(
        "Scenario Risk Trajectory"
    )


    glucose_values = np.linspace(
        4,
        15,
        30
    )


    trajectory = []


    for value in glucose_values:

        temp_patient = patient.copy()

        temp_patient[
            "admission_glucose_mmol_l"
        ] = value


        temp_risk = predict_binary(
            temp_patient,
            M1
        )


        trajectory.append(
            {
                "Glucose":
                    value,

                "HT risk (%)":
                    temp_risk * 100
            }
        )


    trajectory_df = pd.DataFrame(
        trajectory
    )


    st.line_chart(
        trajectory_df,
        x="Glucose",
        y="HT risk (%)"
    )


    st.caption(
        "This trajectory shows the trained model's response "
        "to changing glucose while holding other variables "
        "constant. It does not demonstrate causality."
    )


    # ========================================================
    # DECISION SUPPORT
    # ========================================================

    st.header(
        "7. Human-in-the-Loop Decision Support"
    )


    st.write(
        "The platform separates model prediction from "
        "clinical interpretation."
    )


    findings = []


    if ht_risk >= 0.20:

        findings.append(
            "The model produces a comparatively higher "
            "HT estimate for this synthetic patient."
        )


    if safe_float(
        patient[
            "admission_glucose_mmol_l"
        ]
    ) > 10:

        findings.append(
            "Admission glucose is elevated in the "
            "synthetic patient profile."
        )


    if safe_int(
        patient[
            "baseline_nihss"
        ]
    ) >= 16:

        findings.append(
            "The patient has a high baseline NIHSS "
            "within this simulated cohort."
        )


    if safe_int(
        patient[
            "aspects"
        ]
    ) <= 6:

        findings.append(
            "The patient has a relatively low ASPECTS "
            "within the simulated profile."
        )


    if safe_int(
        patient[
            "atrial_fibrillation"
        ]
    ) == 1:

        findings.append(
            "Atrial fibrillation is present."
        )


    if safe_int(
        patient[
            "large_vessel_occlusion"
        ]
    ) == 1:

        findings.append(
            "Large vessel occlusion is present."
        )


    if not findings:

        findings.append(
            "No predefined high-priority prototype flag "
            "was triggered."
        )


    for finding in findings:

        st.write(
            f"• {finding}"
        )


    st.subheader(
        "Counterfactual interpretation"
    )


    if risk_change < -0.005:

        st.write(
            f"The alternative scenario is associated with "
            f"a {abs(risk_change) * 100:.1f} percentage-point "
            f"lower model-estimated HT risk."
        )

    elif risk_change > 0.005:

        st.write(
            f"The alternative scenario is associated with "
            f"a {abs(risk_change) * 100:.1f} percentage-point "
            f"higher model-estimated HT risk."
        )

    else:

        st.write(
            "The selected alternative scenario produces "
            "little change in the model-estimated HT risk."
        )


    # ========================================================
    # EVIDENCE / LLM ARCHITECTURE
    # ========================================================

    st.header(
        "8. Evidence-Grounded Explanation"
    )


    st.write(
        """
The intended decision-support workflow is:

**Patient data → predictive model → model explanation →
constrained counterfactual simulation → retrieved clinical
evidence → clinician review.**

A generative AI layer can subsequently convert these outputs
and retrieved evidence into a concise explanation. It should
not independently determine treatment.
"""
    )


    st.info(
        "The current public prototype intentionally does not "
        "generate autonomous medication or treatment orders."
    )


    # ========================================================
    # MODEL TRANSPARENCY
    # ========================================================

    with st.expander(
        "Model transparency"
    ):

        rows = []


        for model_name, model in [

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

            metrics = model.get(
                "metrics",
                {}
            )


            rows.append(
                {
                    "Model":
                        model_name,

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
                        ),

                    "AUROC":
                        metrics.get(
                            "AUROC",
                            np.nan
                        ),

                    "MAE":
                        metrics.get(
                            "MAE_mRS",
                            np.nan
                        ),

                    "QWK":
                        metrics.get(
                            "quadratic_weighted_kappa",
                            np.nan
                        )
                }
            )


        transparency_df = pd.DataFrame(
            rows
        )


        st.dataframe(
            transparency_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MIMIC PATIENT EXPLORER
# ============================================================

elif page == "MIMIC Patient Explorer":

    st.title(
        "🏥 MIMIC-IV Patient Explorer"
    )


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


    patient_record = patients[
        patients[
            "subject_id"
        ] == selected_patient
    ]


    patient_admissions = admissions[
        admissions[
            "subject_id"
        ] == selected_patient
    ]


    patient_diagnoses = diagnoses[
        diagnoses[
            "subject_id"
        ] == selected_patient
    ]


    st.subheader(
        "Demographics"
    )

    st.dataframe(
        patient_record,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Admissions"
    )

    st.dataframe(
        patient_admissions,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Diagnoses"
    )


    if not diagnosis_dictionary.empty:

        patient_diagnoses = (
            patient_diagnoses.merge(
                diagnosis_dictionary,
                on=[
                    "icd_code",
                    "icd_version"
                ],
                how="left"
            )
        )


    st.dataframe(
        patient_diagnoses,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MIMIC STROKE COHORT
# ============================================================

elif page == "MIMIC Stroke Cohort":

    st.title(
        "🧠 MIMIC-IV Stroke Cohort"
    )


    if stroke_cohort.empty:

        st.info(
            "No stroke diagnoses were identified."
        )

    else:

        c1, c2, c3 = st.columns(
            3
        )


        c1.metric(
            "Patients",
            stroke_cohort[
                "subject_id"
            ].nunique()
        )


        c2.metric(
            "Admissions",
            stroke_cohort[
                "hadm_id"
            ].nunique()
        )


        c3.metric(
            "Stroke diagnosis records",
            len(
                stroke_cohort
            )
        )


        stroke_types = (
            stroke_cohort[
                "stroke_type"
            ]
            .value_counts()
            .rename_axis(
                "Stroke type"
            )
            .reset_index(
                name="Records"
            )
        )


        st.bar_chart(
            stroke_types,
            x="Stroke type",
            y="Records"
        )


        display_columns = [
            column
            for column in [
                "subject_id",
                "hadm_id",
                "gender",
                "anchor_age",
                "stroke_type",
                "icd_code",
                "long_title",
                "admittime",
                "dischtime"
            ]
            if column in stroke_cohort.columns
        ]


        st.dataframe(
            stroke_cohort[
                display_columns
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AdaptStroke AI | Research prototype | "
    "Synthetic models are not clinically validated."
)