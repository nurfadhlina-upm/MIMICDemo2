import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

SEED = 42
N = 2000

rng = np.random.default_rng(SEED)

OUTPUT_DIR = Path(__file__).parent / "synthetic_data"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "synthetic_asean_stroke_v2.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary(prob):
    prob = np.clip(prob, 0, 1)
    return rng.binomial(1, prob)


def clip_round(x, low, high, decimals=0):
    x = np.clip(x, low, high)

    if decimals == 0:
        return np.round(x).astype(int)

    return np.round(x, decimals)


# ============================================================
# IDENTIFIERS
# ============================================================

stroke_id = [
    f"S{i:04d}"
    for i in range(1, N + 1)
]


# ============================================================
# COUNTRY
#
# These proportions are simulation design choices.
# They are NOT intended to represent ASEAN stroke prevalence.
# ============================================================

country = rng.choice(
    [
        "Malaysia",
        "Indonesia",
        "Vietnam",
        "Philippines"
    ],
    size=N,
    p=[0.30, 0.30, 0.20, 0.20]
)


# ============================================================
# DEMOGRAPHICS
# ============================================================

age = clip_round(
    rng.normal(65, 13, N),
    18,
    95
)

sex = rng.choice(
    ["Male", "Female"],
    size=N,
    p=[0.57, 0.43]
)


# ============================================================
# COMORBIDITIES
# ============================================================

hypertension = binary(
    sigmoid(
        0.30
        + 0.035 * (age - 60)
    )
)

diabetes = binary(
    sigmoid(
        -1.15
        + 0.025 * (age - 60)
    )
)

atrial_fibrillation = binary(
    sigmoid(
        -2.15
        + 0.055 * (age - 60)
    )
)

prior_stroke = binary(
    sigmoid(
        -2.0
        + 0.025 * (age - 60)
        + 0.40 * hypertension
    )
)

dyslipidemia = binary(
    sigmoid(
        -0.70
        + 0.020 * (age - 55)
        + 0.30 * diabetes
    )
)

smoking_probability = np.where(
    sex == "Male",
    0.32,
    0.09
)

smoking = binary(
    smoking_probability
)


# ============================================================
# PRE-STROKE mRS
# ============================================================

prestroke_latent = (
    -2.5
    + 0.045 * (age - 65)
    + 0.60 * prior_stroke
    + 0.35 * diabetes
    + rng.normal(0, 0.8, N)
)

prestroke_mrs = np.select(
    [
        prestroke_latent < -1.2,
        prestroke_latent < -0.3,
        prestroke_latent < 0.5,
        prestroke_latent < 1.3,
        prestroke_latent < 2.0
    ],
    [0, 1, 2, 3, 4],
    default=5
).astype(int)


# ============================================================
# BASELINE NIHSS
#
# Mixed distribution:
# mild + moderate + severe patients.
# ============================================================

severity_group = rng.choice(
    ["mild", "moderate", "severe"],
    size=N,
    p=[0.40, 0.42, 0.18]
)

nihss = np.zeros(N)

mild = severity_group == "mild"
moderate = severity_group == "moderate"
severe = severity_group == "severe"

nihss[mild] = rng.normal(
    3.5,
    2.0,
    mild.sum()
)

nihss[moderate] = rng.normal(
    10,
    3.5,
    moderate.sum()
)

nihss[severe] = rng.normal(
    20,
    5,
    severe.sum()
)

nihss = clip_round(
    nihss,
    0,
    42
)


def nihss_category(score):

    if score <= 4:
        return "Minor"

    elif score <= 15:
        return "Moderate"

    elif score <= 20:
        return "Moderate-Severe"

    return "Severe"


nihss_severity = [
    nihss_category(x)
    for x in nihss
]


# ============================================================
# STROKE CHARACTERISTICS
# ============================================================

lvo = binary(
    sigmoid(
        -2.3
        + 0.12 * nihss
    )
)

stroke_territory = []

for lvo_status in lvo:

    if lvo_status == 1:

        territory = rng.choice(
            [
                "MCA",
                "ICA",
                "Basilar",
                "ACA",
                "PCA"
            ],
            p=[
                0.58,
                0.18,
                0.10,
                0.06,
                0.08
            ]
        )

    else:

        territory = rng.choice(
            [
                "MCA",
                "Lacunar",
                "PCA",
                "ACA",
                "Other"
            ],
            p=[
                0.35,
                0.32,
                0.12,
                0.06,
                0.15
            ]
        )

    stroke_territory.append(
        territory
    )


# ============================================================
# TOAST SUBTYPE
# ============================================================

toast_subtype = []

for af, small in zip(
    atrial_fibrillation,
    np.array(stroke_territory) == "Lacunar"
):

    if af == 1:

        probs = [
            0.15,
            0.58,
            0.07,
            0.05,
            0.15
        ]

    elif small:

        probs = [
            0.08,
            0.05,
            0.70,
            0.04,
            0.13
        ]

    else:

        probs = [
            0.38,
            0.18,
            0.20,
            0.06,
            0.18
        ]

    subtype = rng.choice(
        [
            "Large artery atherosclerosis",
            "Cardioembolism",
            "Small vessel occlusion",
            "Other determined",
            "Undetermined"
        ],
        p=probs
    )

    toast_subtype.append(
        subtype
    )


# ============================================================
# ASPECTS
# ============================================================

aspects = clip_round(
    10
    - 0.11 * nihss
    - 0.7 * lvo
    + rng.normal(0, 1.1, N),
    0,
    10
)


# ============================================================
# VITAL SIGNS
# ============================================================

sbp = clip_round(
    rng.normal(148, 22, N)
    + hypertension * 10,
    80,
    250
)

dbp = clip_round(
    rng.normal(82, 13, N)
    + hypertension * 5,
    40,
    150
)


# ============================================================
# LABORATORY VARIABLES
# ============================================================

glucose = clip_round(
    rng.normal(6.8, 1.4, N)
    + diabetes * 2.8
    + 0.035 * nihss,
    3,
    25,
    1
)

hba1c = clip_round(
    rng.normal(5.7, 0.6, N)
    + diabetes * 1.8,
    4,
    15,
    1
)

platelets = clip_round(
    rng.normal(235, 60, N),
    60,
    600
)

inr = clip_round(
    rng.normal(1.05, 0.12, N)
    + atrial_fibrillation * 0.10,
    0.8,
    3.5,
    2
)

creatinine = clip_round(
    rng.normal(85, 25, N)
    + diabetes * 10
    + 0.35 * (age - 65),
    35,
    400,
    1
)


# ============================================================
# TIME VARIABLES
# ============================================================

onset_to_door_min = clip_round(
    rng.lognormal(
        np.log(120),
        0.55,
        N
    ),
    20,
    720
)

door_to_needle_min = clip_round(
    rng.normal(
        55,
        18,
        N
    ),
    20,
    180
)


# ============================================================
# IV THROMBOLYSIS
# ============================================================

ivt_probability = sigmoid(
    -0.9
    + 0.045 * nihss
    - 0.006 * (
        onset_to_door_min - 120
    )
    - 0.025 * np.maximum(
        age - 85,
        0
    )
)

iv_thrombolysis = binary(
    ivt_probability
)


# ============================================================
# MECHANICAL THROMBECTOMY
# ============================================================

mt_probability = sigmoid(
    -3.0
    + 2.8 * lvo
    + 0.055 * nihss
)

mechanical_thrombectomy = binary(
    mt_probability
)


# ============================================================
# RECANALISATION / TICI
# ============================================================

tici_score = []

for mt in mechanical_thrombectomy:

    if mt == 0:

        tici_score.append(
            "Not applicable"
        )

    else:

        tici_score.append(
            rng.choice(
                [
                    "0-2a",
                    "2b",
                    "2c",
                    "3"
                ],
                p=[
                    0.18,
                    0.35,
                    0.20,
                    0.27
                ]
            )
        )


successful_recanalisation = np.array([
    1 if x in ["2b", "2c", "3"]
    else 0
    for x in tici_score
])


# ============================================================
# HEMORRHAGIC TRANSFORMATION
#
# This is a simulated outcome.
# It is NOT a clinical risk equation.
# ============================================================

ht_logit = (
    -3.20
    + 0.075 * nihss
    + 0.09 * (glucose - 7)
    + 0.45 * atrial_fibrillation
    + 0.45 * iv_thrombolysis
    + 0.35 * mechanical_thrombectomy
    + 0.35 * lvo
    - 0.12 * (aspects - 7)
)

ht_probability = sigmoid(
    ht_logit
)

hemorrhagic_transformation = binary(
    ht_probability
)


# ============================================================
# HT SUBTYPE
# ============================================================

ht_subtype = []

for ht in hemorrhagic_transformation:

    if ht == 0:

        ht_subtype.append(
            "None"
        )

    else:

        ht_subtype.append(
            rng.choice(
                [
                    "HI1",
                    "HI2",
                    "PH1",
                    "PH2"
                ],
                p=[
                    0.32,
                    0.30,
                    0.23,
                    0.15
                ]
            )
        )


# ============================================================
# SYMPTOMATIC ICH
# ============================================================

ph = np.array([
    1 if x in ["PH1", "PH2"]
    else 0
    for x in ht_subtype
])

sich_probability = sigmoid(
    -4.2
    + 0.07 * nihss
    + 1.45 * ph
    + 0.04 * (glucose - 7)
    + 0.25 * iv_thrombolysis
)

symptomatic_ich = binary(
    sich_probability
)


# ============================================================
# DISCHARGE NIHSS
# ============================================================

improvement = (
    rng.normal(
        3.0,
        3.0,
        N
    )
    + 1.3 * iv_thrombolysis
    + 2.5 * (
        mechanical_thrombectomy
        * successful_recanalisation
    )
    - 2.5 * hemorrhagic_transformation
    - 3.0 * symptomatic_ich
)

discharge_nihss = clip_round(
    nihss - improvement,
    0,
    42
)


# ============================================================
# DISCHARGE mRS
# ============================================================

discharge_latent = (
    -1.4
    + 0.14 * nihss
    + 0.025 * (age - 60)
    + 0.55 * prestroke_mrs
    + 0.25 * diabetes
    + 0.30 * prior_stroke
    + 0.65 * hemorrhagic_transformation
    + 1.0 * symptomatic_ich
    - 0.30 * iv_thrombolysis
    - 0.65 * (
        mechanical_thrombectomy
        * successful_recanalisation
    )
    + rng.normal(
        0,
        1.15,
        N
    )
)

discharge_mrs = np.select(
    [
        discharge_latent < -0.5,
        discharge_latent < 0.3,
        discharge_latent < 1.1,
        discharge_latent < 1.9,
        discharge_latent < 2.7,
        discharge_latent < 3.5
    ],
    [0, 1, 2, 3, 4, 5],
    default=6
).astype(int)


# ============================================================
# 90-DAY mRS
#
# Allows improvement/deterioration after discharge.
# ============================================================

mrs90_latent = (
    discharge_latent
    - 0.35
    + 0.035 * discharge_nihss
    + 0.25 * prior_stroke
    + 0.20 * diabetes
    + 0.45 * symptomatic_ich
    + rng.normal(
        0,
        0.8,
        N
    )
)

mrs_90d = np.select(
    [
        mrs90_latent < -0.5,
        mrs90_latent < 0.3,
        mrs90_latent < 1.1,
        mrs90_latent < 1.9,
        mrs90_latent < 2.7,
        mrs90_latent < 3.5
    ],
    [0, 1, 2, 3, 4, 5],
    default=6
).astype(int)


# ============================================================
# 90-DAY MORTALITY
# ============================================================

mortality_probability = sigmoid(
    -4.4
    + 0.050 * (age - 60)
    + 0.075 * nihss
    + 0.40 * prestroke_mrs
    + 0.40 * atrial_fibrillation
    + 0.65 * hemorrhagic_transformation
    + 1.20 * symptomatic_ich
)

mortality_90d = binary(
    mortality_probability
)

mrs_90d[
    mortality_90d == 1
] = 6


# ============================================================
# FUNCTIONAL INDEPENDENCE
# ============================================================

functional_independence_discharge = (
    discharge_mrs <= 2
).astype(int)

functional_independence_90d = (
    mrs_90d <= 2
).astype(int)


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame({

    "stroke_id": stroke_id,

    "data_source":
        "SYNTHETIC_ASEAN_V2",

    "country": country,

    "age": age,

    "sex": sex,

    "hypertension":
        hypertension,

    "diabetes":
        diabetes,

    "atrial_fibrillation":
        atrial_fibrillation,

    "prior_stroke":
        prior_stroke,

    "dyslipidemia":
        dyslipidemia,

    "smoking":
        smoking,

    "prestroke_mrs":
        prestroke_mrs,

    "baseline_nihss":
        nihss,

    "nihss_severity":
        nihss_severity,

    "stroke_territory":
        stroke_territory,

    "toast_subtype":
        toast_subtype,

    "large_vessel_occlusion":
        lvo,

    "aspects":
        aspects,

    "systolic_bp":
        sbp,

    "diastolic_bp":
        dbp,

    "admission_glucose_mmol_l":
        glucose,

    "hba1c":
        hba1c,

    "platelets_10e9_l":
        platelets,

    "inr":
        inr,

    "creatinine_umol_l":
        creatinine,

    "onset_to_door_min":
        onset_to_door_min,

    "iv_thrombolysis":
        iv_thrombolysis,

    "door_to_needle_min":
        np.where(
            iv_thrombolysis == 1,
            door_to_needle_min,
            np.nan
        ),

    "mechanical_thrombectomy":
        mechanical_thrombectomy,

    "tici_score":
        tici_score,

    "successful_recanalisation":
        successful_recanalisation,

    "hemorrhagic_transformation":
        hemorrhagic_transformation,

    "ht_subtype":
        ht_subtype,

    "symptomatic_ich":
        symptomatic_ich,

    "discharge_nihss":
        discharge_nihss,

    "discharge_mrs":
        discharge_mrs,

    "functional_independence_discharge":
        functional_independence_discharge,

    "mrs_90d":
        mrs_90d,

    "functional_independence_90d":
        functional_independence_90d,

    "mortality_90d":
        mortality_90d

})


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION REPORT
# ============================================================

median_nihss = df[
    "baseline_nihss"
].median()

discharge_independence = (
    df[
        "functional_independence_discharge"
    ].mean() * 100
)

independence_90d = (
    df[
        "functional_independence_90d"
    ].mean() * 100
)

ht_rate = (
    df[
        "hemorrhagic_transformation"
    ].mean() * 100
)

sich_rate = (
    df[
        "symptomatic_ich"
    ].mean() * 100
)

mortality_rate = (
    df[
        "mortality_90d"
    ].mean() * 100
)


print()
print(
    "=============================================="
)
print(
    "SYNTHETIC ASEAN STROKE V2 DATASET CREATED"
)
print(
    "=============================================="
)

print(
    f"Patients: {len(df):,}"
)

print(
    f"Median baseline NIHSS: "
    f"{median_nihss:.1f}"
)

print(
    f"Discharge mRS 0-2: "
    f"{discharge_independence:.1f}%"
)

print(
    f"90-day mRS 0-2: "
    f"{independence_90d:.1f}%"
)

print(
    f"Hemorrhagic transformation: "
    f"{ht_rate:.1f}%"
)

print(
    f"Symptomatic ICH: "
    f"{sich_rate:.1f}%"
)

print(
    f"90-day mortality: "
    f"{mortality_rate:.1f}%"
)


# ============================================================
# CALIBRATION CHECKS
#
# These are simulation design ranges, NOT claims of true
# ASEAN-wide prevalence.
# ============================================================

checks = {

    "Median NIHSS":
        7 <= median_nihss <= 11,

    "Discharge mRS 0-2":
        45 <= discharge_independence <= 70,

    "90-day mRS 0-2":
        45 <= independence_90d <= 70,

    "HT":
        7 <= ht_rate <= 18,

    "sICH":
        3 <= sich_rate <= 9,

    "90-day mortality":
        6 <= mortality_rate <= 15
}


print()
print(
    "CALIBRATION CHECK"
)
print(
    "----------------------------------------------"
)

all_pass = True

for name, passed in checks.items():

    status = (
        "PASS"
        if passed
        else "REVIEW"
    )

    print(
        f"{name:<25} {status}"
    )

    if not passed:
        all_pass = False


print()

if all_pass:

    print(
        "All simulation calibration checks passed."
    )

else:

    print(
        "One or more distributions should be "
        "reviewed before modelling."
    )


print()
print(
    f"Saved to:\n{OUTPUT_FILE}"
)