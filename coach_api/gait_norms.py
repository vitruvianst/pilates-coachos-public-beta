"""Pure gait normative calculations extracted from gait_eval.py.

This module intentionally excludes OpenCV/MediaPipe/Tkinter so the Coach API
can run on a lightweight server while preserving the same regression and PI
logic used by the assessment pipeline.
"""

from __future__ import annotations

import math
from typing import Any

from scipy import stats

REGRESSION_DATA = {
    "m": {
        "Gait Speed": {"slope": -0.5715, "intercept": 129.6785, "se": 8.6989, "df": 300},
        "Step Length": {"slope": -0.2531, "intercept": 68.1054, "se": 3.2946, "df": 300},
        "Stride Length": {"slope": -0.5063, "intercept": 136.2108, "se": 6.5892, "df": 300},
        "Cadence": {"slope": -0.4611, "intercept": 130.7106, "se": 7.6887, "df": 300},
        "Stance Time %GC": {"slope": 0.0876, "intercept": 56.432, "se": 1.876, "df": 300},
    },
    "f": {
        "Gait Speed": {"slope": -0.7450, "intercept": 141.4672, "se": 9.2446, "df": 300},
        "Step Length": {"slope": -0.3011, "intercept": 66.7615, "se": 3.1059, "df": 300},
        "Stride Length": {"slope": -0.6021, "intercept": 133.5231, "se": 6.2118, "df": 300},
        "Cadence": {"slope": -0.5889, "intercept": 137.5884, "se": 8.2107, "df": 300},
        "Stance Time %GC": {"slope": 0.0987, "intercept": 55.876, "se": 1.943, "df": 300},
    },
}

COMMON_HEIGHT_M = {"m": 1.64, "f": 1.53}


def calculate_parameter(
    param: str,
    gender: str,
    age: float,
    height_m: float,
    common_height_m: float,
    measured: float = 0.0,
) -> dict[str, Any]:
    data = REGRESSION_DATA[gender][param]
    slope = data["slope"]
    intercept = data["intercept"]
    se = data["se"]
    df = data["df"]

    predicted_common = slope * age + intercept

    if param in ["Gait Speed", "Step Length", "Stride Length"]:
        predicted = predicted_common * (height_m / common_height_m)
        measured_adj = measured * (common_height_m / height_m)
    elif param == "Cadence":
        predicted = predicted_common * math.sqrt(common_height_m / height_m)
        measured_adj = measured * math.sqrt(common_height_m / height_m)
    else:
        predicted = predicted_common
        measured_adj = measured

    t_value = stats.t.ppf(0.975, df)
    ci_margin = t_value * se
    pi_margin = t_value * se * math.sqrt(1 + 1 / (df + 2))

    ci_lower = predicted - ci_margin
    ci_upper = predicted + ci_margin
    pi_lower = predicted - pi_margin
    pi_upper = predicted + pi_margin

    t_stat = abs(measured_adj - predicted_common) / se
    percentile = stats.t.cdf(t_stat, df)
    if measured_adj < predicted_common:
        percentile = 1 - percentile

    return {
        "parameter_name": param,
        "measured": measured,
        "predicted": predicted,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pi_lower": pi_lower,
        "pi_upper": pi_upper,
        "percentile": percentile,
    }


def build_gait_references(age: int, gender: str, height_m: float) -> dict[str, str]:
    """Return CoachOS display references for age 65+ using gait_eval PI logic."""
    if gender not in COMMON_HEIGHT_M:
        raise ValueError("gender must be 'm' or 'f'")
    if not height_m or height_m <= 0:
        raise ValueError("height_m must be positive")

    common = COMMON_HEIGHT_M[gender]
    mapping = {
        "gait_speed_cm_s": ("Gait Speed", "cm/s"),
        "step_length_cm": ("Step Length", "cm"),
        "cadence_spm": ("Cadence", "steps/min"),
    }
    out: dict[str, str] = {}
    for key, (param, unit) in mapping.items():
        result = calculate_parameter(param, gender, age, height_m, common, measured=0.0)
        # report_generator.py renders PI bounds as whole numbers.
        out[key] = f"{result['pi_lower']:.0f} - {result['pi_upper']:.0f} {unit}"
    out["symmetry_pct"] = "> 90%"
    return out
