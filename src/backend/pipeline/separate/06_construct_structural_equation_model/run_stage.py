from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from semopy import Model, calc_stats
from semopy.inspector import inspect as sem_inspect

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.data_utils import available_moderator_columns, zscore_series
from src.backend.utils.io import abs_path, ensure_dir, stage_manifest_path, write_json, write_text
from src.backend.utils.logging_utils import setup_logging
from src.backend.utils.methodology_audit import (
    build_assumption_register,
    build_peer_review_critique_notes,
    render_methodology_audit_text,
)
from src.backend.utils.schemas import SemCoefficient, SemFitResult, StageArtifactManifest, StageConfig

LOGGER = logging.getLogger(__name__)


PRIMARY_OLS_FORMULA = (
    "delta_score ~ attack_present + baseline_score + primary_moderator_z + attack_x_primary_moderator"
)


class Stage06Config(StageConfig):
    primary_moderator: str = "profile_cont_susceptibility_index"
    bootstrap_samples: int = 500


def _pretty_moderator_label(column_name: str) -> str:
    label = column_name
    for prefix in ["profile_cont_", "profile_cat__", "profile_cat_"]:
        if label.startswith(prefix):
            label = label[len(prefix) :]
    label = label.replace("__", " ")
    label = label.replace("_", " ").strip()
    return " ".join(part.capitalize() if part.lower() != "pct" else "%" for part in label.split())


def _normalize_fit_indices(raw: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "Value" in value:
            inner = value["Value"]
            flat[key] = float(inner) if hasattr(inner, "__float__") else inner
        else:
            flat[key] = float(value) if hasattr(value, "__float__") else value
    return flat


def _fit_sem(df: pd.DataFrame, primary_moderator_label: str) -> SemFitResult:
    model_formula = (
        "delta_score ~ attack_present + baseline_score + primary_moderator_z + attack_x_primary_moderator\n"
        "post_score ~ baseline_score + attack_present + primary_moderator_z + attack_x_primary_moderator"
    )

    model = Model(model_formula)
    warnings: List[str] = []

    try:
        model.fit(df)
        converged = True
    except Exception as exc:
        converged = False
        warnings.append(f"semopy fit failed: {exc}")

    coeffs: List[SemCoefficient] = []
    fit_indices: Dict[str, Any] = {}

    if converged:
        est = sem_inspect(model)
        for _, row in est.iterrows():
            rhs = str(row.get("rval", ""))
            if rhs == "primary_moderator_z":
                rhs = f"{primary_moderator_label} (z)"
            elif rhs == "attack_x_primary_moderator":
                rhs = f"attack_present x {primary_moderator_label} (z)"
            coeffs.append(
                SemCoefficient(
                    lhs=str(row.get("lval", "")),
                    op=str(row.get("op", "")),
                    rhs=rhs,
                    estimate=float(row.get("Estimate", 0.0)),
                    std_error=(None if pd.isna(row.get("Std. Err")) else float(row.get("Std. Err"))),
                    z_value=(None if pd.isna(row.get("z-value")) else float(row.get("z-value"))),
                    p_value=(None if pd.isna(row.get("p-value")) else float(row.get("p-value"))),
                )
            )

        stats = calc_stats(model)
        if hasattr(stats, "to_dict"):
            fit_indices = _normalize_fit_indices(stats.to_dict())

    return SemFitResult(
        model_name="semopy_primary_moderation_path",
        model_formula=model_formula,
        converged=converged,
        n_obs=len(df),
        fit_indices=fit_indices,
        coefficients=coeffs,
        warnings=warnings,
    )


def _bootstrap_primary_model(
    df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    terms: List[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: List[Dict[str, float]] = []

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(df), size=len(df))
        sample = df.iloc[sample_idx].copy()
        try:
            model = smf.ols(PRIMARY_OLS_FORMULA, data=sample).fit()
        except Exception:
            continue
        records.append({term: float(model.params.get(term, np.nan)) for term in terms})

    if not records:
        return pd.DataFrame(
            {
                "term": terms,
                "bootstrap_mean": np.nan,
                "bootstrap_std": np.nan,
                "conf_low": np.nan,
                "conf_high": np.nan,
                "n_bootstrap_success": 0,
            }
        )

    boot_df = pd.DataFrame(records)
    summary_rows: List[Dict[str, float | str | int]] = []
    for term in terms:
        values = boot_df[term].dropna()
        summary_rows.append(
            {
                "term": term,
                "bootstrap_mean": float(values.mean()) if len(values) else np.nan,
                "bootstrap_std": float(values.std(ddof=0)) if len(values) else np.nan,
                "conf_low": float(values.quantile(0.025)) if len(values) else np.nan,
                "conf_high": float(values.quantile(0.975)) if len(values) else np.nan,
                "n_bootstrap_success": int(len(values)),
            }
        )
    return pd.DataFrame(summary_rows)


def _fit_exploratory_moderator_models(
    df: pd.DataFrame,
    moderator_columns: List[str],
    primary_moderator: str,
) -> pd.DataFrame:
    columns = [
        "moderator_column",
        "moderator_label",
        "role",
        "interaction_estimate",
        "interaction_std_error",
        "interaction_p_value",
        "interaction_conf_low",
        "interaction_conf_high",
        "moderator_main_estimate",
        "moderator_main_p_value",
    ]
    rows: List[Dict[str, object]] = []
    for column in moderator_columns:
        if column not in df.columns or df[column].nunique() <= 1:
            continue

        work = df.copy()
        work["candidate_moderator_z"] = zscore_series(work[column].astype(float))
        work["attack_x_candidate_moderator"] = (
            work["attack_present"].astype(float) * work["candidate_moderator_z"].astype(float)
        )
        try:
            result = smf.ols(
                "delta_score ~ attack_present + baseline_score + candidate_moderator_z + attack_x_candidate_moderator",
                data=work,
            ).fit(cov_type="HC3")
        except Exception as exc:
            LOGGER.warning("Exploratory moderator model failed for %s: %s", column, exc)
            continue

        conf = result.conf_int()
        rows.append(
            {
                "moderator_column": column,
                "moderator_label": _pretty_moderator_label(column),
                "role": "primary" if column == primary_moderator else "exploratory",
                "interaction_estimate": float(result.params.get("attack_x_candidate_moderator", np.nan)),
                "interaction_std_error": float(result.bse.get("attack_x_candidate_moderator", np.nan)),
                "interaction_p_value": float(result.pvalues.get("attack_x_candidate_moderator", np.nan)),
                "interaction_conf_low": float(conf.loc["attack_x_candidate_moderator", 0]),
                "interaction_conf_high": float(conf.loc["attack_x_candidate_moderator", 1]),
                "moderator_main_estimate": float(result.params.get("candidate_moderator_z", np.nan)),
                "moderator_main_p_value": float(result.pvalues.get("candidate_moderator_z", np.nan)),
            }
        )

    comparison = pd.DataFrame(rows, columns=columns)
    if comparison.empty:
        return comparison
    return comparison.sort_values(["role", "interaction_p_value", "moderator_label"]).reset_index(drop=True)


def _render_report(
    df: pd.DataFrame,
    sem_result: SemFitResult,
    ols_summary: str,
    ols_table: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    exploratory_table: pd.DataFrame,
    primary_moderator: str,
    run_id: str,
) -> str:
    attack_mask = df["attack_present"] == 1
    control_mask = df["attack_present"] == 0

    attack_mean = float(df.loc[attack_mask, "delta_score"].mean()) if attack_mask.any() else 0.0
    control_mean = float(df.loc[control_mask, "delta_score"].mean()) if control_mask.any() else 0.0

    realism_text = "n/a"
    if "attack_realism_score" in df.columns:
        realism_vals = df.loc[attack_mask, "attack_realism_score"].dropna()
        if len(realism_vals) > 0:
            realism_text = f"{float(realism_vals.mean()):.3f}"

    fit_cfi = sem_result.fit_indices.get("CFI")
    fit_rmsea = sem_result.fit_indices.get("RMSEA")
    fit_line = f"CFI={fit_cfi:.3f}, RMSEA={fit_rmsea:.3f}" if fit_cfi is not None and fit_rmsea is not None else "fit indices unavailable"

    primary_label = _pretty_moderator_label(primary_moderator)
    bootstrap_lookup = {
        row["term"]: row
        for row in bootstrap_table.to_dict(orient="records")
    }

    lines = [
        f"Moderation Report - {run_id}",
        "=========================",
        "",
        f"Rows analyzed: {len(df)}",
        f"Primary moderator: {primary_moderator} ({primary_label})",
        f"Attack-present mean delta: {attack_mean:.3f}",
        f"Control mean delta: {control_mean:.3f}",
        f"Mean attack realism score: {realism_text}",
        "",
        "SEM Status",
        "----------",
        f"Converged: {sem_result.converged}",
        f"Fit indices: {fit_line}",
        f"Warnings: {', '.join(sem_result.warnings) if sem_result.warnings else 'none'}",
        "",
        "Primary Robust Model",
        "--------------------",
    ]

    for row in ols_table.to_dict(orient="records"):
        boot = bootstrap_lookup.get(row["term"], {})
        lines.append(
            (
                f"{row['term']}: est={row['estimate']:.4f}, p={row['p_value']:.6f}, "
                f"boot95=[{boot.get('conf_low', np.nan):.4f}, {boot.get('conf_high', np.nan):.4f}]"
            )
        )

    if not exploratory_table.empty:
        lines.extend(["", "Exploratory Moderators", "---------------------"])
        for row in exploratory_table.head(8).to_dict(orient="records"):
            lines.append(
                (
                    f"{row['moderator_label']}: interaction est={row['interaction_estimate']:.4f}, "
                    f"p={row['interaction_p_value']:.6f}"
                )
            )

    lines.extend(
        [
            "",
            "SEM Coefficients (first 12)",
            "---------------------------",
        ]
    )
    for coeff in sem_result.coefficients[:12]:
        lines.append(f"{coeff.lhs} {coeff.op} {coeff.rhs}: est={coeff.estimate:.4f}, p={coeff.p_value}")

    lines.extend(
        [
            "",
            "OLS Supplement",
            "--------------",
            ols_summary,
            "",
            "Caveat",
            "------",
            "This is a pilot simulation. Estimates are directional and exploratory; they should not be interpreted as causal real-world effects.",
        ]
    )
    return "\n".join(lines)


def run_stage(input_path: str, output_dir: str, config: Stage06Config) -> StageArtifactManifest:
    ensure_dir(output_dir)
    df = pd.read_csv(input_path)

    required_columns = [
        "delta_score",
        "attack_present",
        "baseline_score",
        "primary_moderator_z",
        "attack_x_primary_moderator",
        "post_score",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required SEM columns: {missing}")

    sem_result = _fit_sem(df, primary_moderator_label=_pretty_moderator_label(config.primary_moderator))
    ols_model = smf.ols(PRIMARY_OLS_FORMULA, data=df).fit(cov_type="HC3")

    ols_table = pd.DataFrame(
        {
            "term": ols_model.params.index,
            "estimate": ols_model.params.values,
            "std_error": ols_model.bse.values,
            "p_value": ols_model.pvalues.values,
            "conf_low": ols_model.conf_int()[0].values,
            "conf_high": ols_model.conf_int()[1].values,
        }
    )
    bootstrap_table = _bootstrap_primary_model(
        df=df,
        n_bootstrap=config.bootstrap_samples,
        seed=config.seed,
        terms=list(ols_model.params.index),
    )

    exploratory_columns = available_moderator_columns(
        df,
        preferred_order=[
            config.primary_moderator,
            "profile_cont_resilience_index",
            "profile_cont_big_five_neuroticism_mean_pct",
            "profile_cont_big_five_openness_to_experience_mean_pct",
            "profile_cont_age_years",
            "profile_cat__profile_cat_sex_Female",
            "profile_cat__profile_cat_sex_Male",
            "profile_cat__profile_cat_sex_Other",
        ],
    )
    exploratory_table = _fit_exploratory_moderator_models(
        df=df,
        moderator_columns=exploratory_columns,
        primary_moderator=config.primary_moderator,
    )

    out = Path(output_dir)
    spec_txt = out / "sem_model_spec.txt"
    sem_json = out / "sem_result.json"
    sem_coeff_csv = out / "sem_coefficients.csv"
    sem_fit_json = out / "sem_fit_indices.json"
    ols_txt = out / "ols_robust_summary.txt"
    ols_params_csv = out / "ols_robust_params.csv"
    bootstrap_csv = out / "bootstrap_primary_params.csv"
    exploratory_csv = out / "exploratory_moderator_comparison.csv"
    report_txt = out / "moderation_report.txt"
    assumptions_json = out / "assumption_register.json"
    critiques_json = out / "peer_review_critiques.json"
    methodology_txt = out / "methodology_audit.txt"

    write_text(spec_txt, sem_result.model_formula)
    write_json(sem_json, sem_result.model_dump())
    pd.DataFrame([coeff.model_dump() for coeff in sem_result.coefficients]).to_csv(sem_coeff_csv, index=False)
    write_json(sem_fit_json, sem_result.fit_indices)
    write_text(ols_txt, str(ols_model.summary()))
    ols_table.to_csv(ols_params_csv, index=False)
    bootstrap_table.to_csv(bootstrap_csv, index=False)
    exploratory_table.to_csv(exploratory_csv, index=False)

    write_text(
        report_txt,
        _render_report(
            df=df,
            sem_result=sem_result,
            ols_summary=str(ols_model.summary()),
            ols_table=ols_table,
            bootstrap_table=bootstrap_table,
            exploratory_table=exploratory_table,
            primary_moderator=config.primary_moderator,
            run_id=config.run_id,
        ),
    )

    assumptions = build_assumption_register(df, sem_result)
    critiques = build_peer_review_critique_notes(df, sem_result)
    write_json(assumptions_json, assumptions)
    write_json(critiques_json, critiques)
    write_text(methodology_txt, render_methodology_audit_text(assumptions, critiques))

    manifest = StageArtifactManifest(
        stage_id="06",
        stage_name="construct_structural_equation_model",
        input_path=abs_path(input_path),
        primary_output_path=abs_path(report_txt),
        output_files=[
            abs_path(spec_txt),
            abs_path(sem_json),
            abs_path(sem_coeff_csv),
            abs_path(sem_fit_json),
            abs_path(ols_txt),
            abs_path(ols_params_csv),
            abs_path(bootstrap_csv),
            abs_path(exploratory_csv),
            abs_path(report_txt),
            abs_path(assumptions_json),
            abs_path(critiques_json),
            abs_path(methodology_txt),
        ],
        record_count=len(df),
        metadata={
            "sem_converged": sem_result.converged,
            "n_coefficients": len(sem_result.coefficients),
            "primary_moderator": config.primary_moderator,
            "bootstrap_samples": config.bootstrap_samples,
        },
    )

    write_json(stage_manifest_path(output_dir), manifest.model_dump())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 06 - SEM construction")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="run_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-moderator", default="profile_cont_susceptibility_index")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)

    config = Stage06Config(
        stage_name="construct_structural_equation_model",
        run_id=args.run_id,
        seed=args.seed,
        primary_moderator=args.primary_moderator,
        bootstrap_samples=args.bootstrap_samples,
    )

    manifest = run_stage(args.input_path, args.output_dir, config)
    LOGGER.info("Stage 06 completed: %s records", manifest.record_count)


if __name__ == "__main__":
    main()
