import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.spatial import distance
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


META_COLS = ["ttr_score", "hapax_ratio", "zipf_alpha"]
HP_COLS = ["lora_r", "learning_rate", "lora_dropout"]
REQUIRED_COLS = ["dataset_slug", "status", "val_loss"] + META_COLS + HP_COLS[:2]

BOUNDS_REAL = torch.tensor([[4.0, -5.0, 0.0], [64.0, -3.0, 0.3]], dtype=torch.float64)
BOUNDS_REAL_NP = BOUNDS_REAL.numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-dataset-out GP holdout evaluation for hyperparameter recommendation."
    )
    parser.add_argument(
        "--runs-csv",
        type=str,
        default="backend/runs.csv",
        help="Path to runs.csv (default: backend/runs.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="backend/holdout_gp_eval_outputs",
        help="Output directory for metrics, tables, and plots.",
    )
    parser.add_argument(
        "--n-suggestions",
        type=int,
        default=3,
        help="Number of GP suggestions per fold.",
    )
    parser.add_argument(
        "--neighbor-k",
        type=int,
        default=10,
        help="Number of nearest metafeature neighbors used for GP training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _normalize_tensor(x_real: torch.Tensor) -> torch.Tensor:
    return (x_real - BOUNDS_REAL[0]) / (BOUNDS_REAL[1] - BOUNDS_REAL[0])


def _unnormalize_tensor(x_norm: torch.Tensor) -> torch.Tensor:
    return x_norm * (BOUNDS_REAL[1] - BOUNDS_REAL[0]) + BOUNDS_REAL[0]


def _normalize_np(x_real: np.ndarray) -> np.ndarray:
    return (x_real - BOUNDS_REAL_NP[0]) / (BOUNDS_REAL_NP[1] - BOUNDS_REAL_NP[0])


def to_json_value(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def load_and_prepare_dataframe(runs_csv: Path) -> pd.DataFrame:
    if not runs_csv.exists():
        raise FileNotFoundError(f"Could not find runs CSV: {runs_csv}")

    df = pd.read_csv(runs_csv)

    if "lora_dropout" not in df.columns:
        if "dropout" in df.columns:
            df = df.copy()
            df["lora_dropout"] = df["dropout"]
            print("Info: using fallback column 'dropout' -> 'lora_dropout'.")
        else:
            raise ValueError("Missing required column: lora_dropout (or fallback dropout).")

    missing = [col for col in REQUIRED_COLS + ["lora_dropout"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise ValueError("No rows left after filtering status == 'ok'.")

    numeric_cols = ["val_loss"] + META_COLS + HP_COLS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=numeric_cols)
    dropped_nan = before - len(df)

    before_lr = len(df)
    df = df[df["learning_rate"] > 0].copy()
    dropped_lr = before_lr - len(df)

    if dropped_nan > 0:
        print(f"Info: dropped {dropped_nan} rows with invalid numeric values.")
    if dropped_lr > 0:
        print(f"Info: dropped {dropped_lr} rows with non-positive learning_rate.")
    if df.empty:
        raise ValueError("No rows left after numeric validation.")

    return df


class HoldoutBotorchOptimizer:
    def __init__(self, history_df: pd.DataFrame, neighbor_k: int, seed: int):
        self.history_df = history_df.copy()
        self.neighbor_k = max(1, int(neighbor_k))
        self.seed = seed
        self.meta_scaler = StandardScaler().fit(self.history_df[META_COLS])

    def _select_neighbors(self, current_metafeatures: Dict[str, float]) -> pd.DataFrame:
        current_mf_array = self.meta_scaler.transform([[current_metafeatures[c] for c in META_COLS]])
        hist_mf_array = self.meta_scaler.transform(self.history_df[META_COLS])
        dists = distance.cdist(current_mf_array, hist_mf_array, metric="euclidean")[0]

        ranked = self.history_df.copy()
        ranked["distance"] = dists
        similar_runs = ranked.sort_values(by=["distance", "val_loss"], ascending=[True, True]).head(self.neighbor_k)
        return similar_runs.reset_index(drop=True)

    def _fit_gp(self, similar_runs: pd.DataFrame) -> None:
        train_x_real = torch.tensor(
            similar_runs[["lora_r", "learning_rate", "lora_dropout"]].values,
            dtype=torch.float64,
        )
        train_x_real[:, 1] = torch.log10(train_x_real[:, 1])

        train_x = _normalize_tensor(train_x_real)
        train_y = torch.tensor(-similar_runs["val_loss"].values, dtype=torch.float64).unsqueeze(-1)

        unique_points = np.unique(train_x.detach().numpy(), axis=0).shape[0]
        if unique_points < 3:
            raise RuntimeError(
                f"Insufficient unique points ({unique_points}) for stable GP fit."
            )

        self.train_x = train_x
        self.train_y = train_y
        self.similar_runs = similar_runs

        self.gp_model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        fit_gpytorch_mll(mll)

    def recommend_next_configs(
        self, current_metafeatures: Dict[str, float], n_suggestions: int
    ) -> pd.DataFrame:
        if n_suggestions < 1:
            raise ValueError("n_suggestions must be >= 1")

        similar_runs = self._select_neighbors(current_metafeatures)
        self._fit_gp(similar_runs)

        best_f = self.train_y.max()
        q_ei = qExpectedImprovement(model=self.gp_model, best_f=best_f)
        standard_bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)

        torch.manual_seed(self.seed)
        candidates_norm, _ = optimize_acqf(
            acq_function=q_ei,
            bounds=standard_bounds,
            q=n_suggestions,
            num_restarts=10,
            raw_samples=max(20, 10 * n_suggestions),
        )

        self.candidates_norm = candidates_norm
        candidates_real = _unnormalize_tensor(candidates_norm)

        rows: List[Dict[str, float]] = []
        for i, row in enumerate(candidates_real.detach().numpy(), start=1):
            lora_cont = float(row[0])
            lora_snapped = int(2 ** round(np.log2(max(lora_cont, 1e-9))))
            lora_r = max(4, min(64, lora_snapped))
            lr = float(10 ** row[1])
            dropout = float(row[2])

            rows.append(
                {
                    "suggestion_rank": i,
                    "lora_r": int(lora_r),
                    "learning_rate": float(lr),
                    "lora_dropout": float(dropout),
                    "learning_rate_display": round(lr, 6),
                    "lora_dropout_display": round(dropout, 3),
                }
            )

        return pd.DataFrame(rows)

    def plot_1d_surrogates(self, output_dir: Path) -> None:
        if not hasattr(self, "gp_model"):
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        best_idx = self.train_y.argmax().item()
        best_x_norm = self.train_x[best_idx]

        param_titles = ["LoRA r", "Learning Rate", "LoRA Dropout"]
        file_names = [
            "gp_surrogate_lora_r.png",
            "gp_surrogate_learning_rate.png",
            "gp_surrogate_lora_dropout.png",
        ]

        for i in range(3):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
            ax.grid(True, linestyle="--", alpha=0.35)

            test_x = best_x_norm.repeat(200, 1)
            grid_norm = torch.linspace(0, 1, 200, dtype=torch.float64)
            test_x[:, i] = grid_norm

            with torch.no_grad():
                post = self.gp_model.posterior(test_x)
                mean_loss = -post.mean.detach().cpu().numpy().flatten()
                std = np.sqrt(post.variance.detach().cpu().numpy().flatten())

            lower = mean_loss - 1.96 * std
            upper = mean_loss + 1.96 * std

            min_v = BOUNDS_REAL[0, i].item()
            max_v = BOUNDS_REAL[1, i].item()
            grid_real = grid_norm.detach().cpu().numpy() * (max_v - min_v) + min_v
            obs_x_real = self.train_x[:, i].detach().cpu().numpy() * (max_v - min_v) + min_v
            obs_y_loss = -self.train_y.detach().cpu().numpy().flatten()

            if i == 1:
                grid_plot = 10 ** grid_real
                obs_plot = 10 ** obs_x_real
                ax.set_xscale("log")
            else:
                grid_plot = grid_real
                obs_plot = obs_x_real

            ax.fill_between(grid_plot, lower, upper, alpha=0.25, label="95% CI")
            ax.plot(grid_plot, mean_loss, linewidth=2.2, label="GP mean")
            ax.scatter(obs_plot, obs_y_loss, s=40, alpha=0.9, label="Observed")

            if hasattr(self, "candidates_norm"):
                sugg_x_norm = self.candidates_norm[:, i]
                sugg_test = best_x_norm.repeat(self.candidates_norm.shape[0], 1)
                sugg_test[:, i] = sugg_x_norm
                with torch.no_grad():
                    sugg_post = self.gp_model.posterior(sugg_test)
                    sugg_loss = -sugg_post.mean.detach().cpu().numpy().flatten()

                sugg_x_real = sugg_x_norm.detach().cpu().numpy() * (max_v - min_v) + min_v
                if i == 1:
                    sugg_x_plot = 10 ** sugg_x_real
                else:
                    sugg_x_plot = sugg_x_real
                ax.scatter(sugg_x_plot, sugg_loss, marker="*", s=240, label="Suggestions")

            ax.set_title(f"GP Surrogate: {param_titles[i]}")
            ax.set_xlabel(param_titles[i])
            ax.set_ylabel("Validation Loss")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(output_dir / file_names[i], bbox_inches="tight")
            plt.close(fig)


def evaluate_holdout_predictions(
    optimizer: HoldoutBotorchOptimizer,
    heldout_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    heldout_x_real = torch.tensor(
        heldout_df[["lora_r", "learning_rate", "lora_dropout"]].values,
        dtype=torch.float64,
    )
    heldout_x_real[:, 1] = torch.log10(heldout_x_real[:, 1])
    heldout_x_norm = _normalize_tensor(heldout_x_real)

    with torch.no_grad():
        post = optimizer.gp_model.posterior(heldout_x_norm)
        pred_loss = -post.mean.detach().cpu().numpy().flatten()
        pred_std = np.sqrt(post.variance.detach().cpu().numpy().flatten())

    pred_df = heldout_df.copy().reset_index(drop=True)
    pred_df["pred_val_loss"] = pred_loss
    pred_df["pred_std"] = pred_std
    pred_df["residual"] = pred_df["pred_val_loss"] - pred_df["val_loss"]

    actual = pred_df["val_loss"].to_numpy()
    pred = pred_df["pred_val_loss"].to_numpy()
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    rho = spearmanr(pred, actual).correlation
    rho_value = None if rho is None or np.isnan(rho) else float(rho)

    ranked = pred_df.sort_values("pred_val_loss", ascending=True).reset_index(drop=True)
    topk_curve = ranked["val_loss"].cummin().to_numpy()
    best_actual = float(actual.min())

    topk_df = pd.DataFrame(
        {
            "k": np.arange(1, len(ranked) + 1),
            "best_actual_val_loss_so_far": topk_curve,
        }
    )

    def regret_at(k: int):
        if len(topk_curve) >= k:
            return float(topk_curve[k - 1] - best_actual)
        return None

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "spearman": rho_value,
        "best_actual_val_loss": best_actual,
        "regret_at_1": regret_at(1),
        "regret_at_3": regret_at(3),
        "regret_at_5": regret_at(5),
    }
    return pred_df, metrics, topk_df


def score_suggestions_against_holdout(
    suggestions_df: pd.DataFrame, heldout_df: pd.DataFrame
) -> pd.DataFrame:
    heldout_eval = heldout_df.copy().reset_index(drop=True)
    heldout_eval["actual_rank"] = heldout_eval["val_loss"].rank(method="min", ascending=True).astype(int)

    heldout_real = np.column_stack(
        [
            heldout_eval["lora_r"].to_numpy(),
            np.log10(heldout_eval["learning_rate"].to_numpy()),
            heldout_eval["lora_dropout"].to_numpy(),
        ]
    )
    heldout_norm = _normalize_np(heldout_real)

    best_loss = float(heldout_eval["val_loss"].min())
    matched_rows: List[Dict[str, object]] = []
    matched_indices: List[int] = []

    for _, row in suggestions_df.iterrows():
        s_real = np.array([row["lora_r"], math.log10(row["learning_rate"]), row["lora_dropout"]], dtype=float)
        s_norm = _normalize_np(s_real[None, :])[0]
        dists = np.linalg.norm(heldout_norm - s_norm, axis=1)
        nearest_idx = int(np.argmin(dists))
        matched_indices.append(nearest_idx)

        nearest = heldout_eval.iloc[nearest_idx]
        matched_rows.append(
            {
                "suggestion_rank": int(row["suggestion_rank"]),
                "matched_index": nearest_idx,
                "matched_run_id": nearest.get("run_id", ""),
                "matched_val_loss": float(nearest["val_loss"]),
                "matched_actual_rank": int(nearest["actual_rank"]),
                "distance_to_match": float(dists[nearest_idx]),
                "matched_regret_to_best": float(nearest["val_loss"] - best_loss),
            }
        )

    matched_df = pd.DataFrame(matched_rows)
    dup_counts = matched_df["matched_index"].value_counts()
    matched_df["duplicate_match"] = matched_df["matched_index"].map(dup_counts).fillna(0).astype(int) > 1
    return suggestions_df.merge(matched_df, on="suggestion_rank", how="left")


def plot_predicted_vs_actual(pred_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    x = pred_df["val_loss"].to_numpy()
    y = pred_df["pred_val_loss"].to_numpy()
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))

    ax.scatter(x, y, alpha=0.8, s=36)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="black", label="Ideal")
    ax.set_xlabel("Actual val_loss")
    ax.set_ylabel("Predicted val_loss")
    ax.set_title("Predicted vs Actual (Held-out)")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_residual_hist(pred_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    residuals = pred_df["residual"].to_numpy()
    ax.hist(residuals, bins=30, alpha=0.8, edgecolor="black")
    ax.axvline(0.0, linestyle="--", linewidth=1.5, color="black")
    ax.set_xlabel("Residual (predicted - actual)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution (Held-out)")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_topk_curve(topk_df: pd.DataFrame, best_actual: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.plot(topk_df["k"], topk_df["best_actual_val_loss_so_far"], linewidth=2.0, label="Best found by top-k")
    ax.axhline(best_actual, linestyle="--", linewidth=1.5, color="black", label="Global best")
    ax.set_xlabel("k (top predicted runs)")
    ax.set_ylabel("Best actual val_loss so far")
    ax.set_title("Top-k Retrieval Curve")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def print_fold_summary(
    dataset_slug: str,
    train_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    suggestions_scored: pd.DataFrame,
    metrics: Dict[str, object],
) -> None:
    def _fmt_float(v):
        if v is None:
            return "None"
        if isinstance(v, (float, np.floating)) and (math.isnan(v) or math.isinf(v)):
            return "None"
        return f"{float(v):.6f}"

    print(f"\n=== Fold: hold out '{dataset_slug}' ===")
    print(f"Train rows: {len(train_df)} | Held-out rows: {len(heldout_df)}")

    cols = [
        "suggestion_rank",
        "lora_r",
        "learning_rate_display",
        "lora_dropout_display",
        "matched_val_loss",
        "matched_actual_rank",
        "matched_regret_to_best",
        "duplicate_match",
    ]
    print("Suggestions (with nearest held-out match):")
    print(suggestions_scored[cols].to_string(index=False))

    print("Metrics:")
    print(
        (
            f"  MAE={_fmt_float(metrics['mae'])}, RMSE={_fmt_float(metrics['rmse'])}, "
            f"Spearman={metrics['spearman'] if metrics['spearman'] is not None else 'None'}, "
            f"Regret@1={_fmt_float(metrics['regret_at_1'])}, "
            f"Regret@3={_fmt_float(metrics['regret_at_3'])}, "
            f"Regret@5={_fmt_float(metrics['regret_at_5'])}"
        )
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    runs_csv = Path(args.runs_csv)
    if not runs_csv.exists():
        fallback = Path(__file__).resolve().parent / "runs.csv"
        if fallback.exists():
            runs_csv = fallback

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_dataframe(runs_csv)
    dataset_slugs = sorted(df["dataset_slug"].dropna().unique().tolist())
    if len(dataset_slugs) < 2:
        raise ValueError("Need at least 2 unique dataset_slug values for leave-one-out.")

    all_metrics: List[Dict[str, object]] = []
    generated_folds = 0

    for dataset_slug in dataset_slugs:
        fold_out_dir = out_dir / dataset_slug
        fold_out_dir.mkdir(parents=True, exist_ok=True)

        heldout_df = df[df["dataset_slug"] == dataset_slug].copy()
        train_df = df[df["dataset_slug"] != dataset_slug].copy()

        fold_metrics: Dict[str, object] = {
            "dataset_slug": dataset_slug,
            "status": "ok",
            "n_train_total": int(len(train_df)),
            "n_heldout": int(len(heldout_df)),
            "neighbor_k": int(min(args.neighbor_k, len(train_df))),
            "n_suggestions": int(args.n_suggestions),
        }

        try:
            metafeatures = {col: float(heldout_df.iloc[0][col]) for col in META_COLS}
            optimizer = HoldoutBotorchOptimizer(train_df, neighbor_k=args.neighbor_k, seed=args.seed)
            suggestions_df = optimizer.recommend_next_configs(metafeatures, n_suggestions=args.n_suggestions)

            pred_df, eval_metrics, topk_df = evaluate_holdout_predictions(optimizer, heldout_df)
            suggestions_scored = score_suggestions_against_holdout(suggestions_df, heldout_df)

            fold_metrics["n_neighbors_used"] = int(len(optimizer.similar_runs))
            fold_metrics.update(eval_metrics)

            # Suggestion quality summary
            fold_metrics["suggestion_best_matched_val_loss"] = float(
                suggestions_scored["matched_val_loss"].min()
            )
            fold_metrics["suggestion_best_matched_regret"] = float(
                suggestions_scored["matched_regret_to_best"].min()
            )
            fold_metrics["suggestion_mean_matched_regret"] = float(
                suggestions_scored["matched_regret_to_best"].mean()
            )

            # Persist tables
            suggestions_scored.to_csv(fold_out_dir / "suggestions.csv", index=False)
            pred_cols = [
                "run_id",
                "dataset_slug",
                "val_loss",
                "pred_val_loss",
                "pred_std",
                "residual",
                "lora_r",
                "learning_rate",
                "lora_dropout",
            ]
            existing_pred_cols = [c for c in pred_cols if c in pred_df.columns]
            pred_df[existing_pred_cols].to_csv(fold_out_dir / "heldout_predictions.csv", index=False)
            topk_df.to_csv(fold_out_dir / "topk_curve.csv", index=False)

            # Persist plots
            optimizer.plot_1d_surrogates(fold_out_dir)
            plot_predicted_vs_actual(pred_df, fold_out_dir / "predicted_vs_actual.png")
            plot_residual_hist(pred_df, fold_out_dir / "residuals_hist.png")
            plot_topk_curve(
                topk_df,
                best_actual=float(fold_metrics["best_actual_val_loss"]),
                output_path=fold_out_dir / "topk_retrieval_curve.png",
            )

            print_fold_summary(dataset_slug, train_df, heldout_df, suggestions_scored, fold_metrics)
            generated_folds += 1

        except Exception as exc:
            fold_metrics["status"] = "skipped"
            fold_metrics["skip_reason"] = str(exc)
            print(f"\n=== Fold: hold out '{dataset_slug}' skipped ===")
            print(f"Reason: {exc}")

        with (fold_out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump({k: to_json_value(v) for k, v in fold_metrics.items()}, f, indent=2)

        all_metrics.append(fold_metrics)

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    completed = summary_df[summary_df["status"] == "ok"].copy()
    aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs_csv": str(runs_csv),
        "n_folds_total": int(len(summary_df)),
        "n_folds_completed": int((summary_df["status"] == "ok").sum()),
        "n_folds_skipped": int((summary_df["status"] == "skipped").sum()),
        "mean_mae": to_json_value(completed["mae"].mean()) if not completed.empty else None,
        "mean_rmse": to_json_value(completed["rmse"].mean()) if not completed.empty else None,
        "mean_spearman": to_json_value(completed["spearman"].dropna().mean())
        if "spearman" in completed.columns and not completed["spearman"].dropna().empty
        else None,
        "mean_regret_at_1": to_json_value(completed["regret_at_1"].mean())
        if "regret_at_1" in completed.columns and not completed.empty
        else None,
        "mean_regret_at_3": to_json_value(completed["regret_at_3"].mean())
        if "regret_at_3" in completed.columns and not completed.empty
        else None,
        "mean_regret_at_5": to_json_value(completed["regret_at_5"].mean())
        if "regret_at_5" in completed.columns and not completed.empty
        else None,
        "folds": [{k: to_json_value(v) for k, v in m.items()} for m in all_metrics],
    }

    with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print("\n=== Cross-fold summary ===")
    display_cols = [
        c
        for c in [
            "dataset_slug",
            "status",
            "mae",
            "rmse",
            "spearman",
            "regret_at_1",
            "regret_at_3",
            "regret_at_5",
            "suggestion_best_matched_regret",
        ]
        if c in summary_df.columns
    ]
    if display_cols:
        print(summary_df[display_cols].to_string(index=False))
    print(f"\nCompleted folds: {generated_folds}/{len(dataset_slugs)}")
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
