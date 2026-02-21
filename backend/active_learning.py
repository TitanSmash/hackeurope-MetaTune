import pandas as pd
import optuna
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance


class MetaBayesianOptimizer:
    def __init__(self, history_df: pd.DataFrame):
        """
        Initializes the Knowledge Base.
        history_df must contain:
        - Metafeatures: ['ttr_score', 'hapax_ratio', 'zipf_alpha']
        - Hyperparameters: ['lora_r', 'learning_rate', 'dropout']
        - Target Metric: ['val_loss']
        """
        self.history_df = history_df
        self.metafeature_cols = ["ttr_score", "hapax_ratio", "zipf_alpha"]
        self.hp_cols = ["lora_r", "learning_rate", "dropout"]

        # Fit a scaler so metafeatures with different ranges (e.g., TTR vs Alpha)
        # don't skew the distance calculation
        self.scaler = StandardScaler()
        self.scaler.fit(self.history_df[self.metafeature_cols])

    def recommend_next_configs(self, current_metafeatures: dict, n_suggestions=3):
        # 1. SCALE META-FEATURES
        current_mf_array = self.scaler.transform(
            [
                [
                    current_metafeatures["ttr_score"],
                    current_metafeatures["hapax_ratio"],
                    current_metafeatures["zipf_alpha"],
                ]
            ]
        )
        hist_mf_array = self.scaler.transform(self.history_df[self.metafeature_cols])

        # 2. FIND SIMILAR DATASETS (Euclidean Distance)
        distances = distance.cdist(current_mf_array, hist_mf_array, metric="euclidean")[
            0
        ]
        self.history_df["distance"] = distances

        # Grab the top 5 historical runs that were both similar AND had low validation loss
        similar_runs = self.history_df.sort_values(by=["distance", "val_loss"]).head(5)

        # 3. INITIALIZE BAYESIAN OPTIMIZATION (TPE Sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Keep terminal clean
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler()
        )

        # 4. THE MAGIC: WARM START THE OPTIMIZER
        # We inject our DataFrame's historical successes directly into Optuna's brain
        for _, row in similar_runs.iterrows():
            study.enqueue_trial(
                {
                    "lora_r": int(row["lora_r"]),
                    "learning_rate": float(row["learning_rate"]),
                    "dropout": float(row["dropout"]),
                }
            )

        # 5. ASK FOR NEXT CANDIDATES
        # Optuna evaluates the enqueued trials and mathematically predicts the next best steps
        suggestions = []
        for _ in range(n_suggestions):
            trial = study.ask()

            # Define the search space dynamically for nanoGPT LoRA
            lora_r = trial.suggest_categorical("lora_r", [4, 8, 16, 32, 64])
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.3)

            suggestions.append(
                {
                    "lora_r": lora_r,
                    "learning_rate": round(lr, 6),
                    "dropout": round(dropout, 3),
                }
            )

        return suggestions


# --- Test the Engine ---
if __name__ == "__main__":
    # 1. Mock a DataFrame of historical training runs (Your Knowledge Base)
    mock_history = pd.DataFrame(
        [
            # Medical Data (High Diversity, High Hapax)
            {
                "ttr_score": 0.45,
                "hapax_ratio": 0.60,
                "zipf_alpha": 1.1,
                "lora_r": 32,
                "learning_rate": 5e-4,
                "dropout": 0.2,
                "val_loss": 1.2,
            },
            {
                "ttr_score": 0.42,
                "hapax_ratio": 0.55,
                "zipf_alpha": 1.05,
                "lora_r": 64,
                "learning_rate": 8e-4,
                "dropout": 0.25,
                "val_loss": 1.1,
            },
            # Code/Log Data (Highly Repetitive, Low Hapax)
            {
                "ttr_score": 0.05,
                "hapax_ratio": 0.10,
                "zipf_alpha": 1.8,
                "lora_r": 8,
                "learning_rate": 1e-4,
                "dropout": 0.05,
                "val_loss": 0.8,
            },
            {
                "ttr_score": 0.08,
                "hapax_ratio": 0.12,
                "zipf_alpha": 1.7,
                "lora_r": 4,
                "learning_rate": 5e-5,
                "dropout": 0.0,
                "val_loss": 0.9,
            },
        ]
    )

    bo_engine = MetaBayesianOptimizer(mock_history)

    # 2. A user uploads a NEW dataset (e.g., Legal Documents)
    # Our previous extraction script determines it is highly diverse:
    new_dataset_metafeatures = {
        "ttr_score": 0.40,
        "hapax_ratio": 0.50,
        "zipf_alpha": 1.08,
    }

    print("🧠 Bayesian Optimizer (Warm-Started) Candidate Suggestions:")
    candidates = bo_engine.recommend_next_configs(
        new_dataset_metafeatures, n_suggestions=3
    )

    for i, config in enumerate(candidates):
        print(f"Candidate {i+1}: {config}")
