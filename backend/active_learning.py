import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Scikit-Learn imports for the Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C


class GaussianProcessSurrogate:
    def __init__(self, history_df: pd.DataFrame):
        self.history_df = history_df

        # 7-Dimensional feature space
        self.metafeature_cols = ["ttr_score", "hapax_ratio", "zipf_alpha"]
        self.feature_cols = ["lora_r", "learning_rate", "lora_dropout", "batch_size"]
        self.all_cols = self.metafeature_cols + self.feature_cols

        # Dynamically calculate bounds for metafeatures based on historical extremes
        meta_mins = [self.history_df[c].min() for c in self.metafeature_cols]
        meta_maxs = [self.history_df[c].max() for c in self.metafeature_cols]

        # Prevent zero-width bounds just in case history is uniform
        for i in range(len(meta_mins)):
            if meta_mins[i] == meta_maxs[i]:
                meta_maxs[i] += 1e-4

        # 7D Ground Truth bounds for normalization
        self.bounds_real = np.array(
            [
                meta_mins + [4.0, 1e-5, 0.0, 2],  # min values (Meta + Hypers)
                meta_maxs + [64.0, 0.003, 0.2, 18],  # max values (Meta + Hypers)
            ],
            dtype=np.float64,
        )

    def _normalize(self, x_real):
        """Scales inputs to [0, 1] for better GP stability"""
        return (x_real - self.bounds_real[0]) / (
            self.bounds_real[1] - self.bounds_real[0]
        )

    def fit(self):
        """Trains the Gaussian Process on the historical runs."""
        self.train_x_real = self.history_df[self.all_cols].values
        self.train_y = self.history_df[
            "val_loss"
        ].values  # Predicting raw validation loss

        # Normalize the 7D input space
        self.train_x_norm = self._normalize(self.train_x_real)

        # Define and fit the Matern Kernel GP
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            # alpha=1e-3,
            n_restarts_optimizer=10,
            random_state=42,
            normalize_y=True,
        )
        self.gp.fit(self.train_x_norm, self.train_y)

        print("✅ Gaussian Process Surrogate fitted successfully.")

    def plot_1d_surrogates(
        self, current_metafeatures: dict, n_samples=20, file_prefix="gp_surrogate"
    ):
        """
        Plots a marginal slice of the GP for each hyperparameter using N random samples,
        while locking the metafeatures to the specific target dataset context.
        """
        if not hasattr(self, "gp"):
            print("Run .fit() first to train the GP.")
            return

        # prepare random samples withing the bounds
        samples = []

        for _ in range(n_samples):
            sample = []
            # Add the fixed metafeatures for the target dataset
            for col in self.metafeature_cols:
                sample.append(current_metafeatures[col])

            # Add random values for the hyperparameters within their bounds
            for i, col in enumerate(self.feature_cols):

                low = self.bounds_real[0][len(self.metafeature_cols) + i]
                high = self.bounds_real[1][len(self.metafeature_cols) + i]
                if i == 0:  # lora_r and batch_size should be integers
                    sample.append(
                        random.choice([4, 8, 16, 32, 64])
                    )  # +1 to include the upper bound
                elif i == 3:  # batch_size should be integers
                    sample.append(random.choice([4, 8, 16]))
                else:
                    sample.append(np.random.uniform(low, high))

            samples.append(sample)

        samples = np.array(samples)
        samples_norm = self._normalize(samples)

        means, stds = self.gp.predict(samples_norm, return_std=True)

        # output the 5 best samples according to the upper confidence bound (mean - 1.96*std)
        ucb = means - 1.96 * stds
        best_indices = np.argsort(ucb)[:5]
        print("Top 5 recommended hyperparameter configurations:")
        for idx in best_indices:
            config = {
                col: samples[idx][len(self.metafeature_cols) + i]
                for i, col in enumerate(self.feature_cols)
            }
            print(
                f"Config: {config}, Predicted Loss: {means[idx]:.4f}, UCB: {ucb[idx]:.4f}"
            )

        print("5 worst recommended hyperparameter configurations:")

        worst_indices = np.argsort(ucb)[-5:]
        for idx in worst_indices:
            config = {
                col: samples[idx][len(self.metafeature_cols) + i]
                for i, col in enumerate(self.feature_cols)
            }
            print(
                f"Config: {config}, Predicted Loss: {means[idx]:.4f}, UCB: {ucb[idx]:.4f}"
            )

        # Plotting each hyperparameter's surrogate slice
        for i, col in enumerate(self.feature_cols):
            plt.figure(figsize=(8, 5))
            plt.scatter(
                samples[:, len(self.metafeature_cols) + i],
                means,
                alpha=0.6,
                label="GP Predicted Loss",
            )

            plt.title(f"GP Surrogate Slice for {col}", fontsize=14)
            plt.xlabel(col)
            plt.ylabel("Predicted Validation Loss")
            plt.ylim(2.5, 3.5)
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{file_prefix}_{col}.png")
            plt.show()


if __name__ == "__main__":
    try:
        mock_history = pd.read_csv("nanoGPT.csv")
    except FileNotFoundError:
        print("Please ensure 'nanoGPT.csv' exists in the directory.")
        exit()

    test_df = mock_history[mock_history["dataset_id"] == "roneneldan/TinyStories"]
    train_df = mock_history[mock_history["dataset_id"] == "roneneldan/TinyStories"]

    # Initialize and fit the Surrogate
    surrogate = GaussianProcessSurrogate(train_df)
    surrogate.fit()

    # Generate the plots for the target context
    if not test_df.empty:
        current_metafeatures = test_df.iloc[0].to_dict()
        # You can adjust n_samples higher for a denser "cloud"
        surrogate.plot_1d_surrogates(
            current_metafeatures, n_samples=50, file_prefix="gp_surrogate"
        )
    else:
        print("Test dataset is empty, cannot generate contextual plots.")
