import pandas as pd
from pathlib import Path
from time import time
from dotenv import load_dotenv
from book_comparator.config import get_settings, logger
from book_comparator.utils import load_pairs
from book_comparator.classifier import classifier

dotenv_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path)

class ExperimentRunner:
    def __init__(self):
        self.settings = get_settings()
        self.info = load_pairs(self.settings)
        self.results_dir = self.settings.RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        start_time = time()

        logger.info("=== Book Comparator Experiment ===")
        logger.info(f"Positive pairs loaded: {len(self.info['positive_images'])}")
        logger.info(f"Negative pairs loaded: {len(self.info['negative_images'])}")

        logger.info(f"\nRunning classifier on {self.settings.DEFAULT_N_SAMPLES} samples...")
        df, metrics = classifier(
            self.info,
            n_samples=self.settings.DEFAULT_N_SAMPLES,
            threshold=self.settings.DEFAULT_THRESHOLD
        )

        csv_path = self.results_dir / "experiment_results.csv"
        csv_path = self.results_dir / "experiment_results.csv"
        columns = ["image1", "image2", "actual_label", "predicted_label"]
        if "score_gemini" in df.columns:
            columns.append("score_gemini")
        if "score_orb" in df.columns:
            columns.append("score_orb")
        df[columns].to_csv(csv_path, index=False)
        logger.info(f"\nResults saved to: {csv_path}")
        logger.info("\n=== Experiment Metrics ===")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")

        end_time = time()
        duration = end_time - start_time
        logger.info(f"\nTotal execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")

if __name__ == "__main__":
    evaluate = ExperimentRunner()
    evaluate.run()