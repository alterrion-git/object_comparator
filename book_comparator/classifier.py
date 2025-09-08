import random
import cv2
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .ensemble_comparator import EnsembleBookComparator
from .orb_comparator import ORBBookComparator
from .gemini_comparator import GeminiBookComparator


def classifier(
    info: Dict,
    n_samples: int = 10,
    prompt_template=None,
    threshold: float = 0.3
) -> Tuple[pd.DataFrame, Dict]:
    records = []
    
    comparator = EnsembleBookComparator(threshold=threshold, prompt_template=prompt_template) # Can change this to different classifiers
    
    pos_pairs = random.sample(info["positive_images"], min(n_samples, len(info["positive_images"])))
    neg_pairs = random.sample(info["negative_images"], min(n_samples, len(info["negative_images"])))

    for pairs, label in [(pos_pairs, 1), (neg_pairs, 0)]:
        for img1_path, img2_path in pairs:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            scores = comparator.compare_images(img1, img2)
            
            record = {
                "image1": str(img1_path),
                "image2": str(img2_path),
                "actual_label": label
            }

            if isinstance(scores, tuple):
                score_orb, score_gemini = scores
                record.update({
                    "score_orb": score_orb,
                    "score_gemini": score_gemini
                })
                record["predicted_label"] = 1 if (score_gemini == 1 and score_orb >= threshold) else 0
            else:
                score = scores
                if isinstance(comparator, ORBBookComparator):
                    record["score_orb"] = score
                    record["predicted_label"] = 1 if score >= threshold else 0
                else:
                    record["score_gemini"] = score
                    record["predicted_label"] = score
            
            records.append(record)

    df = pd.DataFrame(records)

    results = {
        "threshold": threshold,
        "accuracy": accuracy_score(df["actual_label"], df["predicted_label"]),
        "precision": precision_score(df["actual_label"], df["predicted_label"], zero_division=0),
        "recall": recall_score(df["actual_label"], df["predicted_label"], zero_division=0),
        "f1": f1_score(df["actual_label"], df["predicted_label"], zero_division=0),
    }

    return df, results