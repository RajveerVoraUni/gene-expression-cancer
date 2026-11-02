import numpy as np
from itertools import combinations
from typing import List, Set, Dict


# calculate jaccard overlap between two feature sets
def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    union = len(set_a.union(set_b))
    if union == 0:
        return 1.0
    return len(set_a.intersection(set_b)) / union


# calculate dice sorensen similarity between two feature sets
def dice_similarity(set_a: Set, set_b: Set) -> float:
    total = len(set_a) + len(set_b)
    if total == 0:
        return 1.0
    return 2.0 * len(set_a.intersection(set_b)) / total


# calculate mean pairwise stability across all fold combinations
def compute_stability_metrics(feature_sets: List[Set]) -> Dict[str, float]:
    if len(feature_sets) < 2:
        return {"mean_jaccard": 1.0, "mean_dice": 1.0, "std_jaccard": 0.0}

    pairs = list(combinations(range(len(feature_sets)), 2))
    jaccards = [jaccard_similarity(feature_sets[i], feature_sets[j]) for i, j in pairs]
    dices = [dice_similarity(feature_sets[i], feature_sets[j]) for i, j in pairs]

    return {
        "mean_jaccard": float(np.mean(jaccards)),
        "std_jaccard": float(np.std(jaccards)),
        "mean_dice": float(np.mean(dices)),
        "std_dice": float(np.std(dices)),
        "n_pairs": len(pairs),
    }


# rank features by selection count across splits
def compute_selection_frequencies(feature_sets: List[Set], all_feature_names: List[str] = None):
    counts = {}
    for f_set in feature_sets:
        for f in f_set:
            counts[f] = counts.get(f, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts
