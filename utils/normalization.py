import numpy as np

def normalize_scores(scores):
    if len(scores) == 0:
        return np.array([])
    min_score, max_score = np.min(scores), np.max(scores)
    if max_score == min_score:
        return np.ones(len(scores)) * 50
    return ((scores - min_score) / (max_score - min_score)) * 90 + 10
