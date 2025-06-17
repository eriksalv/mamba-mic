import numpy as np


def kfold(data: list, k: int, shuffle: bool = True, seed: int = 42):
    data = np.array(data)
    if shuffle:
        data = np.random.default_rng(seed=seed).permutation(data)

    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size : (i + 1) * fold_size]
        train_indices = np.concatenate(
            [indices[: i * fold_size], indices[(i + 1) * fold_size :]]
        )
        folds.append((train_indices, test_indices))
    return [
        (data[train_indices].tolist(), data[test_indices].tolist())
        for train_indices, test_indices in folds
    ]