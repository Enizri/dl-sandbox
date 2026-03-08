import numpy as np


def softmax_loss(scores: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute softmax cross-entropy loss and gradient.

    Args:
        scores (np.ndarray): Raw scores (N, C)
        y (np.ndarray): True labels (N,)

    Returns:
        loss (float): Cross-entropy loss
        dscores (np.ndarray): Gradient w.r.t scores (N, C)
    """
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    N = scores.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N

    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N

    return loss, dscores