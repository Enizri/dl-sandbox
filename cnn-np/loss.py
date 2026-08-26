import numpy as np


def softmax_loss(scores, y):
    """
    Softmax cross-entropy loss.

    Args:
        scores: (N, C) raw class scores
        y: (N,) integer class labels

    Returns:
        loss: scalar
        dscores: (N, C) gradient of loss w.r.t. scores
    """
    N = scores.shape[0]

    shifted = scores - np.max(scores, axis=1, keepdims=True)  # numerical stability
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    correct_logprobs = -np.log(probs[np.arange(N), y] + 1e-15)
    loss = np.sum(correct_logprobs) / N

    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N

    return loss, dscores
