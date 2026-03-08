import numpy as np


class LinearClassifier:
    """Simple linear classifier using NumPy."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        """
        Initialize weights with small random values.

        Args:
            input_dim (int): Number of input features (D)
            num_classes (int): Number of output classes (C)
        """
        self.W = 0.001 * np.random.randn(input_dim, num_classes)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw class scores.

        Args:
            X (np.ndarray): Input data of shape (N, D)

        Returns:
            np.ndarray: Class scores of shape (N, C)
        """
        return X.dot(self.W)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input data.

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Predicted labels (N,)
        """
        scores = self.forward(X)
        return np.argmax(scores, axis=1)