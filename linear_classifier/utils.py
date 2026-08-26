import pickle
import numpy as np
import os

def load_cifar10(
    path: str = None,
    train_batches: list = [1],
    val_split: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 dataset with selectable training batches and validation split.

    Args:
        path (str): Path to CIFAR-10 batches folder
        train_batches (list): List of batch numbers to use for training (1-5)
        val_split (float): Fraction of training data to use for validation

    Returns:
        X_train, y_train, X_val, y_val, X_test, Y_test
    """


    # Set default path to datasets/cifar-10-batches-py/ relative to this file
    if path is None:
        # Find project root and use datasets/cifar-10-batches-py relative to it
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "datasets", "cifar-10-batches-py")
        path = os.path.normpath(path) + os.sep
    
    def load_batch(file_path: str) -> tuple[np.ndarray, np.ndarray]:
        with open(file_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            X = batch[b"data"]
            y = np.array(batch[b"labels"])
            return X, y

    # Load and concatenate selected training batches
    X_list, y_list = [], []
    for batch_num in train_batches:
        Xb, yb = load_batch(f"{path}data_batch_{batch_num}")
        X_list.append(Xb)
        y_list.append(yb)
    X_full = np.concatenate(X_list, axis=0)
    y_full = np.concatenate(y_list, axis=0)

    # Split into train/val
    num_train = int((1 - val_split) * X_full.shape[0])
    X_train = X_full[:num_train]
    y_train = y_full[:num_train]
    X_val = X_full[num_train:]
    y_val = y_full[num_train:]

    # Load test batch
    X_test, Y_test = load_batch(path + "test_batch")

    # Normalize pixels
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_val, y_val, X_test, Y_test