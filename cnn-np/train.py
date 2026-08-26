import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "linear_classifier"))
from utils import load_cifar10  # noqa: E402

from model import SimpleCNN
from loss import softmax_loss


def train(
    learning_rate: float = 1e-2,
    num_epochs: int = 5,
    batch_size: int = 64,
    train_batches: list = [1],
) -> SimpleCNN:
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(train_batches=train_batches)

    # (N, 3072) -> (N, 3, 32, 32)
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_val = X_val.reshape(-1, 3, 32, 32)

    model = SimpleCNN(num_classes=10)
    num_train = X_train.shape[0]

    for epoch in range(num_epochs):
        perm = np.random.permutation(num_train)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_train, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            scores = model.forward(xb)
            loss, dscores = softmax_loss(scores, yb)
            model.backward(dscores)

            for param, grad in model.params_and_grads():
                param -= learning_rate * grad

            epoch_loss += loss
            num_batches += 1

        val_scores = model.forward(X_val[:200])  # subsample for a quick check
        val_acc = np.mean(np.argmax(val_scores, axis=1) == y_val[:200])
        print(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {epoch_loss / num_batches:.4f} | Val Acc (sample): {val_acc:.4f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train(learning_rate=args.lr, num_epochs=args.epochs, batch_size=args.batch_size)
