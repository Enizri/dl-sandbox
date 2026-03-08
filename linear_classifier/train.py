import numpy as np
from model import LinearClassifier
from loss import softmax_loss
from utils import load_cifar10
import argparse
from colorama import init, Fore, Style



def train_linear_classifier(
    learning_rate: float = 1e-3,
    num_epochs: int = 10,
    batch_size: int = 100,
    train_batches: list = [1],
) -> tuple[LinearClassifier, list[float]]:
    """Train linear classifier on CIFAR-10."""
    X_train, y_train, X_val, y_val, X_test, Y_test = load_cifar10(train_batches=train_batches)
    input_dim = X_train.shape[1]
    num_classes = 10
    print(Style.BRIGHT + Fore.CYAN + f"Starting training using batches: {train_batches}" + Style.RESET_ALL)
    
    model = LinearClassifier(input_dim, num_classes)
    train_loss_history = []

    prev_loss = None
    for epoch in range(num_epochs):
        # Shuffle training data and create batches
        perm = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[perm[i : i + batch_size]]
            y_batch = y_train[perm[i : i + batch_size]]

            scores = model.forward(X_batch)
            loss, grad = softmax_loss(scores, y_batch)
            train_loss_history.append(loss)

            # Gradient descent update
            model.W -= learning_rate * X_batch.T.dot(grad)

        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        val_acc = np.mean(val_preds == y_val)
        test_acc = np.mean(test_preds == Y_test)

        # Colorful logging
        if epoch == 0:
            color = Fore.RED + Style.BRIGHT
        elif prev_loss is not None and loss < prev_loss:
            color = Fore.GREEN + Style.BRIGHT
        else:
            color = Fore.YELLOW + Style.BRIGHT
        print(color + f"Epoch {epoch+1}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}" + Style.RESET_ALL)
        prev_loss = loss

    return model, train_loss_history


if __name__ == "__main__":
    init(autoreset=True)
    parser = argparse.ArgumentParser(description="Train a linear classifier on CIFAR-10.")
    parser.add_argument('--train-batches', nargs='+', type=int, default=[1], help='List of training batch numbers (1-5)')
    args = parser.parse_args()

    model, loss_history = train_linear_classifier(train_batches=args.train_batches)
    # Optionally save loss history for notebook
    np.save("results/train_loss.npy", np.array(loss_history))
    print(Style.BRIGHT + Fore.CYAN + "Training complete." + Style.RESET_ALL)