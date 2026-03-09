import numpy as np
from model import LinearClassifier
from loss import softmax_loss
from utils import load_cifar10
import argparse
from colorama import init, Fore, Style




def train_linear_classifier(
    learning_rate: float = 1e-2,
    num_epochs: int = 10,
    batch_size: int = 100,
    train_batches: list = [1],
    optimizer: str = "sgd",
    l2reg: float = 0.0,
    lr_schedule: str = None,
) -> tuple[LinearClassifier, list[float]]:
    
    """Train linear classifier on CIFAR-10 with optimizer and L2 regularization options."""
    X_train, y_train, X_val, y_val, X_test, Y_test = load_cifar10(train_batches=train_batches)
    input_dim = X_train.shape[1]
    num_classes = 10
    print(Style.BRIGHT + Fore.CYAN + f"Starting training using batches: {train_batches}, optimizer: {optimizer}, l2reg: {l2reg}, lr_schedule: {lr_schedule}" + Style.RESET_ALL)

    model = LinearClassifier(input_dim, num_classes)
    train_loss_history = []
    # Initialize model and optimizer state momentum 
    v = np.zeros_like(model.W)  # For Adam/AdamW
    m = np.zeros_like(model.W)  # For Adam/AdamW
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    prev_loss = None
    for epoch in range(num_epochs):
        # Cosine annealing learning rate
        if lr_schedule == "cosine":
            lr = learning_rate * 0.5 * (1 + np.cos(np.pi * epoch / num_epochs))
        else:
            lr = learning_rate
        #train on random batches each epoch  
        perm = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[perm[i : i + batch_size]]
            y_batch = y_train[perm[i : i + batch_size]]

            scores = model.forward(X_batch)
            loss, grad = softmax_loss(scores, y_batch)
            # Add L2 regularization (for sgd/adam) or weight decay (for adamw)
            if optimizer == "adamw":
                train_loss_history.append(loss)  # AdamW: no loss penalty, only decoupled weight decay
            else:
                if l2reg > 0:
                    loss += 0.5 * l2reg * np.sum(model.W * model.W)
                    grad += l2reg * model.W
                train_loss_history.append(loss)

            # Optimizer update options for sgd, adam, and adamw from bash args
            g = X_batch.T.dot(grad)
            if optimizer == "sgd":
                model.W -= lr * g
            elif optimizer == "adam":
                t += 1
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g * g)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                model.W -= lr * m_hat / (np.sqrt(v_hat) + eps)
            elif optimizer == "adamw":
                t += 1
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g * g)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                # AdamW: decoupled weight decay
                model.W -= lr * m_hat / (np.sqrt(v_hat) + eps)
                if l2reg > 0:
                    model.W -= lr * l2reg * model.W
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

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
    #reset colorama for colorful terminal output
    init(autoreset=True)
    parser = argparse.ArgumentParser(description="Train a linear classifier on CIFAR-10.")
    parser.add_argument('-b', '--train-batches', nargs='+', type=int, default=[1], help='List of training batch numbers (1-5)')
    parser.add_argument('-o', '--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'], help='Optimizer to use (sgd, adam, or adamw)')
    parser.add_argument('-l', '--lr-schedule', type=str, default=None, choices=[None, 'cosine'], help='Learning rate schedule (None or cosine)')
    parser.add_argument('-r', '--l2reg', type=float, default=0.0, help='L2 regularization strength')
    args = parser.parse_args()

    model, loss_history = train_linear_classifier(
        train_batches=args.train_batches,
        optimizer=args.optimizer,
        l2reg=args.l2reg,
        lr_schedule=args.lr_schedule
    )
    # Optionally save loss history for notebook
    np.save("results/train_loss.npy", np.array(loss_history))
    # Save model weights
    np.save("results/model_weights.npy", model.W)
    print(Style.BRIGHT + Fore.CYAN + "Training complete. Model weights saved to results/model_weights.npy" + Style.RESET_ALL)