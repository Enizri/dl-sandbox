# linear_classifier

A linear image classifier trained on CIFAR-10, implemented from scratch with NumPy (no autograd frameworks).

- `model.py` — `LinearClassifier`: weights `W` and bias `b`, forward pass produces class scores.
- `loss.py` — softmax cross-entropy loss and its gradient.
- `train.py` — training loop: mini-batch SGD, optional AdamW-style momentum, cosine learning rate schedule, CLI args for experimenting with hyperparameters.
- `utils.py` — CIFAR-10 loading/preprocessing helpers.
- `linear_demo.ipynb` — step-by-step walkthrough of training and inspecting the model, including a look at why a purely linear model struggles on CIFAR-10.
- `classifier_demo.ipynb` — end-to-end training run using `train.py` plus visualization of the learned per-class weight templates.

Run training directly:

```bash
python train.py --optimizer adamw --lr_schedule cosine
```
