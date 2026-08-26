# cnn-np

A small CNN implemented from scratch with NumPy, trained on CIFAR-10.

- `layers.py` — `Conv2D` (im2col-based), `ReLU`, `MaxPool2D`, `Flatten`, `Dense`, each with forward/backward.
- `model.py` — `SimpleCNN`: Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Dense.
- `loss.py` — softmax cross-entropy loss and gradient.
- `train.py` — training loop (mini-batch SGD) on CIFAR-10.
- `two_layers_demo.ipynb` — earlier 2-layer NumPy net on an XOR toy problem, kept as the starting point this grew out of.

```bash
python train.py --epochs 5 --batch_size 64 --lr 1e-2
```
