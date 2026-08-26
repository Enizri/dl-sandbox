from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense


class SimpleCNN:
    """
    A small CNN for 32x32x3 images (CIFAR-10 shaped):

        Conv(3->8, 3x3) -> ReLU -> MaxPool(2x2)
        Conv(8->16, 3x3) -> ReLU -> MaxPool(2x2)
        Flatten -> Dense(16*8*8 -> num_classes)
    """

    def __init__(self, num_classes=10, in_channels=3, input_hw=32):
        self.layers = [
            Conv2D(in_channels, 8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(size=2, stride=2),
            Conv2D(8, 16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(size=2, stride=2),
            Flatten(),
        ]
        flat_dim = 16 * (input_hw // 4) * (input_hw // 4)
        self.layers.append(Dense(flat_dim, num_classes))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params_and_grads(self):
        pg = []
        for layer in self.layers:
            pg.extend(layer.params_and_grads())
        return pg
