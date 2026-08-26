"""
From-scratch, NumPy-only building blocks for a small CNN.

Every layer implements:
    forward(x)  -> out
    backward(dout) -> dx      (and stashes parameter grads on self.dW / self.db)

Conv2D and MaxPool2D use an im2col/col2im approach so the actual
convolution/pooling reduces to a single matrix multiply / max-reduction
instead of nested Python loops over every pixel.
"""

import numpy as np


def _im2col(x, kh, kw, stride, pad):
    """
    x: (N, C, H, W) -> columns: (C*kh*kw, N*out_h*out_w)
    """
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    cols = np.zeros((C, kh, kw, N, out_h, out_w), dtype=x.dtype)
    for y in range(kh):
        y_max = y + stride * out_h
        for xk in range(kw):
            x_max = xk + stride * out_w
            cols[:, y, xk, :, :, :] = x_padded[:, :, y:y_max:stride, xk:x_max:stride].transpose(1, 0, 2, 3)

    cols = cols.reshape(C * kh * kw, N * out_h * out_w)
    return cols, out_h, out_w


def _col2im(cols, x_shape, kh, kw, stride, pad):
    """Inverse of _im2col: scatter-add columns back into an (N, C, H, W) gradient."""
    N, C, H, W = x_shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    cols = cols.reshape(C, kh, kw, N, out_h, out_w)
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    dx_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    for y in range(kh):
        y_max = y + stride * out_h
        for xk in range(kw):
            x_max = xk + stride * out_w
            dx_padded[:, :, y:y_max:stride, xk:x_max:stride] += cols[:, y, xk, :, :, :].transpose(1, 0, 2, 3)

    if pad == 0:
        return dx_padded
    return dx_padded[:, :, pad:-pad, pad:-pad]


class Conv2D:
    """2D convolution over (N, C_in, H, W) inputs."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.stride = stride
        self.pad = padding
        self.kh = self.kw = kernel_size

        # He initialization, good default for ReLU networks
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

        self.dW = None
        self.db = None
        self._cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        F = self.W.shape[0]

        cols, out_h, out_w = _im2col(x, self.kh, self.kw, self.stride, self.pad)
        W_row = self.W.reshape(F, -1)  # (F, C*kh*kw)

        out = W_row @ cols + self.b[:, None]  # (F, N*out_h*out_w)
        out = out.reshape(F, N, out_h, out_w).transpose(1, 0, 2, 3)  # (N, F, out_h, out_w)

        self._cache = (x.shape, cols)
        return out

    def backward(self, dout):
        x_shape, cols = self._cache
        N, C, H, W = x_shape
        F = self.W.shape[0]

        dout_flat = dout.transpose(1, 0, 2, 3).reshape(F, -1)  # (F, N*out_h*out_w)

        self.db = dout_flat.sum(axis=1)
        self.dW = (dout_flat @ cols.T).reshape(self.W.shape)

        W_row = self.W.reshape(F, -1)
        dcols = W_row.T @ dout_flat  # (C*kh*kw, N*out_h*out_w)
        dx = _col2im(dcols, x_shape, self.kh, self.kw, self.stride, self.pad)
        return dx

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask

    def params_and_grads(self):
        return []


class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        s = self.size
        out_h = (H - s) // self.stride + 1
        out_w = (W - s) // self.stride + 1

        out = np.zeros((N, C, out_h, out_w))
        self._argmax = np.zeros((N, C, out_h, out_w, 2), dtype=int)

        for i in range(out_h):
            hs = i * self.stride
            for j in range(out_w):
                ws = j * self.stride
                window = x[:, :, hs:hs + s, ws:ws + s]  # (N, C, s, s)
                flat = window.reshape(N, C, -1)
                idx = np.argmax(flat, axis=2)
                out[:, :, i, j] = np.take_along_axis(flat, idx[..., None], axis=2)[..., 0]
                self._argmax[:, :, i, j, 0] = idx // s
                self._argmax[:, :, i, j, 1] = idx % s

        self._x_shape = x.shape
        return out

    def backward(self, dout):
        N, C, H, W = self._x_shape
        s = self.size
        out_h, out_w = dout.shape[2], dout.shape[3]
        dx = np.zeros(self._x_shape)

        for i in range(out_h):
            hs = i * self.stride
            for j in range(out_w):
                ws = j * self.stride
                di = self._argmax[:, :, i, j, 0]
                dj = self._argmax[:, :, i, j, 1]
                for n in range(N):
                    for c in range(C):
                        dx[n, c, hs + di[n, c], ws + dj[n, c]] += dout[n, c, i, j]
        return dx

    def params_and_grads(self):
        return []


class Flatten:
    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._shape)

    def params_and_grads(self):
        return []


class Dense:
    """Fully connected layer: out = x @ W + b"""

    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.dW = None
        self.db = None
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]
