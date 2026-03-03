"""Tensor class with autograd — the heart of picotorch."""
import numpy as np
from picotorch import ops_dispatch as dispatch


class Tensor:
    """A multi-dimensional array with automatic differentiation."""

    def __init__(self, data, device='cpu', requires_grad=False,
                 _children=(), _op='', _shape=None, _data_raw=None):
        if _data_raw is not None:
            self._data = _data_raw
            self.shape = _shape if _shape else (_data_raw.size,)
        else:
            if isinstance(data, (int, float)):
                arr = np.array([data], dtype=np.float32)
                self.shape = (1,)
            elif isinstance(data, np.ndarray):
                self.shape = data.shape
                arr = data.astype(np.float32).ravel()
            elif isinstance(data, list):
                arr = np.array(data, dtype=np.float32)
                self.shape = arr.shape
                arr = arr.ravel()
            else:
                raise TypeError(f"Cannot create Tensor from {type(data)}")

            if device == 'cuda':
                self._data = dispatch.to_device(arr, 'cpu', 'cuda')
            else:
                self._data = np.ascontiguousarray(arr)

        self.device = device
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def size(self):
        return self._data.size

    @property
    def ndim(self):
        return len(self.shape)

    def numpy(self):
        if self.device == 'cuda':
            return dispatch.to_device(self._data, 'cuda', 'cpu').reshape(self.shape)
        return self._data.copy().reshape(self.shape)

    def to(self, device):
        if device == self.device:
            return self
        new_data = dispatch.to_device(self._data, self.device, device)
        return Tensor(None, device=device, requires_grad=self.requires_grad,
                      _shape=self.shape, _data_raw=new_data)

    def _ensure_tensor(self, other):
        if isinstance(other, Tensor):
            return other
        return Tensor(other, device=self.device)

    def __add__(self, other):
        other = self._ensure_tensor(other)
        out_data = dispatch.add(self._data, other._data, self.device)
        out = Tensor(None, device=self.device, _children=(self, other), _op='+',
                     _shape=self.shape, _data_raw=out_data)

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor(None, device=self.device,
                                       _shape=self.shape, _data_raw=dispatch.zeros(self.size, self.device))
                dispatch.add_inplace(self.grad._data, out.grad._data, self.device)
            if other.requires_grad:
                if other.grad is None:
                    other.grad = Tensor(None, device=self.device,
                                        _shape=other.shape, _data_raw=dispatch.zeros(other.size, self.device))
                dispatch.add_inplace(other.grad._data, out.grad._data, self.device)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        return self + (-self._ensure_tensor(other))

    def __rsub__(self, other):
        return self._ensure_tensor(other) + (-self)

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        out_data = dispatch.mul(self._data, other._data, self.device)
        out = Tensor(None, device=self.device, _children=(self, other), _op='*',
                     _shape=self.shape, _data_raw=out_data)

        def _backward():
            if self.requires_grad:
                g = dispatch.mul(other._data, out.grad._data, self.device)
                if self.grad is None:
                    self.grad = Tensor(None, device=self.device,
                                       _shape=self.shape, _data_raw=dispatch.zeros(self.size, self.device))
                dispatch.add_inplace(self.grad._data, g, self.device)
            if other.requires_grad:
                g = dispatch.mul(self._data, out.grad._data, self.device)
                if other.grad is None:
                    other.grad = Tensor(None, device=self.device,
                                        _shape=other.shape, _data_raw=dispatch.zeros(other.size, self.device))
                dispatch.add_inplace(other.grad._data, g, self.device)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (self._ensure_tensor(other) ** -1)

    def __rtruediv__(self, other):
        return self._ensure_tensor(other) * (self ** -1)

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only int/float powers supported"
        out_data = dispatch.pow_scalar(self._data, exponent, self.device)
        out = Tensor(None, device=self.device, _children=(self,), _op=f'**{exponent}',
                     _shape=self.shape, _data_raw=out_data)

        def _backward():
            if self.requires_grad:
                coeff_data = dispatch.pow_scalar(self._data, exponent - 1, self.device)
                n_arr = dispatch.fill(self.size, exponent, self.device)
                g = dispatch.mul(n_arr, coeff_data, self.device)
                g = dispatch.mul(g, out.grad._data, self.device)
                if self.grad is None:
                    self.grad = Tensor(None, device=self.device,
                                       _shape=self.shape, _data_raw=dispatch.zeros(self.size, self.device))
                dispatch.add_inplace(self.grad._data, g, self.device)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def relu(self):
        out_data = dispatch.relu(self._data, self.device)
        out = Tensor(None, device=self.device, _children=(self,), _op='ReLU',
                     _shape=self.shape, _data_raw=out_data)

        def _backward():
            if self.requires_grad:
                if self.device == 'cpu':
                    m = (self._data > 0).astype(np.float32)
                else:
                    import cupy as cp
                    m = (self._data > 0).astype(cp.float32)
                g = dispatch.mul(m, out.grad._data, self.device)
                if self.grad is None:
                    self.grad = Tensor(None, device=self.device,
                                       _shape=self.shape, _data_raw=dispatch.zeros(self.size, self.device))
                dispatch.add_inplace(self.grad._data, g, self.device)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def sum(self):
        out_data = dispatch.sum_all(self._data, self.device)
        out = Tensor(None, device=self.device, _children=(self,), _op='sum',
                     _shape=(1,), _data_raw=out_data)

        def _backward():
            if self.requires_grad:
                if self.device == 'cpu':
                    scalar_grad = float(out.grad._data[0])
                else:
                    scalar_grad = float(out.grad._data.get()[0])
                g = dispatch.fill(self.size, scalar_grad, self.device)
                if self.grad is None:
                    self.grad = Tensor(None, device=self.device,
                                       _shape=self.shape, _data_raw=dispatch.zeros(self.size, self.device))
                dispatch.add_inplace(self.grad._data, g, self.device)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def matmul(self, other):
        assert self.ndim == 2 and other.ndim == 2, "matmul requires 2D tensors"
        m, n = self.shape
        n2, p = other.shape
        assert n == n2, f"matmul shape mismatch: ({m},{n}) @ ({n2},{p})"

        out_data = dispatch.matmul(self._data, other._data, self.device, m, n, p)
        out = Tensor(None, device=self.device, _children=(self, other), _op='@',
                     _shape=(m, p), _data_raw=out_data.ravel())

        def _backward():
            if self.requires_grad:
                g = dispatch.matmul_backward_a(out.grad._data, other._data, self.device, m, n, p)
                if self.grad is None:
                    self.grad = Tensor(None, device=self.device,
                                       _shape=self.shape, _data_raw=dispatch.zeros(self.size, self.device))
                dispatch.add_inplace(self.grad._data, g.ravel(), self.device)
            if other.requires_grad:
                g = dispatch.matmul_backward_b(self._data, out.grad._data, self.device, m, n, p)
                if other.grad is None:
                    other.grad = Tensor(None, device=self.device,
                                        _shape=other.shape, _data_raw=dispatch.zeros(other.size, self.device))
                dispatch.add_inplace(other.grad._data, g.ravel(), self.device)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def reshape(self, *shape):
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        total = 1
        for s in new_shape:
            total *= s
        assert total == self.size, f"Cannot reshape {self.shape} to {new_shape}"

        out = Tensor(None, device=self.device, _children=(self,), _op='reshape',
                     _shape=new_shape, _data_raw=self._data)
        out.requires_grad = self.requires_grad

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor(None, device=self.device,
                                       _shape=self.shape, _data_raw=dispatch.zeros(self.size, self.device))
                dispatch.add_inplace(self.grad._data, out.grad._data, self.device)

        out._backward = _backward
        return out

    def zero_grad(self):
        self.grad = None

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = Tensor(None, device=self.device,
                           _shape=self.shape, _data_raw=dispatch.ones(self.size, self.device))

        for v in reversed(topo):
            v._backward()

    def item(self):
        assert self.size == 1, f"item() only for size-1 tensors, got {self.size}"
        if self.device == 'cuda':
            return float(self._data.get()[0])
        return float(self._data[0])

    def __repr__(self):
        data_str = np.array2string(self.numpy(), precision=4, separator=', ')
        return f"Tensor({data_str}, device='{self.device}')"

    def __len__(self):
        return self.shape[0]
