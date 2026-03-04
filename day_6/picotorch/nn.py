"""Neural network modules built on picotorch Tensor."""
import numpy as np
from picotorch.tensor import Tensor
from picotorch import ops_dispatch as dispatch


def _cat(tensors):
    """Concatenate a list of 1-D Tensors into a single 1-D Tensor (differentiable)."""
    device = tensors[0].device
    data_parts = [t._data for t in tensors]
    out_data = np.concatenate(data_parts)
    total = out_data.size
    out = Tensor(None, device=device, _children=tuple(tensors), _op='cat',
                 _shape=(total,), _data_raw=out_data)
    out.requires_grad = any(t.requires_grad for t in tensors)

    def _backward():
        offset = 0
        for t in tensors:
            sz = t.size
            if t.requires_grad:
                g = out.grad._data[offset:offset + sz].copy()
                if t.grad is None:
                    t.grad = Tensor(None, device=device,
                                    _shape=t.shape, _data_raw=dispatch.zeros(sz, device))
                dispatch.add_inplace(t.grad._data, g, device)
            offset += sz

    out._backward = _backward
    return out


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        scale = (2.0 / nin) ** 0.5
        self.w = Tensor(np.random.uniform(-scale, scale, nin).astype(np.float32),
                        requires_grad=True)
        self.b = Tensor([0.0], requires_grad=True)
        self.nonlin = nonlin

    def __call__(self, x):
        act = (self.w * x).sum() + self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({self.w.size})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else _cat(out)

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(repr(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i != len(nouts)-1)
                       for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(repr(layer) for layer in self.layers)}]"
