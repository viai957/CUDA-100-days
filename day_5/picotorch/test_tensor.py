"""Tests for Tensor class."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from picotorch.tensor import Tensor
import numpy as np

# Test 1: creation
t = Tensor([1.0, 2.0, 3.0])
assert t.shape == (3,), f"Expected (3,), got {t.shape}"
assert t.device == 'cpu'
print("PASS: creation")

# Test 2: add
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])
c = a + b
np.testing.assert_allclose(c.numpy(), [5.0, 7.0, 9.0])
print("PASS: add")

# Test 3: mul
c = a * b
np.testing.assert_allclose(c.numpy(), [4.0, 10.0, 18.0])
print("PASS: mul")

# Test 4: backward through add
a = Tensor([2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0], requires_grad=True)
c = a + b
d = c.sum()
d.backward()
np.testing.assert_allclose(a.grad.numpy(), [1.0, 1.0])
np.testing.assert_allclose(b.grad.numpy(), [1.0, 1.0])
print("PASS: backward add")

# Test 5: backward through mul
a = Tensor([2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0], requires_grad=True)
c = a * b
d = c.sum()
d.backward()
np.testing.assert_allclose(a.grad.numpy(), [4.0, 5.0])
np.testing.assert_allclose(b.grad.numpy(), [2.0, 3.0])
print("PASS: backward mul")

# Test 6: relu backward
a = Tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
b = a.relu()
c = b.sum()
c.backward()
np.testing.assert_allclose(a.grad.numpy(), [0.0, 1.0, 0.0, 1.0])
print("PASS: backward relu")

# Test 7: matmul
a = Tensor([[1.0, 2.0], [3.0, 4.0]])
b = Tensor([[5.0, 6.0], [7.0, 8.0]])
c = a.matmul(b)
np.testing.assert_allclose(c.numpy().reshape(2,2), [[19., 22.], [43., 50.]])
print("PASS: matmul")

# Test 8: pow + backward
a = Tensor([2.0, 3.0], requires_grad=True)
b = a ** 2
c = b.sum()
c.backward()
np.testing.assert_allclose(a.grad.numpy(), [4.0, 6.0])
print("PASS: backward pow")

# Test 9: repr
t = Tensor([1.0, 2.0])
r = repr(t)
assert "Tensor(" in r and "device='cpu'" in r
print("PASS: repr")

# Test 10: item
t = Tensor([42.0])
assert t.item() == 42.0
print("PASS: item")

print("\nAll tensor tests passed!")
