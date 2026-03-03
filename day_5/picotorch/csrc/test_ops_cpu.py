"""
Test suite for picotorch C CPU kernels via Python ctypes.
Verifies all functions in libops_cpu.dylib.
"""
import ctypes
import os
import sys

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), "libops_cpu.dylib")
lib = ctypes.CDLL(lib_path)

# --- Helper: create a ctypes float array from a Python list ---
def float_array(values):
    return (ctypes.c_float * len(values))(*values)

def zeros(n):
    return (ctypes.c_float * n)(*([0.0] * n))

def to_list(arr):
    return list(arr)

# --- Configure function signatures ---
lib.pt_add.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_float, ctypes.c_int]
lib.pt_add.restype = None

lib.pt_mul.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int]
lib.pt_mul.restype = None

lib.pt_matmul.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int] * 3
lib.pt_matmul.restype = None

lib.pt_matmul_backward_a.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int] * 3
lib.pt_matmul_backward_a.restype = None

lib.pt_matmul_backward_b.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int] * 3
lib.pt_matmul_backward_b.restype = None

lib.pt_pow_scalar.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2 + [ctypes.c_float, ctypes.c_int]
lib.pt_pow_scalar.restype = None

lib.pt_relu.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2 + [ctypes.c_int]
lib.pt_relu.restype = None

lib.pt_sum.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2 + [ctypes.c_int]
lib.pt_sum.restype = None

lib.pt_fill.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_int]
lib.pt_fill.restype = None

lib.pt_add_inplace.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2 + [ctypes.c_int]
lib.pt_add_inplace.restype = None

lib.pt_add_scalar.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2 + [ctypes.c_float, ctypes.c_int]
lib.pt_add_scalar.restype = None


# --- Tolerance helper ---
def approx_equal(a, b, tol=1e-5):
    return all(abs(x - y) < tol for x, y in zip(a, b))

def assert_close(actual, expected, label, tol=1e-5):
    if isinstance(actual, (list, tuple)):
        ok = approx_equal(actual, expected, tol)
    else:
        ok = abs(actual - expected) < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok:
        print(f"    expected: {expected}")
        print(f"    actual:   {actual}")
    return ok


# ===================== Tests =====================
all_passed = True

def run_test(name, fn):
    global all_passed
    print(f"\n--- {name} ---")
    result = fn()
    if not result:
        all_passed = False


def test_add():
    a = float_array([1.0, 2.0, 3.0])
    b = float_array([4.0, 5.0, 6.0])
    out = zeros(3)
    # alpha=1.0: out = a + b
    lib.pt_add(a, b, out, ctypes.c_float(1.0), 3)
    r1 = assert_close(to_list(out), [5.0, 7.0, 9.0], "add with alpha=1.0")
    # alpha=2.0: out = a + 2*b
    lib.pt_add(a, b, out, ctypes.c_float(2.0), 3)
    r2 = assert_close(to_list(out), [9.0, 12.0, 15.0], "add with alpha=2.0")
    # alpha=0.0: out = a
    lib.pt_add(a, b, out, ctypes.c_float(0.0), 3)
    r3 = assert_close(to_list(out), [1.0, 2.0, 3.0], "add with alpha=0.0")
    return r1 and r2 and r3

def test_mul():
    a = float_array([2.0, 3.0, 4.0])
    b = float_array([5.0, 6.0, 7.0])
    out = zeros(3)
    lib.pt_mul(a, b, out, 3)
    return assert_close(to_list(out), [10.0, 18.0, 28.0], "element-wise multiply")

def test_matmul():
    # A = [[1,2],[3,4]]  (2x2), B = [[5,6],[7,8]] (2x2)
    a = float_array([1.0, 2.0, 3.0, 4.0])
    b = float_array([5.0, 6.0, 7.0, 8.0])
    out = zeros(4)
    lib.pt_matmul(a, b, out, 2, 2, 2)
    # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    return assert_close(to_list(out), [19.0, 22.0, 43.0, 50.0], "matmul 2x2 @ 2x2")

def test_matmul_nonsquare():
    # A = [[1,2,3]] (1x3), B = [[4],[5],[6]] (3x1)
    a = float_array([1.0, 2.0, 3.0])
    b = float_array([4.0, 5.0, 6.0])
    out = zeros(1)
    lib.pt_matmul(a, b, out, 1, 3, 1)
    # 1*4 + 2*5 + 3*6 = 32
    return assert_close(to_list(out), [32.0], "matmul 1x3 @ 3x1")

def test_matmul_backward_a():
    # grad_out (2x2), b (2x2) => grad_a (2x2)
    # grad_a = grad_out @ b^T
    grad_out = float_array([1.0, 0.0, 0.0, 1.0])  # identity
    b = float_array([5.0, 6.0, 7.0, 8.0])
    grad_a = zeros(4)
    lib.pt_matmul_backward_a(grad_out, b, grad_a, 2, 2, 2)
    # b^T = [[5,7],[6,8]], I @ b^T = b^T = [[5,7],[6,8]]
    return assert_close(to_list(grad_a), [5.0, 7.0, 6.0, 8.0], "matmul backward A (identity grad_out)")

def test_matmul_backward_b():
    # a (2x2), grad_out (2x2) => grad_b (2x2)
    # grad_b = a^T @ grad_out
    a = float_array([1.0, 2.0, 3.0, 4.0])
    grad_out = float_array([1.0, 0.0, 0.0, 1.0])  # identity
    grad_b = zeros(4)
    lib.pt_matmul_backward_b(a, grad_out, grad_b, 2, 2, 2)
    # a^T = [[1,3],[2,4]], a^T @ I = a^T = [[1,3],[2,4]]
    return assert_close(to_list(grad_b), [1.0, 3.0, 2.0, 4.0], "matmul backward B (identity grad_out)")

def test_pow_scalar():
    a = float_array([1.0, 2.0, 3.0, 4.0])
    out = zeros(4)
    lib.pt_pow_scalar(a, out, ctypes.c_float(2.0), 4)
    r1 = assert_close(to_list(out), [1.0, 4.0, 9.0, 16.0], "pow with exponent=2")
    lib.pt_pow_scalar(a, out, ctypes.c_float(0.5), 4)
    r2 = assert_close(to_list(out), [1.0, 1.41421356, 1.73205081, 2.0], "pow with exponent=0.5 (sqrt)", tol=1e-4)
    return r1 and r2

def test_relu():
    a = float_array([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = zeros(5)
    lib.pt_relu(a, out, 5)
    return assert_close(to_list(out), [0.0, 0.0, 0.0, 1.0, 2.0], "relu")

def test_sum():
    a = float_array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = zeros(1)
    lib.pt_sum(a, out, 5)
    return assert_close([out[0]], [15.0], "sum reduction")

def test_fill():
    out = zeros(5)
    lib.pt_fill(out, ctypes.c_float(3.14), 5)
    return assert_close(to_list(out), [3.14] * 5, "fill", tol=1e-4)

def test_add_inplace():
    out = float_array([1.0, 2.0, 3.0])
    a = float_array([10.0, 20.0, 30.0])
    lib.pt_add_inplace(out, a, 3)
    return assert_close(to_list(out), [11.0, 22.0, 33.0], "add inplace")

def test_add_scalar():
    a = float_array([1.0, 2.0, 3.0])
    out = zeros(3)
    lib.pt_add_scalar(a, out, ctypes.c_float(10.0), 3)
    return assert_close(to_list(out), [11.0, 12.0, 13.0], "add scalar")


# ===================== Run all tests =====================
run_test("pt_add", test_add)
run_test("pt_mul", test_mul)
run_test("pt_matmul (square)", test_matmul)
run_test("pt_matmul (non-square)", test_matmul_nonsquare)
run_test("pt_matmul_backward_a", test_matmul_backward_a)
run_test("pt_matmul_backward_b", test_matmul_backward_b)
run_test("pt_pow_scalar", test_pow_scalar)
run_test("pt_relu", test_relu)
run_test("pt_sum", test_sum)
run_test("pt_fill", test_fill)
run_test("pt_add_inplace", test_add_inplace)
run_test("pt_add_scalar", test_add_scalar)

print("\n" + "=" * 40)
if all_passed:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
