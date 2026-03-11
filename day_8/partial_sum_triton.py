"""
Day 8 - Partial Sum (Triton): Block-wise inclusive scan (prefix sum)

Mirrors day_8/partialSum.cu:
  - Each block/program processes BLOCK_SIZE elements.
  - Performs inclusive scan (prefix sum) within the block in shared memory.
  - Output[i] = sum(Input[block_start : i+1]) for the block containing i.

Math:
  For block b covering indices [b*BLOCK_SIZE, (b+1)*BLOCK_SIZE):
    output[b*BLOCK_SIZE + j] = input[b*BLOCK_SIZE] + ... + input[b*BLOCK_SIZE + j]
  (no cross-block aggregation; partial sums per block only).

Inputs / Outputs:
  input: 1D int32 tensor on CUDA, length n
  output: 1D int32 tensor on CUDA, length n (inclusive scan per block)

Assumptions:
  - input is contiguous, on CUDA, dtype int32 (matching CUDA int)
  - n > 0

Parallel Strategy:
  - 1D grid; each Triton program handles one block of BLOCK_SIZE elements.
  - Load block → inclusive cumsum → store (no shared-memory explicit loop;
    Triton compiles efficient scan).

Build / Run:
  python day_8/partial_sum_triton.py
"""

import torch
import triton
import triton.language as tl

# Match CUDA kernel: 256 elements per block
BLOCK_SIZE_DEFAULT = 256


@triton.jit
def partial_sum_kernel(
    input_ptr,
    output_ptr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Per-block inclusive scan: load a block, compute inclusive cumsum, store.
    Same semantics as partialSum.cu partialSumKernel (shared memory → scan → write).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load block (coalesced); out-of-bounds as 0 for scan
    block = tl.load(input_ptr + offsets, mask=mask, other=0)

    # Inclusive scan in the block (axis=0 for 1D)
    scanned = tl.cumsum(block, axis=0)

    tl.store(output_ptr + offsets, scanned, mask=mask)


def partial_sum_triton(
    input_tensor: torch.Tensor,
    *,
    block_size: int = BLOCK_SIZE_DEFAULT,
) -> torch.Tensor:
    """
    Partial sum: per-block inclusive prefix sum (same as day_8/partialSum.cu).

    Args:
        input_tensor: 1D int32 tensor on CUDA, length n.
        block_size: Number of elements per block (default 256 to match CUDA).

    Returns:
        1D int32 tensor of same shape; each block holds its own inclusive scan.
    """
    assert input_tensor.is_cuda, "Input must be on CUDA"
    assert input_tensor.dim() == 1, "Input must be 1D"
    assert input_tensor.dtype == torch.int32, "Input must be int32 (match CUDA int)"
    assert input_tensor.is_contiguous(), "Input must be contiguous"

    n = input_tensor.shape[0]
    assert n > 0, "Length must be positive"

    output = torch.empty_like(input_tensor)
    grid = (triton.cdiv(n, block_size),)

    partial_sum_kernel[grid](
        input_tensor,
        output,
        n=n,
        BLOCK_SIZE=block_size,
    )

    return output


def partial_sum_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    """Per-block inclusive scan on CPU (same semantics as CUDA/Triton)."""
    n = input_tensor.shape[0]
    block_size = BLOCK_SIZE_DEFAULT
    out = torch.empty_like(input_tensor)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = input_tensor[start:end].clone()
        out[start:end] = torch.cumsum(block, dim=0)
    return out


def test_partial_sum(n: int = 1_000_000) -> None:
    """
    Test partial sum with correctness check and timing.
    Matches day_8/partialSum.cu: n=1000000, rand % 100, then "Partial Sum - result OK".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available; skipping test.")
        return

    torch.manual_seed(0)
    # Match CUDA: values in [0, 99]
    input_host = torch.randint(0, 100, (n,), dtype=torch.int32)
    input_tensor = input_host.to(device)

    # Warmup
    for _ in range(5):
        _ = partial_sum_triton(input_tensor)
    torch.cuda.synchronize()

    # Time kernel
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    output = partial_sum_triton(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    print(f"Partial Sum - elapsed time: {elapsed_ms:.3f} ms")

    # Correctness: compare to per-block inclusive scan reference
    ref = partial_sum_reference(input_host)
    output_host = output.cpu()
    max_diff = (output_host - ref).abs().max().item()
    if max_diff == 0:
        print("Partial Sum - result OK")
    else:
        print(f"Partial Sum - MISMATCH max|out - ref| = {max_diff}")


if __name__ == "__main__":
    test_partial_sum(1_000_000)