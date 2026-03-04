"""End-to-end: train a tiny MLP on XOR-like data using picotorch Tensor engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from picotorch.tensor import Tensor
from picotorch.nn import MLP
import numpy as np

np.random.seed(42)

# XOR dataset
X = [
    Tensor([0.0, 0.0]),
    Tensor([0.0, 1.0]),
    Tensor([1.0, 0.0]),
    Tensor([1.0, 1.0]),
]
y_true = [0.0, 1.0, 1.0, 0.0]

# Small MLP: 2 -> 4 -> 4 -> 1
model = MLP(2, [4, 4, 1])

lr = 0.05
print("Training XOR...")
for epoch in range(100):
    # Forward pass
    total_loss = Tensor([0.0], requires_grad=True)
    for xi, yi in zip(X, y_true):
        pred = model(xi)
        diff = pred + Tensor([-yi])
        loss = diff ** 2
        total_loss = total_loss + loss

    # Backward pass
    total_loss.backward()

    # SGD update
    for p in model.parameters():
        if p.grad is not None:
            p._data -= lr * p.grad._data

    # Zero gradients
    model.zero_grad()

    if (epoch + 1) % 20 == 0:
        print(f"  epoch {epoch+1:3d}, loss = {total_loss.item():.4f}")

# Verify predictions
print("\nPredictions after training:")
correct = 0
for xi, yi in zip(X, y_true):
    pred = model(xi)
    pred_val = pred.item()
    predicted_class = 1.0 if pred_val > 0.5 else 0.0
    match = "OK" if predicted_class == yi else "MISS"
    if predicted_class == yi:
        correct += 1
    print(f"  input={xi.numpy()}, target={yi}, pred={pred_val:.4f} [{match}]")

print(f"\nAccuracy: {correct}/4")
assert correct >= 3, f"Expected at least 3/4 correct, got {correct}/4"
print("PASS: end-to-end training")
