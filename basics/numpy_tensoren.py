import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
01 - NumPy & Tensoren Grundlagen
================================
Alles in ML basiert auf Tensoren (mehrdimensionale Arrays).
Hier lernst du die Grundlagen, die du fuer alles Weitere brauchst.
"""

import numpy as np

# ============================================================
# 1. Arrays erstellen - das Fundament von ML
# ============================================================

# Vektoren (1D) - z.B. ein einzelnes Datensample mit 3 Features
vektor = np.array([1.0, 2.0, 3.0])
print(f"Vektor: {vektor}")
print(f"Shape: {vektor.shape}")  # (3,)

# Matrix (2D) - z.B. ein Batch von 3 Samples mit je 4 Features
matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])
print(f"\nMatrix:\n{matrix}")
print(f"Shape: {matrix.shape}")  # (3, 4) = 3 Zeilen, 4 Spalten

# 3D Tensor - z.B. ein Batch von Bildern (batch, hoehe, breite)
tensor_3d = np.random.randn(2, 3, 4)  # 2 Bilder, 3x4 Pixel
print(f"\n3D Tensor Shape: {tensor_3d.shape}")

# ============================================================
# 2. Wichtige Operationen fuer ML
# ============================================================

# Dot Product (Skalarprodukt) - DAS ist die Kernoperation in Neural Networks
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"\nDot Product: {a} . {b} = {dot}")

# Matrix-Multiplikation - so funktioniert ein Neural Network Layer
weights = np.random.randn(4, 3)  # 4 Input-Features -> 3 Output-Features
input_data = np.random.randn(2, 4)  # Batch von 2 Samples, je 4 Features
output = input_data @ weights  # Matrix-Multiplikation
print(f"\nInput shape: {input_data.shape}")
print(f"Weights shape: {weights.shape}")
print(f"Output shape: {output.shape}")  # (2, 3)

# ============================================================
# 3. Aktivierungsfunktionen - machen Neural Networks "intelligent"
# ============================================================

def relu(x):
    """ReLU: Negative Werte -> 0, positive bleiben."""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid: Quetscht alles zwischen 0 und 1."""
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """Softmax: Wandelt Zahlen in Wahrscheinlichkeiten um (summe = 1)."""
    exp_x = np.exp(x - np.max(x))  # Numerisch stabil
    return exp_x / exp_x.sum()

x = np.array([-2, -1, 0, 1, 2])
print(f"\nInput:   {x}")
print(f"ReLU:    {relu(x)}")
print(f"Sigmoid: {sigmoid(x).round(3)}")
print(f"Softmax: {softmax(x).round(3)} (Summe: {softmax(x).sum():.1f})")

# ============================================================
# 4. Ein simples "Neural Network" nur mit NumPy
# ============================================================

print("\n" + "=" * 50)
print("Mini Neural Network (XOR Problem)")
print("=" * 50)

# XOR: Die einfachste Aufgabe, die ein einzelnes Neuron NICHT loesen kann
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
y = np.array([[0], [1], [1], [0]])  # XOR Output

# Zufaellige Gewichte
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5   # Input -> Hidden (2->4 Neuronen)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5   # Hidden -> Output (4->1 Neuron)
b2 = np.zeros((1, 1))

learning_rate = 1.0

for epoch in range(10000):
    # Forward Pass
    hidden = sigmoid(X @ W1 + b1)       # Input -> Hidden
    output = sigmoid(hidden @ W2 + b2)  # Hidden -> Output

    # Loss berechnen (Mean Squared Error)
    loss = np.mean((y - output) ** 2)

    # Backward Pass (Backpropagation)
    d_output = (output - y) * output * (1 - output)
    d_hidden = (d_output @ W2.T) * hidden * (1 - hidden)

    W2 -= learning_rate * hidden.T @ d_output
    b2 -= learning_rate * d_output.sum(axis=0, keepdims=True)
    W1 -= learning_rate * X.T @ d_hidden
    b1 -= learning_rate * d_hidden.sum(axis=0, keepdims=True)

    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

# Ergebnis
print(f"\nErgebnis nach Training:")
for i in range(4):
    pred = output[i, 0]
    print(f"  {X[i]} -> {pred:.4f} (erwartet: {y[i, 0]})")

print("\n* Das war ein echtes Neural Network! Nur mit NumPy, keine Libraries.")
print("  In PyTorch geht das gleiche in ~10 Zeilen.")
