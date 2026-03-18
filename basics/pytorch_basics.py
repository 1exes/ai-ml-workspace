import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
01 - PyTorch Grundlagen
========================
PyTorch ist DAS Framework fuer Deep Learning.
Hier lernst du Tensoren, Autograd und dein erstes echtes Neural Network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA verfuegbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende: {device}\n")

# ============================================================
# 1. Tensoren - wie NumPy, aber mit GPU-Support und Autograd
# ============================================================

# Erstellen
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.randn(3)  # Normalverteilte Zufallswerte
print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a . b = {torch.dot(a, b):.4f}")

# GPU Transfer (wenn verfuegbar)
a_gpu = a.to(device)
print(f"Tensor auf {a_gpu.device}")

# ============================================================
# 2. Autograd - automatische Ableitung (Backpropagation!)
# ============================================================

print("\n--- Autograd Demo ---")
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1

y.backward()  # Berechne dy/dx
print(f"x = {x.item()}")
print(f"y = x^2 + 2x + 1 = {y.item()}")
print(f"dy/dx = 2x + 2 = {x.grad.item()}")  # Bei x=3: 2*3+2 = 8

# ============================================================
# 3. Ein Neural Network als Klasse
# ============================================================

class SimpleNet(nn.Module):
    """Ein einfaches 3-Layer Neural Network."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),   # Layer 1
            nn.ReLU(),                             # Aktivierung
            nn.Linear(hidden_size, hidden_size),  # Layer 2
            nn.ReLU(),                             # Aktivierung
            nn.Linear(hidden_size, output_size),  # Output Layer
        )

    def forward(self, x):
        return self.layers(x)

# ============================================================
# 4. Training Loop - das Herzstueck von Deep Learning
# ============================================================

print("\n--- Training eines Neural Networks ---")

# Synthetische Daten (Spiralen - schwer fuer lineares Modell)
np.random.seed(42)
N = 300  # Punkte pro Klasse
K = 3    # Anzahl Klassen

X_np = np.zeros((N * K, 2))
y_np = np.zeros(N * K, dtype=int)
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
    X_np[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y_np[ix] = j

# NumPy -> PyTorch Tensoren
X = torch.FloatTensor(X_np).to(device)
y = torch.LongTensor(y_np).to(device)

# DataLoader fuer Mini-Batches
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Modell erstellen
model = SimpleNet(input_size=2, hidden_size=64, output_size=K).to(device)
criterion = nn.CrossEntropyLoss()       # Loss Funktion
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Optimizer

# Training
for epoch in range(200):
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in loader:
        # Forward Pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward Pass
        optimizer.zero_grad()  # Gradienten zuruecksetzen
        loss.backward()        # Gradienten berechnen
        optimizer.step()       # Gewichte updaten

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    if (epoch + 1) % 50 == 0:
        acc = correct / total
        print(f"Epoch {epoch + 1:3d} | Loss: {total_loss / len(loader):.4f} | Accuracy: {acc:.1%}")

# Finale Accuracy
model.eval()
with torch.no_grad():
    outputs = model(X)
    _, predicted = outputs.max(1)
    acc = (predicted == y).float().mean()
    print(f"\nFinale Accuracy: {acc:.1%}")

# Model Info
total_params = sum(p.numel() for p in model.parameters())
print(f"Modell-Parameter: {total_params:,}")
print(f"Modell-Groesse: ~{total_params * 4 / 1024:.1f} KB (float32)")

print("\n* Das ist der Standard-Loop: forward -> loss -> backward -> update")
print("  Jedes ML-Training (auch GPT!) funktioniert so.")

# ============================================================
# UEBUNGEN
# ============================================================
#
# Aufgabe 1: Aendere die Netzwerk-Architektur in SimpleNet:
#   - Fuege einen dritten Hidden Layer hinzu (z.B. 32 Neuronen)
#   - Ersetze ReLU durch nn.Tanh() oder nn.LeakyReLU()
#   - Trainiere erneut und vergleiche die finale Accuracy.
#   Welche Aktivierungsfunktion funktioniert am besten fuer Spiralen?
#
# Aufgabe 2: Implementiere Learning Rate Scheduling:
#   - Verwende torch.optim.lr_scheduler.StepLR (step_size=50, gamma=0.5)
#   - Logge die Learning Rate alle 50 Epochen mit scheduler.get_last_lr()
#   - Vergleiche das Trainingsverhalten mit vs. ohne Scheduler.
#   Verbessert sich die Konvergenz?
