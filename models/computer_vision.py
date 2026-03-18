import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
Computer Vision mit vortrainierten Modellen
=============================================
Bilder verstehen mit Deep Learning - Transfer Learning,
Feature Extraction und Image Classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# 1. Vortrainierte Modelle laden
# ============================================================
print("\n" + "=" * 60)
print("1. VORTRAINIERTE MODELLE - Was gibt es?")
print("=" * 60)

available_models = {
    "ResNet-18": {"params": "11.7M", "top1": "69.8%", "size": "45 MB"},
    "ResNet-50": {"params": "25.6M", "top1": "76.1%", "size": "98 MB"},
    "EfficientNet-B0": {"params": "5.3M", "top1": "77.1%", "size": "21 MB"},
    "MobileNet-V3-Small": {"params": "2.5M", "top1": "67.7%", "size": "10 MB"},
    "VGG-16": {"params": "138M", "top1": "71.6%", "size": "528 MB"},
    "ViT-B/16": {"params": "86.6M", "top1": "81.1%", "size": "330 MB"},
}

header = f"{'Modell':<22s} {'Parameter':>10s} {'Top-1 Acc':>10s} {'Groesse':>8s}"
print(f"\n{header}")
print("-" * len(header))
for name, info in available_models.items():
    print(f"{name:<22s} {info['params']:>10s} {info['top1']:>10s} {info['size']:>8s}")

print("\n* Alle vortrainiert auf ImageNet (1000 Klassen, 1.2M Bilder)")
print("  -> Du brauchst KEIN eigenes Training fuer viele Aufgaben!")

# ============================================================
# 2. ResNet laden und untersuchen
# ============================================================
print("\n" + "=" * 60)
print("2. RESNET-18 ARCHITEKTUR")
print("=" * 60)

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.eval()
resnet.to(device)

# Architektur anzeigen
total_params = sum(p.numel() for p in resnet.parameters())
trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
print(f"\nParameter gesamt:     {total_params:>12,}")
print(f"Trainierbar:          {trainable:>12,}")
print(f"Modell-Groesse:       ~{total_params * 4 / 1024 / 1024:.0f} MB (float32)")

print("\nLayer-Uebersicht:")
layers = []
for name, module in resnet.named_children():
    params = sum(p.numel() for p in module.parameters())
    layers.append((name, type(module).__name__, params))
    print(f"  {name:<12s} {type(module).__name__:<25s} {params:>10,} params")

# ============================================================
# 3. Synthetische Bilder erstellen und klassifizieren
# ============================================================
print("\n" + "=" * 60)
print("3. BILDER ERSTELLEN UND KLASSIFIZIEREN")
print("=" * 60)

# Preprocessing (ImageNet Standard)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Synthetische Bilder erstellen
def create_test_images():
    """Erstelle einfache Test-Bilder mit numpy."""
    images = {}

    # Rotes Quadrat auf weissem Hintergrund
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    img[60:164, 60:164] = [220, 50, 50]
    images["Rotes Quadrat"] = img

    # Blaue Kreise (Punkte)
    img = np.ones((224, 224, 3), dtype=np.uint8) * 240
    for cx, cy, r in [(70, 70, 25), (150, 100, 35), (100, 160, 20)]:
        yy, xx = np.ogrid[-cy:224-cy, -cx:224-cx]
        mask = xx**2 + yy**2 <= r**2
        img[mask] = [50, 50, 200]
    images["Blaue Kreise"] = img

    # Gruene Streifen
    img = np.ones((224, 224, 3), dtype=np.uint8) * 250
    for i in range(0, 224, 20):
        img[i:i+10, :] = [50, 180, 50]
    images["Gruene Streifen"] = img

    # Gradient (Sonnenuntergang-artig)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        r = int(255 * (1 - y / 224))
        g = int(100 * (1 - y / 224))
        b = int(50 + 150 * (y / 224))
        img[y, :] = [r, g, b]
    images["Gradient"] = img

    return images

images = create_test_images()

# ImageNet Labels laden (Top-Labels fuer unsere Bilder)
weights = models.ResNet18_Weights.DEFAULT
categories = weights.meta["categories"]

print("\nKlassifikation synthetischer Bilder:")
print("(Das Modell kennt nur ImageNet-Klassen, keine abstrakten Formen)\n")

all_preds = {}
for name, img_np in images.items():
    img_pil = Image.fromarray(img_np)
    input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = resnet(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5 = torch.topk(probs, 5)

    all_preds[name] = (img_np, top5)
    print(f"  {name}:")
    for i in range(3):
        idx = top5.indices[i].item()
        prob = top5.values[i].item()
        print(f"    {i+1}. {categories[idx]} ({prob:.1%})")
    print()

# ============================================================
# 4. Feature Extraction
# ============================================================
print("=" * 60)
print("4. FEATURE EXTRACTION - Bilder als Vektoren")
print("=" * 60)

# Entferne den letzten Layer (fc) -> bekomme Feature-Vektoren
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

print("\nFeature-Vektoren extrahieren:")
features = {}
for name, img_np in images.items():
    img_pil = Image.fromarray(img_np)
    input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = feature_extractor(input_tensor).squeeze()
    features[name] = feat.cpu().numpy()
    print(f"  {name}: Vektor mit {feat.shape[0]} Dimensionen")

# Aehnlichkeit berechnen
print("\nAehnlichkeit zwischen Bildern (Cosine Similarity):")
names = list(features.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = features[names[i]], features[names[j]]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        print(f"  {names[i]:.<20s} vs {names[j]:.<20s} -> {sim:.3f}")

# ============================================================
# 5. Transfer Learning Pattern
# ============================================================
print("\n" + "=" * 60)
print("5. TRANSFER LEARNING - So nutzt du vortrainierte Modelle")
print("=" * 60)

print("""
Transfer Learning in 3 Schritten:

  1. Vortrainiertes Modell laden (z.B. ResNet-18)
  2. Letzten Layer ersetzen (1000 Klassen -> deine Klassen)
  3. Nur den neuen Layer trainieren (oder alles fine-tunen)
""")

# Demo: Neuen Classifier Head erstellen
num_classes = 5  # z.B. Hund, Katze, Vogel, Fisch, Pferd
transfer_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Alle Gewichte einfrieren
for param in transfer_model.parameters():
    param.requires_grad = False

# Nur den letzten Layer ersetzen (trainierbar)
in_features = transfer_model.fc.in_features
transfer_model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes),
)

total = sum(p.numel() for p in transfer_model.parameters())
trainable = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
print(f"Transfer Learning Setup:")
print(f"  Gesamt-Parameter:     {total:>10,}")
print(f"  Trainierbar:          {trainable:>10,} ({trainable/total:.1%})")
print(f"  Eingefroren:          {total - trainable:>10,} ({(total-trainable)/total:.1%})")
print(f"\n  -> Du trainierst nur {trainable/total:.1%} der Parameter!")
print(f"     Das spart enorm Zeit und braucht wenig Daten.")

print("""
Training Code (Kurzversion):
  optimizer = torch.optim.Adam(transfer_model.fc.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(10):
      for images, labels in dataloader:
          outputs = transfer_model(images)
          loss = criterion(outputs, labels)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
""")

# ============================================================
# 6. Data Augmentation fuer Bilder
# ============================================================
print("=" * 60)
print("6. DATA AUGMENTATION")
print("=" * 60)

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("\nTypische Augmentationen:")
augmentations = [
    ("RandomResizedCrop", "Zufaelliger Ausschnitt + Skalierung"),
    ("RandomHorizontalFlip", "Zufaellig horizontal spiegeln"),
    ("RandomRotation(15)", "Bis zu 15 Grad drehen"),
    ("ColorJitter", "Helligkeit, Kontrast, Saettigung variieren"),
    ("RandomErasing", "Zufaellige Bereiche ausloeschen"),
    ("GaussianBlur", "Leichtes Verwischen"),
]
for name, desc in augmentations:
    print(f"  {name:<25s} -> {desc}")

# ============================================================
# 7. Visualisierung
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Computer Vision - Bilder & Klassifikation", fontsize=16)

# Original Bilder + Top Predictions
for idx, (name, (img_np, top5)) in enumerate(all_preds.items()):
    ax = axes[0, idx]
    ax.imshow(img_np)
    top_label = categories[top5.indices[0].item()]
    top_prob = top5.values[0].item()
    ax.set_title(f"{name}\n-> {top_label} ({top_prob:.0%})", fontsize=9)
    ax.axis("off")

# Augmentierte Versionen des ersten Bildes
orig_img = Image.fromarray(list(images.values())[0])
aug_transforms = [
    ("Original", transforms.Compose([])),
    ("Flip", transforms.RandomHorizontalFlip(p=1.0)),
    ("Rotation", transforms.RandomRotation(30)),
    ("Color Jitter", transforms.ColorJitter(brightness=0.5, contrast=0.5)),
]
for idx, (aug_name, aug_t) in enumerate(aug_transforms):
    ax = axes[1, idx]
    aug_img = aug_t(orig_img)
    ax.imshow(np.array(aug_img))
    ax.set_title(f"Augmentation: {aug_name}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("computer_vision.png", dpi=150)
print("\n[OK] Plot gespeichert als 'computer_vision.png'")

# ============================================================
# UEBUNGEN
# ============================================================
print("\n" + "=" * 60)
print("UEBUNGEN")
print("=" * 60)
print("""
1. ANDERES MODELL TESTEN
   Ersetze resnet18 durch efficientnet_b0:
     model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
   Sind die Vorhersagen anders? Wie gross ist das Modell?

2. ECHTE BILDER KLASSIFIZIEREN
   Lade ein echtes Bild von der Festplatte:
     img = Image.open("mein_bild.jpg").convert("RGB")
     input_tensor = preprocess(img).unsqueeze(0)
   Was erkennt ResNet?

3. FEATURE-VERGLEICH
   Extrahiere Features aus verschiedenen Layern (nicht nur dem letzten).
   Fruehe Layer = Kanten/Texturen, spaete Layer = Objekte/Konzepte.

4. EIGENER CLASSIFIER
   Erstelle einen Datensatz mit 3 Kategorien (z.B. Formen),
   nutze Transfer Learning und trainiere den Classifier Head.
""")
