import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pickle


MODEL_SAVE = "model/cnn_model.pth"
LABEL_SAVE = "model/label_map.pkl"
BATCH_SIZE = 64
EPOCHS     = 15
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Loading MNIST dataset...")
full_train = datasets.MNIST(root='dataset', train=True,  download=True, transform=transform)
test_ds    = datasets.MNIST(root='dataset', train=False, download=True, transform=transform)

val_size   = int(0.15 * len(full_train))
train_size = len(full_train) - val_size
train_ds, val_ds = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {train_size} | Val: {val_size} | Test: {len(test_ds)}")


class HandwrittenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)

model     = HandwrittenCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)


os.makedirs("model", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

id_to_label = {i: str(i) for i in range(10)}
with open(LABEL_SAVE, "wb") as f:
    pickle.dump(id_to_label, f)

best_val_acc     = 0.0
patience_counter = 0
PATIENCE         = 7


print("\nTraining started...")
for epoch in range(EPOCHS):
    model.train()
    t_correct, t_total = 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        t_correct += (out.argmax(1) == y).sum().item()
        t_total   += y.size(0)

    # Validation
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y  = x.to(DEVICE), y.to(DEVICE)
            out   = model(x)
            v_loss    += criterion(out, y).item()
            v_correct += (out.argmax(1) == y).sum().item()
            v_total   += y.size(0)

    t_acc      = 100 * t_correct / t_total
    v_acc      = 100 * v_correct / v_total
    avg_v_loss = v_loss / len(val_loader)
    scheduler.step(avg_v_loss)

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  Train: {t_acc:.2f}%  Val: {v_acc:.2f}%  Loss: {avg_v_loss:.4f}")

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), MODEL_SAVE)
        print(f">> Best model saved  (Val Acc: {v_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break


model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
model.eval()
test_correct, test_total = 0, 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        test_correct += (out.argmax(1) == y).sum().item()
        test_total   += y.size(0)

print(f"\n========================================")
print(f"Training Complete!")
print(f"Best Validation Accuracy : {best_val_acc:.2f}%")
print(f"Test Accuracy            : {100 * test_correct / test_total:.2f}%")
print(f"Model saved  -> {MODEL_SAVE}")
print(f"Labels saved -> {LABEL_SAVE}")
print(f"========================================")