import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import DefectDataset, val_transform
from model import get_model


ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "data" / "processed" / "test"
MODEL_PATH = ROOT / "models" / "classification" / "resnet50_bottle.pth"

BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


test_dataset = DefectDataset(
    TEST_DIR,
    transform=val_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


model = get_model()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()


correct = 0
total = 0

class_correct = [0, 0]
class_total = [0, 0]


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):

            label = labels[i].item()

            class_total[label] += 1

            if predicted[i] == labels[i]:
                class_correct[label] += 1


accuracy = 100 * correct / total

print("\nTest Results")
print("----------------------")
print(f"Total Images: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2f}%")

print("\nClass-wise Accuracy")

for i, name in enumerate(["Good", "Defect"]):

    if class_total[i] > 0:
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print(f"{name}: {class_accuracy:.2f}%")