import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import SegmentationDataset, image_transform, mask_transform
from unet import UNet


ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "data" / "processed" / "test"
MODEL_PATH = ROOT / "models" / "segmentation" / "unet_bottle.pth"

BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


test_dataset = SegmentationDataset(
    TEST_DIR,
    image_transform,
    mask_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


model = UNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


total_iou = 0
total_dice = 0
count = 0


with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        predictions = torch.sigmoid(outputs)

        predictions = (predictions > 0.5).float()

        intersection = (predictions * masks).sum(
            dim=(1, 2, 3)
        )

        union = (
            predictions + masks - predictions * masks
        ).sum(dim=(1, 2, 3))

        iou = (
            (intersection + 1e-7) /
            (union + 1e-7)
        )

        dice = (
            (2 * intersection + 1e-7) /
            (
                predictions.sum(dim=(1, 2, 3)) +
                masks.sum(dim=(1, 2, 3)) +
                1e-7
            )
        )

        total_iou += iou.sum().item()
        total_dice += dice.sum().item()

        count += images.size(0)


mean_iou = total_iou / count
mean_dice = total_dice / count


print("\nSegmentation Results")
print("----------------------")
print(f"Total Images: {count}")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Mean Dice: {mean_dice:.4f}")