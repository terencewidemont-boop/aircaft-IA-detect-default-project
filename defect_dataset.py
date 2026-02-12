
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class DefectDataset(Dataset):
    def __init__(self, root_dir, image_size=224):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.class_to_idx = {"intact": 0, "cracked": 1}

        for cls, idx in self.class_to_idx.items():
            for img_path in (self.root_dir / cls).glob("*"):
                self.samples.append((img_path, idx))

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
