import shutil
import random
from pathlib import Path

def split_dataset(raw_dir, output_dir, ratios=(0.7, 0.15, 0.15)):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    for class_dir in raw_dir.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob('*'))
        random.shuffle(images)

        n = len(images)
        train_end = int(ratios[0] * n)
        val_end = train_end + int(ratios[1] * n)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split_name, files in splits.items():
            split_path = output_dir / split_name / class_dir.name
            split_path.mkdir(parents=True, exist_ok=True)

            for file in files:
                shutil.copy(file, split_path / file.name)
