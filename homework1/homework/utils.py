import csv
import os

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        self.img_data = []
        with open(os.path.join(dataset_path, 'labels.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                print(row)
                img_filename, label, track = row
                img_path = os.path.join(dataset_path, img_filename)
                label_idx = LABEL_NAMES.index(label)
                self.img_data.append((img_path, label_idx))

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize image
            transforms.ToTensor(),  # Convert to tensor
        ])

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path, label = self.img_data[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label

def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
