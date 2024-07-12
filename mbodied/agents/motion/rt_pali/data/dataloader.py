import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from mbodied.agents.motion.rt_pali.action_tokenizer.action_tokenizer import ActionTokenizer
from mbodied.agents.motion.rt_pali.data.utils import scale_pose_data


class CustomDataset(Dataset):
    """Custom dataset for handling robot action data.

    Args:
        dataset (Dataset): The underlying dataset.
        statistics (dict): Precomputed statistics for normalization.
        transform (callable, optional): A function/transform to apply to the data.

    """
    def __init__(self, dataset, statistics, transform=None):
        self.dataset = dataset
        self.statistics = statistics
        self.transform = transform
        self.action_tokenizer = ActionTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        relative_action = item['relative_action']
        image = item['image']
        supervision = item['supervision']
        language_instruction = f"robot action {item['observation']['task']}"

        x = relative_action['pose']['x']
        y = relative_action['pose']['y']
        z = relative_action['pose']['z']
        roll = relative_action['pose']['roll']
        pitch = relative_action['pose']['pitch']
        yaw = relative_action['pose']['yaw']
        grasp = relative_action['grasp']

        # Collecting the values in a dictionary
        pose_data = {
            'terminated': supervision,
            'x': x,
            'y': y,
            'z': z,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'grasp': grasp
        }

        # Apply transformation (e.g., scale) to the pose data
        if self.transform:
            pose_data = self.transform(pose_data, self.statistics)

        # Convert PIL image to tensor
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # Convert HWC to CHW format
        
        tokenizer_tokens = self.action_tokenizer.tokenize(pose_data)

        return {
            'image': image_tensor,
            'language_instruction': language_instruction,
            'action_tokens': tokenizer_tokens
        }


def compute_statistics(dataset):
    all_data = {
        'x': [],
        'y': [],
        'z': [],
        'roll': [],
        'pitch': [],
        'yaw': [],
        'grasp': []
    }

    for idx in tqdm(range(len(dataset)), desc="Computing Statistics", unit="sample"):
        pose_data = dataset[idx]['relative_action']['pose']
        grasp_data = dataset[idx]['relative_action']['grasp']

        for key in all_data:
            if key == 'grasp':
                all_data[key].append(grasp_data)
            else:
                all_data[key].append(pose_data[key])

    statistics = {}
    for key, values in all_data.items():
        values = np.array(values)
        statistics[key] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'std': np.std(values)
        }

    return statistics

# Use in PyTorch Lightning


class MyDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.batch_size = batch_size
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            )


def main():
    # Load the dataset
    ds = load_dataset("mbodiai/oxe_utokyo_xarm_pick_place")['default']

    # Take a subset of the dataset
    subset_size = 10  # Define the size of the subset you want to take
    ds = ds.select(range(subset_size))
    # Compute the statistics first
    statistics = compute_statistics(ds)

    # Create the CustomDataset with precomputed statistics
    dataset = CustomDataset(ds, statistics, transform=scale_pose_data)

    data_module = MyDataModule(dataset)

    # Compute and print statistics (precomputed)
    for key, stats in statistics.items():
        print(
            f"{key} - min: {stats['min']}, max: {stats['max']}, mean: {stats['mean']}, std: {stats['std']}")

    # Example of processing some data
    for batch in data_module.train_dataloader():
        print("", batch)
        descaled_input = {k: dataset.descale(v, k) for k, v in batch.items()}
        print("Descaled Input: ", descaled_input)
        break  # Process only a single batch


if __name__ == "__main__":
    main()

