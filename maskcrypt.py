"""
MaskCrypt: Federated Learning with Selective Homomorphic Encryption.
"""
import maskcrypt_trainer
import maskcrypt_client
import maskcrypt_server

from maskcrypt_callbacks import MaskCryptCallback

from torch import nn

from plato.datasources import base
from functools import partial
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale
from datasets import load_dataset
from torch.utils.data import DataLoader

model = partial(
    nn.Sequential,
    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Increased to 16 channels
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Increased to 32 channels
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # Increased to 48 channels
    nn.ReLU(),
    nn.BatchNorm2d(48),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),  # Increased to 64 channels
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Flatten(),
    nn.Linear(64 * 6 * 6, 256),  # Adjusted input size for 96x96 images
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(256, 600),  # Updated to 600 classes for SOCOFing
)

pytorch_transforms = Compose(
    [Grayscale(num_output_channels=1), ToTensor(), Normalize((0.5,), (0.5,))]  # Normalization for grayscale images
)

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch


class DataSource(base.DataSource):
    """Source for SOCOFing dataset."""

    def __init__(self, **kwargs):
        super().__init__()
        # Load the dataset using `load_dataset`
        train_dict = load_dataset("dataloader", data_dir="altered")
        test_dict = load_dataset("dataloader", data_dir="validate")

        # Assign the datasets
        self.trainset = train_dict["train"].with_transform(apply_transforms)
        self.testset = test_dict["train"].with_transform(apply_transforms)


    def apply_transforms(self, example):
        """Apply transformations to the dataset."""
        example["image"] = pytorch_transforms(example["image"])
        return example

    def num_train_examples(self):
        """Return the number of training examples."""
        return len(self.trainset)

    def num_test_examples(self):
        """Return the number of validation examples."""
        return len(self.testset)


def main():
    """A Plato federated learning training session using selective homomorphic encryption."""
    trainer = maskcrypt_trainer.Trainer
    client = maskcrypt_client.Client(model=model, datasource=DataSource, trainer=trainer, callbacks=[MaskCryptCallback])
    server = maskcrypt_server.Server(model=model, datasource=DataSource, trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()