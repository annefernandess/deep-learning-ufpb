from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class FlattenImages(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self.dataset[index]
        return image.flatten(), label


def load_cifar10(
    data_dir: str | Path | None = None,
) -> tuple[Dataset, Dataset, Dataset, Dataset]:
    root = (
        Path(data_dir)
        if data_dir is not None
        else Path(__file__).resolve().parents[1] / "data"
    )
    transform = ToTensor()

    train_cnn = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_cnn = CIFAR10(root=root, train=False, download=True, transform=transform)

    return (
        train_cnn,
        test_cnn,
        FlattenImages(train_cnn),
        FlattenImages(test_cnn),
    )
