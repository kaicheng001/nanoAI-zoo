from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_repo_root() -> Path:
    # Return the absolute path to the project root directory.
    # Assuming this file is located in "<repo_root>/utils/",
    # the root is one level above (parents[1]).
    return Path(__file__).resolve().parents[1]


def resolve_path(path_str):
    path_obj = Path(path_str)
    return (
        path_obj if path_obj.is_absolute() else (get_repo_root() / path_obj).resolve()
    )


def load_single_image(image_path, image_size=(224, 224)):
    """Load an image from a file path or a PIL.Image, then preprocess."""
    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        path = resolve_path(image_path)
        image = Image.open(path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def load_batch_images(
    image_path_list, image_size=(224, 224), expected_batch_size=None
) -> torch.Tensor:
    image_tensors = [load_single_image(path, image_size) for path in image_path_list]
    batch_tensor = torch.cat(image_tensors, dim=0)

    if expected_batch_size is not None:
        assert batch_tensor.size(0) == expected_batch_size, (
            f"Expected batch size {expected_batch_size}, but got {batch_tensor.size(0)}"
        )
    return batch_tensor


if __name__ == "__main__":
    # test single image loading
    single_image_tensor = load_single_image("./assets/img.jpg")

    print("\n--- Single Image Test ---")
    print("Single image tensor shape:")
    print(single_image_tensor.shape)
    print(f"min: {single_image_tensor.min():.3f}, max: {single_image_tensor.max():.3f}")

    # test CIFAR10 batch loading into <repo>/data
    repo_root = get_repo_root()
    data_dir = repo_root / "data"
    dataset = CIFAR10(root=str(data_dir), train=False, download=True)

    temp_dir = repo_root / "temp_images"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_image_paths = []

    for index in range(8):
        image_path = temp_dir / f"img_{index}.png"
        Image.fromarray(dataset.data[index]).save(image_path)
        temp_image_paths.append(image_path)

    batch_tensor = load_batch_images(
        temp_image_paths, image_size=(224, 224), expected_batch_size=8
    )

    print("\n--- CIFAR10 Batch Test ---")
    print("Batch image tensor shape:")
    print(batch_tensor.shape)

    for path in temp_image_paths:
        path.unlink()
    temp_dir.rmdir()
