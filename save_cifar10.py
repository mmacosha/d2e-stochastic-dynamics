# save_cifar10_by_class.py

import os
from PIL import Image
import torchvision
from torchvision import transforms

from tqdm.auto import tqdm


def main():
    # Output directory
    output_dir = "/workspace/writeable/gen_images/cifar10_by_class"
    os.makedirs(output_dir, exist_ok=True)

    # Download CIFAR-10 train and test datasets
    transform = transforms.ToPILImage()
    trainset = torchvision.datasets.CIFAR10(root='/workspace/writeable/cifar10', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='/workspace/writeable/cifar10', train=False, download=True)

    # Get class names
    class_names = trainset.classes
    
    class_counters = {name: 0 for name in class_names}

    # Create class folders
    for name in class_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    def save_images(dataset, split, class_counters):
        # Keep a counter for each class to avoid filename collisions
        for idx, (img, label) in tqdm(enumerate(dataset), total=len(dataset)):
            class_name = class_names[label]
            # Convert to PIL Image if needed
            if not isinstance(img, Image.Image):
                img = transform(img)
            # Filename: zero-padded index, e.g. 000123.png
            count = class_counters[class_name]
            filename = f"{count:06d}.png"
            save_path = os.path.join(output_dir, class_name, filename)
            img.save(save_path)
            class_counters[class_name] += 1

    print("Saving training images...")
    save_images(trainset, "train", class_counters)
    print("Saving test images...")
    save_images(testset, "test", class_counters)
    print("Done.")


if __name__ == "__main__":
    main()
