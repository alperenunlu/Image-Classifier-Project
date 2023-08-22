import torch
from torch import nn
from torchvision import transforms, datasets
from torchvision.models import resnet50, vgg16

import numpy as np

import argparse


def get_train_args():
    parser = argparse.ArgumentParser(description="Train a model to classify images")

    parser.add_argument("data_dir", type=str, help="Directory containing the data")
    parser.add_argument(
        "--save_dir", type=str, help="Directory to save the model to", default="."
    )
    parser.add_argument(
        "--arch", type=str, help="VGG16 or ResNet50", default="resnet50"
    )
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate", default=0.003
    )
    parser.add_argument(
        "--hidden_units", type=int, help="Number of hidden units", default=512
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=20)
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    return args


def get_predict_args():
    parser = argparse.ArgumentParser(description="Predict the class of an image")

    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint")
    parser.add_argument(
        "--top_k", type=int, help="Return top K most likely classes", default=5
    )
    parser.add_argument(
        "--category_names",
        type=str,
        help="Path to the category names file",
        default="cat_to_name.json",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--previous_model",
        action="store_true",
        help="Previous model trained on Notebook",
    )

    args = parser.parse_args()

    return args


def get_data(data_dir):
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
    }

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"], batch_size=64, shuffle=True
        ),
        "valid": torch.utils.data.DataLoader(
            image_datasets["valid"], batch_size=64, shuffle=True
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"], batch_size=64, shuffle=True
        ),
    }

    return dataloaders, image_datasets, data_transforms


def get_model(arch, hidden_units):
    if arch.lower() == "vgg16":
        model = vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1),
        )

    elif arch.lower() == "resnet50":
        model = resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1),
        )

    else:
        raise RuntimeError(f"Unknown model architecture: {arch}")

    return model


def get_predict_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model = get_model(checkpoint["arch"], checkpoint["hidden_units"])
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def load_checkpoint(filepath, device=torch.device("cuda")):
    checkpoint = torch.load(filepath, device)
    model = resnet50()
    model.fc = checkpoint["fc"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    criterion = checkpoint["criterion"]
    optimizer = checkpoint["optimizer"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = checkpoint["scheduler"]
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epochs = checkpoint["epochs"]

    return model


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """

    # TODO: Process a PIL image for use in a PyTorch model

    # Resizing
    width, height = image.size

    if width > height:
        new_width = round(width * (256 / height))
        new_height = 256
    else:
        new_width = 256
        new_height = round(height * (256 / width))

    image = image.resize((new_width, new_height))

    # Cropping
    width, height = image.size

    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2

    image = image.crop((left, top, right, bottom))

    # Normalizing
    np_image = np.array(image) / 255.0

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - means) / stds

    # Transposing
    np_image = np_image.transpose((2, 0, 1))

    return np_image
