import torch
from torchvision import datasets

import utils


def main():
    data_dir, save_dir, device, arch, learning_rate, dropout, hidden_layer, epochs = utils.parse_train_args()

    train_transforms, validation_transforms, _ = utils.get_transforms()

    train_datasets = datasets.ImageFolder('./flowers/train', transform = train_transforms)
    validation_datasets = datasets.ImageFolder('./flowers/valid', transform = validation_transforms)

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(validation_datasets, batch_size = 32)

    model, criterion, optimizer = utils.create_network(device, arch, hidden_layer, dropout, learning_rate)
    utils.train(model, criterion, optimizer, epochs, train_loader, validation_loader, device)

    utils.save(model, optimizer, arch, epochs, save_dir)


if __name__== "__main__":
    main()