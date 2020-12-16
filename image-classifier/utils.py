import argparse

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, models, transforms

from PIL import Image


arch_layers = {
    'alexnet': 9216,
    'densenet121': 1024,
    'resnet18': 512,
    'vgg16': 25088
}


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',                                action = 'store', default = './flowers')
    parser.add_argument('--save_dir',      dest = 'save_dir',      action = 'store', default = '.')
    parser.add_argument('--gpu',           dest = 'gpu',           action = 'store_true')
    parser.add_argument('--arch',          dest = 'arch',          action = 'store', default = 'vgg16', choices = ['alexnet', 'densenet121', 'resnet18', 'vgg16'])
    parser.add_argument('--learning_rate', dest = 'learning_rate', action = 'store', default = 0.001, type = float)
    parser.add_argument('--dropout',       dest = 'dropout',       action = 'store', default = 0.5,   type = float)
    parser.add_argument('--hidden_units',  dest = 'hidden_units',  action = 'store', default = 512,   type = int)
    parser.add_argument('--epochs',        dest = 'epochs',        action = 'store', default = 5,     type = int)

    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    device = torch.device('cuda' if args.gpu else 'cpu')
    arch = args.arch
    learning_rate = args.learning_rate
    dropout = args.dropout
    hidden_layer = args.hidden_units
    epochs = args.epochs

    return data_dir, save_dir, device, arch, learning_rate, dropout, hidden_layer, epochs


def parse_predict_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint',                                action = 'store', default = 'checkpoint.pth')
    parser.add_argument('--gpu',            dest = 'gpu',            action = 'store_true')
    parser.add_argument('--category_names', dest = 'category_names', action = 'store', default = 'cat_to_name.json')
    parser.add_argument('--filepath',       dest = 'filepath',       action = 'store', default = 'flowers/test/28/image_05230.jpg')
    parser.add_argument('--top_k',          dest = 'top_k',          action = 'store', default = 3, type = int)

    args = parser.parse_args()

    checkpoint = args.checkpoint
    device = torch.device('cuda' if args.gpu else 'cpu')
    category_names = args.category_names
    filepath = args.filepath
    top_k = args.top_k

    return checkpoint, device, category_names, filepath, top_k


def get_transforms():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    return train_transforms, validation_transforms, test_transforms


def create_network(device, arch, hidden_layer, dropout, learning_rate):
    model = getattr(models, arch)(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(arch_layers[arch], hidden_layer),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_layer, hidden_layer),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_layer, 102),
                                     nn.LogSoftmax(dim = 1))

    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    return model, criterion, optimizer


def test(model, criterion, loader):
    loss_count = 0
    accuracy = 0

    model.eval()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)

            probabilities = torch.exp(outputs)
            top_p, top_class = probabilities.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

            loss = criterion(outputs, labels)
            loss_count += loss.item()

    model.train()

    return loss_count, accuracy / len(validation_loader)


def train(model, criterion, optimizer, epochs, train_loader, validation_loader, device):
    train_losses, validation_losses = [], []

    for epoch in range(epochs):
        train_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        else:
            validation_loss, accuracy = test(model, criterion, validation_loader)
            print(f'Epoch #{epoch + 1}')
            print(f'Train Loss: {train_loss}')
            print(f'Validation Loss: {validation_loss}')
            print(f'Accuracy: {(accuracy * 100):.2f}%')

            train_losses.append(train_loss / len(train_loader))
            validation_losses.append(validation_loss / len(validation_loader))

    return train_losses, validation_losses


def plot_losses(train_losses, validation_losses):
    _, ax = plt.subplots()

    ax.plot(train_losses, 'r', label = 'Train')
    ax.plot(validation_losses, 'b--', label = 'Validation')
    ax.legend(loc = 'upper center', shadow = True)


def save(model, optimizer, arch, epochs, save_dir):
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {'arch': arch,
                  'hidden_layer': hidden_layer,
                  'dropout': dropout,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier,
                  'state': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')


def load(save_dir, device):
    checkpoint = torch.load(f'{save_dir}/checkpoint.pth')

    arch = checkpoint['arch']
    hidden_layer = checkpoint['hidden_layer']
    dropout = checkpoint['dropout']
    learning_rate = checkpoint['learning_rate']

    model, criterion, optimizer = create_network(device, arch, hidden_layer, dropout, learning_rate)

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    epochs = checkpoint['epochs']

    return model, criterion, optimizer, epochs


def process_image(image):
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])

    return process(Image.open(image))


def predict(image_path, model, topk):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        image = process_image(image_path).unsqueeze_(0).type(torch.cuda.FloatTensor)
        image.to(device)

        output = model.forward(image)
        probability = F.softmax(output, dim = 1)

    return probability.topk(topk, dim = 1)