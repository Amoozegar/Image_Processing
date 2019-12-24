import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import argparse
from collections import OrderedDict 

##.................................Python Functions...............................###
def dataLoader(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {
    'train_transforms' : transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),


    'validation_transforms' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]),

    'test_transforms' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    }
    # TODO: Load the datasets with ImageFolder 
    image_datasets = {
        'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
        'validation_data' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation_transforms']),
        'test_data': datasets.ImageFolder(test_dir, transform=data_transforms['test_transforms'])
    }


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
    'trainloader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=50, shuffle=True),
    'validationloader': torch.utils.data.DataLoader(image_datasets['validation_data'], batch_size=50),
    'testloader': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=50)
    }
    return dataloaders


def model_load(arch):
    if arch == 'vgg16': 
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
    return model

def classifier(model, hidden_units,num_in_features, dropout):
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_in_features, hidden_units, bias=True)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=dropout)),
                              ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model

# Putting the above into functions, so they can be used later
def do_deep_learning(model, dataloaders, epochs, print_every, criterion, optimizer, GPU):
    lr = 0.001
    
    epochs = 3
    print_every = 5
    steps = 0
    loss_show=[]

    # change to cuda
    if torch.cuda.is_available() and  GPU==True:
        model.to('cuda')
 

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders['trainloader']):
            steps += 1
            if torch.cuda.is_available() and  GPU==True:
                inputs,labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0


                for ii, (inputs2,labels2) in enumerate(dataloaders['validationloader']):
                    optimizer.zero_grad()
                   
                    inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda')
                    model.to('cuda')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vlost += criterion(outputs,labels2)

                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                vlost = vlost / len(dataloaders['validationloader'])
                accuracy = accuracy /len(dataloaders['validationloader'])


                print("Epoch: {}/{} | ".format(e+1, epochs),
                  "Training Loss: {:.4f} | ".format(running_loss/print_every),
                  "Validation Loss: {:.4f} | ".format(vlost),
                  "Validation Accuracy: {:.4f}".format(accuracy))

                running_loss = 0
    
def check_accuracy_on_test(testloader, GPU, model):    
    correct = 0
    total = 0
    if torch.cuda.is_available() and  GPU==True:
        model.to('cuda')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available() and  GPU==True:
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
 
def save_checkpoint(model, epochs, dataloaders, optimizer, structure):
    model.class_to_idx = dataloaders['trainloader'].dataset.class_to_idx
    model.epochs = epochs
    checkpoint = { 'structure' :structure,
                    'classifier': model.classifier,
                     'state_dict': model.state_dict(),
                     'optimizer_dict':optimizer.state_dict(),
                     'class_to_idx': model.class_to_idx,
                     'epoch': model.epochs,
                     'batch_size': dataloaders['trainloader'].batch_size}
    torch.save(checkpoint, 'checkpoint.pth')
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['structure'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model






def main():
    # Define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Path to dataset ',action="store", default="ImageClassifier/flowers")
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available', default=True)
    parser.add_argument('--epochs', help='Number of epochs', action="store", type=int, default=1)
    parser.add_argument('--dropout', help = "probability of drop out", action = "store", default = 0.5)
    parser.add_argument('--arch', help='Model architecture', action="store", default="vgg16", type = str)
    parser.add_argument('--learning_rate', help="learning rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', help='Number of hidden units', dest="hidden_units", action="store", default=1024)
    parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file', default="ImageClassifier/checkpoint.pth")
    
    args = parser.parse_args()
    directory = args.data_dir
    checkpnt = args.checkpoint
    lr = args.learning_rate
    structure = args.arch
    dropout = args.dropout
    hidden_layer = args.hidden_units
    GPU = args.gpu
    epochs = args.epochs
    
    print ('GPU', GPU)
    
    print ('loading the data')
    dataloaders = dataLoader(directory)
    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print ('loading model')
    
    
    if (structure=="vgg16"):
        num_in_features = model.classifier[0].in_features
    elif ( structure=="densenet"):
        num_in_features = model.fc.in_features
    model = classifier(model, hidden_layer,num_in_features, dropout)
    print_every = 5
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )
    print ('start deep learning')
    do_deep_learning(model, dataloaders, epochs, print_every, criterion, optimizer, GPU)
    check_accuracy_on_test(dataloaders['testloader'], GPU, model)
    save_checkpoint(model, epochs,dataloaders, optimizer, structure)

     
if __name__ == '__main__': main()    
    


