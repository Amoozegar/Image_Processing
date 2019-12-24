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
import json


###..................Functions.......................
def load_checkpoint(filepath):
    checkpoint = torch.load('checkpoint.pth')
    model = getattr(models, checkpoint['structure'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_pil = Image.open(image)
   
    width, height = img_pil.size
    
    size = 256, 256
    if width > height:
        ratio = float(width) / float(height)
        newheight = ratio * size[0]
        img_pil = img_pil.resize((size[0], int(round(newheight))), Image.ANTIALIAS)
    else:
        ratio = float(height) / float(width)
        newwidth = ratio * size[0]
        img_pil = img_pil.resize((int(round(newwidth)), size[0]), Image.ANTIALIAS)

    adjustments = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])    
    img_tensor = adjustments(img_pil)
    
    return img_tensor


def predict(image_path, model, cat_to_name, GPU, topk=5):   
    if (GPU==False):
        model.to('cpu')
    else:
        model.to('gpu')
    img = process_image(image_path)
    img.unsqueeze_(0)
    top_class=[]
    with torch.no_grad():
        logits = model.forward(img)
    
    pobabilities = F.softmax(logits,dim=1)
    mylist = pobabilities.topk(topk)[1].tolist()
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    for x in mylist[0]:
        top_class.append(idx_to_class[x])
    
    top_probability = pobabilities.topk(topk)[0][0]
        
    top_flowers = [cat_to_name[lab] for lab in top_class]
    return top_probability, top_class,top_flowers

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', default='ImageClassifier/flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str,help='input to predict')
    parser.add_argument('--dir', action="store",dest="data_dir", default="ImageClassifier/flowers",help='Path to dataset')
    parser.add_argument('--checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str,help='Save trained model checkpoint to file')
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', help='name of category of flowers')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available', default=False)

    args = parser.parse_args()
    directory = args.data_dir
    img = args.input
    checkpnt = args.checkpoint
    top_k = args.top_k
    GPU = args.gpu
    
    
    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint(checkpnt) 
    img_tensor = process_image(img)
    top_probability, top_class,top_flowers = predict(img, model, cat_to_name, GPU, top_k)
    print ('top probability :', top_probability)
    print ('top class :', top_class)
    print ('top flowers :', top_flowers)
                        
if __name__ == '__main__': main()                          
                        
                       
                        
                    
    
    
    