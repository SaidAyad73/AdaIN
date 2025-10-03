import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision import  models
from torchvision.transforms import ToTensor,Resize,Compose,RandomCrop,Lambda
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import random

class AdaIN(nn.Module):
  def __init__(self,eps):
    super().__init__()
    self.eps = eps
  def forward(self,x,y):
    if x.shape != y.shape:
      raise ValueError(f'x and y must have the same shape x.shape {x.shape} y.shape {y.shape}')

    normal_x = (x - x.mean(axis = (2,3),keepdim=True)) / (x.std(axis = (2,3),keepdim=True) + self.eps)
    y_mean = y.mean(axis = (2,3),keepdim=True) #for each sample and each channel
    y_std = y.std(axis = (2,3),keepdim=True) #for each sample and each channel
    return normal_x * y_std + y_mean
class ContentLoss(nn.Module):
  def __init__(self,encoder):
    super().__init__()
    self.encoder = encoder
  def forward(self,y_gen,t):
    y_gen = self.encoder(y_gen)
    return ((y_gen-t)**2).sum()

class StyleLoss(nn.Module):
  def __init__(self,encoder,to_layer:int):
    super().__init__()
    self.encoder = encoder
    self.layers_range = range(0,to_layer)
  def forward(self,y_gen,y_style):
    loss = 0.0
    for i,layer in enumerate(self.encoder):
      if i in self.layers_range:
        y_gen = layer(y_gen)
        y_style = layer(y_style)
        mean_g = y_gen.mean(dim=(2, 3))
        mean_s = y_style.mean(dim=(2, 3))
        std_g = y_gen.std(dim=(2, 3))
        std_s = y_style.std(dim=(2, 3))
        # loss += ((y_gen.mean(dim = (2,3)) - y_style.mean(dim = (2,3)))**2) + ((y_gen.std(dim = (2,3)) - y_style.std(dim = (2,3)))**2)
        loss += ((mean_g - mean_s) ** 2 + (std_g - std_s) ** 2).sum()
    return loss

class AdaINModel(nn.Module):
  """
  Note freeze encoder before starting
  """
  def __init__(self,encoder,decoder,ada):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.ada = ada
  def forward(self,x_content,x_style):
    x_content = self.encoder(x_content)
    x_style = self.encoder(x_style)
    ada_out = self.ada(x_content,x_style)
    x_gen = self.decoder(ada_out)

    return {
        'x_gen':x_gen,
        'ada_out':ada_out,
    }

class ImagesDataset(Dataset):
  def __init__(self,image_paths,transform=None):
    self.image_paths = image_paths
    self.transform = transform
  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, index):
    image = self.image_paths[index]
    image = Image.open(image).convert("RGB")
    if self.transform:
      image = Compose(self.transform)(image)
    return image

def get_vgg_encoder():
    vgg19 = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1)
    encoder = copy.deepcopy(vgg19.features[:22])
    return encoder

def get_decoder():
    decoder = nn.Sequential(
        nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=False),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=False),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=False),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=False),
        nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=False),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=False),
        nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(inplace=False),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=False),
        nn.Conv2d(64, 3, 3, 1, 1),
    )
    return decoder

def get_model(checkpoint_path = 'model_weightsV5.pth'):
    ada = AdaIN(.000001)
    encoder = get_vgg_encoder()
    decoder = get_decoder()
    model = AdaINModel(encoder,decoder,ada)
    for i, layer in enumerate(model.encoder):
        if isinstance(layer, nn.ReLU):
            model.encoder[i] = nn.ReLU(inplace=False)
    
    model.encoder.requires_grad_(False)
    model.ada.requires_grad_(False)

    model = torch.compile(model) # u forget this idiot
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    
    return model

def resizeWithAspectRatio(image):
  # print('shape',image.shape)
  width,height = image.shape[-1:-3:-1]
  short_d = -1 if width < height else -2
  long_d = -2 if short_d == -1 else -1
  # print(f'short {short_d}, long {long_d}')
  ratio = image.shape[long_d]/image.shape[short_d]
  long_new_size = int(ratio*image.shape[long_d])
  short_new_size = 256
  if long_new_size < 256:
    long_new_size = 256
    short_new_size = 256 # todo keep aspect ratio
  new_shape = list(image.shape[-2::1])
  new_shape[long_d] = long_new_size
  new_shape[short_d] = short_new_size
  # print(f'newshape {new_shape}')

  return Resize(new_shape)(image)


# def evaluate(data,y_styles,alpha):
#   model.eval()
#   loss = 0.0
#   con_loss = 0.0
#   sty_loss = 0.0
#   styles_iter = iter(y_styles)
#   with torch.no_grad():

#     for i,images in enumerate(data):
#       try:
#         y_style = next(styles_iter)
#         images = images.to(device)
#         y_style = y_style.to(device)

#         y = model(images,y_style)
#         y_gen,ada_out = y['x_gen'],y['ada_out']
#         content_loss = c_loss(y_gen,ada_out)
#         style_loss = s_loss(y_gen,y_style)
#         loss += content_loss.item() + alpha*style_loss.item()
#         con_loss += content_loss.item()
#         sty_loss += style_loss.item()

#       except StopIteration:
#         styles_iter = iter(y_styles)
#         continue
#     loss = loss/len(data)
#     con_loss = con_loss/len(data)
#     sty_loss = sty_loss/len(data)
#     # print(f'content loss {con_loss}, style loss {sty_loss}, total loss {loss}')
#     return con_loss,sty_loss,loss

