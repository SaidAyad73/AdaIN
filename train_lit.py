# %%
# !wget -O train.py https://raw.githubusercontent.com/SaidAyad73/AdaIN/refs/heads/main/train.py
# !wget -O utils.py https://raw.githubusercontent.com/SaidAyad73/AdaIN/refs/heads/main/utils.py

# %%
# !pip install lightning 

# %%
# from utils import *
import utils

import lightning as L
import importlib
import torch
from torch import nn
from itertools import cycle


# %%
importlib.reload(utils)
import utils


# %%
BATCH_SIZE = 16
from torchvision.transforms import ToTensor, Lambda, RandomCrop
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import ImagesDataset
import os
transform = [
    ToTensor(),
    Lambda(utils.resizeWithAspectRatio),
    RandomCrop((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
]
images_train_path = '/kaggle/working/image_train'
images_test_path = '/kaggle/working/test_path'
images_val_path = '/kaggle/working/test_path'
styles_train_path = '/kaggle/working/style_path'
images_train_paths = [os.path.join(root, file) for root,dirs,files in os.walk(images_train_path) for file in files if file.endswith(('png','jpg','jpeg'))]
images_val_paths = [os.path.join(root, file) for root,dirs,files in os.walk(images_val_path) for file in files if file.endswith(('png','jpg','jpeg'))]
images_test_paths = [os.path.join(root, file) for root,dirs,files in os.walk(images_test_path) for file in files if file.endswith(('png','jpg','jpeg'))]
styles_train_paths = [os.path.join(root, file) for root,dirs,files in os.walk(styles_train_path) for file in files if file.endswith(('png','jpg','jpeg'))]

train_dataset = ImagesDataset(images_train_paths,transform=transform)
val_dataset = ImagesDataset(images_val_paths,transform=transform)
test_dataset = ImagesDataset(images_test_paths,transform=transform)
styles_dataset = ImagesDataset(styles_train_paths,transform=transform) # using train images as styles
print(f'Train dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')
print(f'Style dataset size: {len(styles_dataset)}')

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True,drop_last=True)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True,drop_last=True)
styles_loader = DataLoader(styles_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)

# %%
"""
todo
scheduling
loging
vis
"""
from lightning.pytorch.callbacks import TQDMProgressBar

class AdaINLitModule(L.LightningModule):
    def __init__(self,encoder,decoder,lr,scheduler = None,scheduler_args = None,content_weight = 1.0, style_weight = 1.0,checkpoint = None): #check if you need to compile
        """
        encoder: nn.Module not compiled
        decoder: nn.Module not compiled
        """
        super().__init__()
        for param in encoder.parameters():
            param.requires_grad = False
        for i, layer in enumerate(encoder):
            if isinstance(layer, nn.ReLU):
                encoder[i] = nn.ReLU(inplace=False)
        
        # encoder = torch.compile(encoder)
        # decoder = torch.compile(decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        self.model = utils.AdaINModel(encoder,decoder,utils.AdaIN(1e-5))
        # self.model = torch.compile(self.model)
        
        self.content_loss = utils.ContentLoss(encoder=self.encoder)
        self.style_loss = utils.StyleLoss(encoder=self.encoder)
        # self.content_loss = torch.compile(self.content_loss)
        # self.style_loss = torch.compile(self.style_loss)
        
        self.lr = lr
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args if scheduler_args is not None else {}
        self.train_loss = [] # content, style, total
        self.val_loss = [] # content, style, total

        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))
            
    def forward(self,content,style):
        return self.model(content,style)
    
    def training_step(self,batch,batch_idx):
        content,style = batch
        generated = self.model(content,style)
        y_gen,ada_out = generated['x_gen'],generated['ada_out']
        c_loss = self.content_loss(y_gen,ada_out)
        s_loss = self.style_loss(y_gen,style)
        loss = c_loss * self.content_weight + s_loss * self.style_weight
        self.train_loss.append((c_loss.item(),s_loss.item(),loss.item()))
        self.log('train_loss',loss,prog_bar=True)
        self.log(f'train_content_loss * {self.content_weight}',c_loss,prog_bar=True)
        self.log(f'train_style_loss * {self.style_weight}',s_loss,prog_bar=True)
        return loss
    
    def validation_step(self,batch,batch_idx):
        content,style = batch
        generated = self.model(content,style)
        y_gen,ada_out = generated['x_gen'],generated['ada_out']
        c_loss = self.content_loss(y_gen,ada_out)
        s_loss = self.style_loss(y_gen,style)
        loss = c_loss * self.content_weight + s_loss * self.style_weight
        self.val_loss.append((c_loss.item(),s_loss.item(),loss.item()))
        self.log('val_loss',loss,prog_bar=True)
        self.log(f'val_content_loss * {self.content_weight}',c_loss,prog_bar=True)
        self.log(f'val_style_loss * {self.style_weight}',s_loss,prog_bar=True)
        return loss
    
    def test_step(self,batch,batch_idx):
        content,style = batch
        generated = self.model(content,style)
        y_gen,ada_out = generated['x_gen'],generated['ada_out']
        c_loss = self.content_loss(y_gen,ada_out)
        s_loss = self.style_loss(y_gen,style)
        loss = c_loss * self.content_weight + s_loss * self.style_weight
        self.log('test_loss',loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        if self.scheduler is None:
            return optimizer
        return [optimizer],[self.scheduler(optimizer,**self.scheduler_args)]

    def setup(self, stage: str):
        if stage == "fit":
            print(f"Compiling model on rank {self.global_rank}...")
            self.model = torch.compile(self.model)
            self.content_loss = torch.compile(self.content_loss)
            self.style_loss = torch.compile(self.style_loss)
            

lr = 1e-3
start_factor = 0.001
end_factor = 1.0
scheduler = torch.optim.lr_scheduler.LinearLR
content_weight = 1.0
style_weight = 10.0
checkpoint =  None
# decoder = torch.compile(utils.get_decoder())
# encoder = torch.compile(utils.get_vgg_encoder())
encoder = utils.get_vgg_encoder()
decoder = utils.get_decoder()
# model = torch.compile(model)
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss',dirpath='./checkpoints',filename='adain-{epoch:02d}-{val_loss:.2f}',save_top_k=3,mode='min',save_last=True,every_n_train_steps=500)
trainer = L.Trainer(devices='auto',val_check_interval=int(.25*len(train_loader)),max_epochs=10,enable_progress_bar=True,accumulate_grad_batches=1,reload_dataloaders_every_n_epochs=1,callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=1)])
model = AdaINLitModule(encoder,decoder,lr,scheduler,scheduler_args={'start_factor': start_factor, 'end_factor': end_factor},content_weight=content_weight,style_weight=style_weight,checkpoint=checkpoint)


# %%
# del trainer, model,encoder,decoder,train_loader,val_loader,test_loader,styles_loader

# %%
len(train_loader),len(val_loader),len(test_loader),len(styles_loader)

# %%
t_loader = zip(train_loader, cycle(styles_loader))
v_loader = zip(val_loader, cycle(styles_loader))
trainer.fit(model,train_dataloaders=t_loader,val_dataloaders=v_loader)


