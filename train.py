import argparse
from accelerate import Accelerator
import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import cycle
from utils import *



def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument('--scheduler_startfactor', type=float, default=1.0)
    parser.add_argument('--scheduler_endfactor', type=float, default=1.0)
    parser.add_argument("--images_train_path", type=str, default=None)
    parser.add_argument("--images_val_path", type=str, default=None)
    parser.add_argument("--images_test_path", type=str, default=None)
    parser.add_argument("--styles_train_path", type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    # parser.add_argument("--save_path", type=str, default=f"model_weights_{timestamp}.pth")
    parser.add_argument('--broadcast_styles', type=bool, default=True) # not implemented will always be true
    parser.add_argument('--accumulative_steps', type=int, default=1)
    parser.add_argument('--eval_interval', type=float, default=0.25) # every epoch partially
    parser.add_argument('--save_interval', type=int, default=1) # every epoch
    parser.add_argument('--alpha', type=float, default=1.0)
    return parser.parse_args()

# !accelerate launch train.py --images_train_path /kaggle/working/image_train --images_val_path /kaggle/working/test_path --images_test_path /kaggle/working/test_path --styles_train_path /kaggle/working/style_path --epochs 10 --batch_size 32 --lr 0.0001 --checkpoint_path model_weightsV5.pth --eval_interval 0.25 --save_interval 1 --accumulative_steps 2 --alpha 10.0 --weight_decay 0.01 # --scheduler_startfactor 1.0 --scheduler_endfactor 0.01
model = None
device = None
c_loss = None
s_loss = None
def evaluate(data,y_styles,alpha):
  model.eval()
  loss = 0.0
  con_loss = 0.0
  sty_loss = 0.0
  styles_iter = iter(y_styles)
  with torch.no_grad():

    for i,images in enumerate(data):
      try:
        y_style = next(styles_iter)
        images = images.to(device)
        y_style = y_style.to(device)

        y = model(images,y_style)
        y_gen,ada_out = y['x_gen'],y['ada_out']
        content_loss = c_loss(y_gen,ada_out)
        style_loss = s_loss(y_gen,y_style)
        loss += content_loss.item() + alpha*style_loss.item()
        con_loss += content_loss.item()
        sty_loss += style_loss.item()

      except StopIteration:
        styles_iter = iter(y_styles)
        continue
    loss = loss/len(data)
    con_loss = con_loss/len(data)
    sty_loss = sty_loss/len(data)
    print(f'content loss {con_loss}, style loss {sty_loss}, total loss {loss}')
    return con_loss,sty_loss,loss

def main():
    print('new test')
    
    global model,c_loss,s_loss,device
    args = get_args()
    gradient_accumulation_steps = args.accumulative_steps
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    device = accelerator.device
    print(f"Using device: {device}")

    transform = [
        ToTensor(),
        Lambda(resizeWithAspectRatio),
        RandomCrop((224,224)),
    ]
    images_train_paths = [os.path.join(root, file) for root,dirs,files in os.walk(args.images_train_path) for file in files if file.endswith(('png','jpg','jpeg'))]
    images_val_paths = [os.path.join(root, file) for root,dirs,files in os.walk(args.images_val_path) for file in files if file.endswith(('png','jpg','jpeg'))]
    images_test_paths = [os.path.join(root, file) for root,dirs,files in os.walk(args.images_test_path) for file in files if file.endswith(('png','jpg','jpeg'))]
    styles_train_paths = [os.path.join(root, file) for root,dirs,files in os.walk(args.styles_train_path) for file in files if file.endswith(('png','jpg','jpeg'))]
    
    train_dataset = ImagesDataset(images_train_paths,transform=transform)
    val_dataset = ImagesDataset(images_val_paths,transform=transform)
    test_dataset = ImagesDataset(images_test_paths,transform=transform)
    styles_dataset = ImagesDataset(styles_train_paths,transform=transform) # using train images as styles
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    print(f'Style dataset size: {len(styles_dataset)}')

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4,pin_memory=True,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4,pin_memory=True,drop_last=True)
    styles_loader = DataLoader(styles_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    
    model = get_model(args.checkpoint_path)
    model.to(device)
    c_loss = ContentLoss(model.encoder).to(device)
    s_loss = StyleLoss(model.encoder,1).to(device)
    print(f'using model with checkpoint: {args.checkpoint_path}')
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    model, optimizer, train_loader, val_loader, test_loader, styles_loader,c_loss,s_loss = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, styles_loader,c_loss,s_loss
    )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.scheduler_startfactor, end_factor=args.scheduler_endfactor, total_iters=len(train_loader)*args.epochs)
    accelerator.register_for_checkpointing(lr_scheduler)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    print(f'Total training steps: {len(train_loader)*args.epochs} with {args.epochs} epochs')
    print(f'Batch size per device: {len(train_loader)}, effective batch size: {args.batch_size*accelerator.num_processes*args.accumulative_steps}')
    
    train_steps = len(train_loader)
    alpha = args.alpha
    con_loss,sty_loss,tot_loss = 0,0,0
    content_loss_history = []
    style_loss_history = []
    total_loss_history = []
    


    for epoch in range(args.epochs):
        if accelerator.is_main_process:  # single GPU or main process
            loop = tqdm.tqdm(zip(train_loader, cycle(styles_loader)),
                            unit='batch', total=len(train_loader))
        else:
            loop = zip(train_loader, cycle(styles_loader))  # no tqdm on other GPUs
        # loop = tqdm.tqdm(zip(train_loader, cycle(styles_loader)),unit='batch',total=len(train_loader)) # test this
        next_eval = int(len(train_loader)*args.eval_interval)
        for i,(images,styles) in enumerate(loop):
            with accelerator.accumulate(model):
                images = images.to(device)
                styles = styles.to(device)
                y = model(images,styles)
                y_gen,ada_out = y['x_gen'],y['ada_out']

                content_loss = c_loss(y_gen,ada_out)
                style_loss = s_loss(y_gen,styles)
                loss = content_loss + args.alpha*style_loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # if accelerator.is_main_process:
                #     content_loss_history.append(content_loss.item())
                #     style_loss_history.append(style_loss.item())
                #     total_loss_history.append(loss.item())
                
                if i % next_eval == 0 and i > 0:
                    con_loss,sty_loss,tot_loss = evaluate(val_loader,styles_loader,alpha)
                    model.train()
                    next_eval = next_eval + int(len(train_loader)*args.eval_interval)
                    # plt.clf()
                    # plt.plot(total_loss_history,label='total loss')
                    # plt.plot(content_loss_history,label='content loss')
                    # plt.plot(style_loss_history,label='style loss')
                    # plt.xlabel('iterations')
                
                # loop.set_postfix(content_loss = content_loss.item(),style_loss = f'{style_loss.item()*args.alpha}/ {style_loss.item()}',loss=loss.item(),lr=optimizer.param_groups[0]['lr'],val_con_loss=con_loss,val_sty_loss=sty_loss,val_tot_loss=tot_loss)
                if isinstance(loop,tqdm.tqdm):
                    loop.set_postfix_str(
                        f"content={content_loss.item():.4f} | "+
                        f"style={style_loss.item()*args.alpha:.4f}/{style_loss.item():.4f} | "+
                        f"loss={loss.item():.4f} | "+
                        f"lr={optimizer.param_groups[0]['lr']:.6f} | "+
                        f"val_con={con_loss:.4f} | "+
                        f"val_sty={sty_loss:.4f} | "+
                        f"val_tot={tot_loss:.4f}"
                    )
                    loop.set_description(f'Epoch [{epoch+1}/{args.epochs}]')
        
        if accelerator.is_main_process and (epoch+1) % args.save_interval == 0:
            timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
            save_path = f"model_weights_{timestamp}.pth"
            torch.save(model.state_dict(),save_path)
            print(f'Saved model checkpoint at {save_path}')
if __name__ == "__main__":
    main()
