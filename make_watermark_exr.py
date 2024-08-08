from rp import *

if not 'cracker_video' in dir():
    #Use one of these
    cracker_video=load_video('watermark_extraction/shutter_cracker.webm',use_cache=True)
    cracker_video=as_numpy_array(load_images('watermark_extraction',use_cache=True))
    
    cracker_video=cracker_video/255
    cracker_colors=cracker_video[:,170:180,120:480]
    #display_image_slideshow(cracker_colors)
    cracker_colors=cracker_colors.mean((1,2))
    cracker_background = cracker_video + 0
    cracker_background[:, 180:260, 120:480, :] = cracker_colors[:,None,None,:]
    
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Optional: Function to select a device
def select_torch_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = select_torch_device(prefer_gpu=True)

class CompositeImage(nn.Module):
    def __init__(self, H, W):
        super(CompositeImage, self).__init__()

        # self._foreground = nn.Parameter(torch.zeros(H, W, 1))
        # self._alpha = nn.Parameter(torch.zeros(H, W, 1))

        self._foreground = nn.Parameter(torch.rand(H, W, 1))
        self._alpha = nn.Parameter(torch.rand(H, W, 1))
        self.do_foreground_sigmoid=True
        self.do_alpha_sigmoid=True

    def get_foreground(self):
        if self.do_foreground_sigmoid:
            return torch.sigmoid(self._foreground)
        else:
            return self._foreground.clamp(0,1)

    def get_alpha(self):
        if self.do_alpha_sigmoid:
            return torch.sigmoid(self._alpha)
        else:
            #out=torch.sigmoid(self._alpha)
            #out=(out-.5)*1.1+.5
            out=torch.tanh(self._alpha)/2+.5
            #out=out.clamp(0,1)
            return out            
            #return self._alpha.clamp(-1,1)/2+.5

    def forward(self, backgrounds):
        alpha = self.get_alpha()[None, ...]  # Expanding dims for batch processing
        foreground = self.get_foreground()[None, ...]
        if toc()>.1:
            display_image(
                vertically_concatenated_images(
                    as_numpy_array(foreground[0])[:, :, 0],
                    full_range(as_numpy_array(alpha[0])[:, :, 0]),
                )
            )
            tic()
        composite_image = alpha * foreground + (1 - alpha) * backgrounds
        return composite_image

def get_rgba_overlay(background_images, composite_images):
    B, H, W, _ = background_images.shape
    background_images = torch.tensor(background_images, dtype=torch.float32).to(device)
    composite_images = torch.tensor(composite_images, dtype=torch.float32).to(device)

    # Initialize the composite module
    composite_module = CompositeImage(H, W).to(device)
    
    # Define the optimizer, using LBFGS for optimization
    #optimizer = optim.LBFGS(composite_module.parameters(), lr=1.0, max_iter=20)
    optimizer = optim.SGD(composite_module.parameters(), lr=1000000,momentum=0,weight_decay=0)
    #optimizer = optim.Adam(composite_module.parameters(),lr=.01)

    # Define the loss function
    criterion = nn.MSELoss()

    # # Run the optimizer
    # num_iter = 500
    # for _ in tqdm(range(num_iter)):
    #     optimizer.zero_grad()
    #     output = composite_module(background_images)
    #     loss = criterion(output, composite_images)
    #     loss.backward()
    #     optimizer.step()


    def closure():
        optimizer.zero_grad()
        output = composite_module(background_images)
        loss = criterion(output, composite_images)
        #loss += composite_module.get_alpha().mean()*.1
        loss.backward()
        
        print(float(loss))
        return loss

    # Run the optimizer
    num_iter = 30
    for _ in tqdm(range(num_iter)):
        optimizer.step(closure)

    # Run the optimizer with full-range colors
    print("PHASE 2 OPTIMIZATION")
    sleep(1)
    composite_module.do_foreground_sigmoid=False
    composite_module._foreground.data[180:260, 120:480, :] = full_range(composite_module.get_foreground()[180:260, 120:480, :]) 
    num_iter = 300
    for _ in tqdm(range(num_iter)):  
        composite_module._foreground.data[180:260, 120:480, :] = full_range(composite_module.get_foreground()[180:260, 120:480, :]) 
        optimizer.step(closure)

    num_iter = 1000
    composite_module.do_alpha_sigmoid=False
    for _ in tqdm(range(num_iter)):  
        composite_module._foreground.data[180:260, 120:480, :] = full_range(composite_module.get_foreground()[180:260, 120:480, :]) 
        optimizer.step(closure)

    # Extract the parameters and form the final RGBA image
    foreground = composite_module.get_foreground().cpu().detach().numpy()
    alpha = composite_module.get_alpha().cpu().detach().numpy()
    rgba_image = np.concatenate([foreground, alpha], axis=-1)  # Concatenate to form the RGBA channels

    return rgba_image
# Note: Ensure the background_images and composite_images are scaled between 0 and 1 before passing them to this function.
#
#ans=get_rgba_overlay(cracker_background[::200],cracker_video[::200])
#ans=get_rgba_overlay(cracker_background[::50],cracker_video[::50])
#ans=get_rgba_overlay(cracker_background[::10],cracker_video[::10])
ans=get_rgba_overlay(cracker_background,cracker_video)
#ans=get_rgba_overlay(cracker_background[:2],cracker_video[:2])
#######
rgb=ans[:,:,0]+0
alpha=ans[:,:,1]+0
#alpha[alpha<.0205]=0#Thesholding it because of innacuracies...

rgb=as_rgb_image(rgb)
rgba=with_alpha_channel(rgb,alpha)
output=np.zeros_like(rgba)
output[180:260, 120:480, :]=rgba[180:260, 120:480, :]
#output=rgba


output=crop_image_zeros(output)
output[:,:,:3]=full_range(output[:,:,:3]) #Make higher contrast 
save_openexr_image(output,'watermark.exr')


display_alpha_image(output,tile_size=100)
display_image(full_range(get_alpha_channel(output)))
