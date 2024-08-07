if not 'cracker_video' in dir():
    cracker_video=load_video('shutter_cracker.webm',use_cache=True)
    cracker_video=cracker_video/255
    cracker_colors=cracker_video[:,60:150,120:480].mean((1,2))
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

    def get_foreground(self):
        return torch.sigmoid(self._foreground)

    def get_alpha(self):
        return torch.sigmoid(self._alpha)

    def forward(self, backgrounds):
        alpha = self.get_alpha()[None, ...]  # Expanding dims for batch processing
        foreground = self.get_foreground()[None, ...]
        if toc()>.1:
            display_image(as_numpy_array(foreground[0])[:,:,0])
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
    optimizer = optim.SGD(composite_module.parameters(), lr=1000000,momentum=0)
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
        loss.backward()
        print(float(loss))
        return loss

    # Run the optimizer
    num_iter = 500
    for _ in tqdm(range(num_iter)):
        optimizer.step(closure)


    # Extract the parameters and form the final RGBA image
    foreground = composite_module.get_foreground().cpu().detach().numpy()
    alpha = composite_module.get_alpha().cpu().detach().numpy()
    rgba_image = np.concatenate([foreground, alpha], axis=-1)  # Concatenate to form the RGBA channels

    return rgba_image
# Note: Ensure the background_images and composite_images are scaled between 0 and 1 before passing them to this function.
#
#ans=get_rgba_overlay(cracker_background[::200],cracker_video[::200])
ans=get_rgba_overlay(cracker_background[::50],cracker_video[::50])
#ans=get_rgba_overlay(cracker_background[::10],cracker_video[::10])
#ans=get_rgba_overlay(cracker_background[:2],cracker_video[:2])
#######
rgb=ans[:,:,0]
alpha=ans[:,:,1]
rgb=as_rgb_image(rgb)
rgba=with_alpha_channel(rgb,alpha)
output=np.zeros_like(rgba)
output[180:260, 120:480, :]=rgba[180:260, 120:480, :]
save_openexr_image(output,'watermark.exr')
display_alpha_image(output,tile_size=100)