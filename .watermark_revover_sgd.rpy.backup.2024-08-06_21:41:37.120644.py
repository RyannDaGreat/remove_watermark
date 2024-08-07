import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device=select_torch_device(prefer_used=True)

class CompositeImage(nn.Module):
    def __init__(self, H, W):
        super(CompositeImage, self).__init__()
        # Initialize the foreground and alpha mask with random values and set them as parameters
        self.foreground = nn.Parameter(torch.randn(H, W, 3))
        self.alpha = nn.Parameter(torch.randn(H, W, 1))

    def forward(self, backgrounds):
        # Expand alpha and foreground to batch size for broadcasting
        alpha = self.alpha[None, ...]
        foreground = self.foreground[None, ...]
        # Compute the composite image
        composite_image = alpha * foreground + (1 - alpha) * backgrounds
        return composite_image

def get_rgba_overlay(background_images, composite_images):
    B, H, W, _ = background_images.shape
    background_images = torch.tensor(background_images, dtype=torch.float32).to(device)
    composite_images = torch.tensor(composite_images, dtype=torch.float32).to(device)

    # Initialize the composite module
    composite_module = CompositeImage(H, W).to(device)
    
    # Define the optimizer, using LBFGS which is often used for small to medium-sized optimization problems
    optimizer = optim.LBFGS(composite_module.parameters(), lr=1.0, max_iter=2)
    #optimizer = optim.SGD(composite_module.parameters(),lr=1)

    # Define the loss function
    criterion = nn.MSELoss()

    def closure():
        optimizer.zero_grad()
        output = composite_module(background_images)
        loss = criterion(output, composite_images)
        loss.backward()
        print(float(loss))
        return loss

    # Run the optimizer
    num_iter = 1000
    for _ in tqdm(range(num_iter)):
        optimizer.step(closure)

    # Extract the parameters and form the final RGBA image
    foreground = composite_module.foreground.cpu().detach().numpy()
    alpha = composite_module.alpha.cpu().detach().numpy()
    rgba_image = np.concatenate([foreground, alpha], axis=-1)  # Concatenate to form the RGBA channels

    return rgba_image

# Note: Ensure the background_images and composite_images are scaled between 0 and 1 before passing them to this function.

get_rgba_overlay(cracker_background[::200],cracker_video[::200])