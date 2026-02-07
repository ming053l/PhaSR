import torch
import os
import torchvision.transforms as transforms


class Augment_RGB_torch:
### rotate and flip
    def __init__(self, rotate=0):
        self.rotate = rotate
        pass
    def transform0(self, torch_tensor):
        return torch_tensor  

    def transform1(self, torch_tensor):
        H, W = torch_tensor.shape[1], torch_tensor.shape[2]
        train_transform = transforms.Compose([
        transforms.RandomRotation((self.rotate,self.rotate), interpolation=transforms.InterpolationMode.BILINEAR, expand=False),
        transforms.Resize((int(H * 1.3), int(W * 1.3)), antialias=True),
        # CenterCrop，if the size is larger than the original size, the excess will be filled with black
        transforms.CenterCrop([H, W])
        ])
        return train_transform(torch_tensor)

    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform8(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


