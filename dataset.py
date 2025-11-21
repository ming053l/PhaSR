import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import load_normal, load_img, depthToPoint, process_normal, load_depth, Augment_RGB_torch
import torch.nn.functional as F
import random
import torchvision.transforms as transforms


augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None, debug=False, ps=256):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'shadow_free'
        input_dir = 'origin'
        depth_dir = 'depth'
        normal_dir = 'normal'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))       # shadow free
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))    # origin
        depth_files = sorted(os.listdir(os.path.join(rgb_dir, depth_dir)))    # depth
        normal_files = sorted(os.listdir(os.path.join(rgb_dir, normal_dir)))  # noraml map

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files]        # shadow free
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files]     # origin
        self.depth_filenames = [os.path.join(rgb_dir, depth_dir, x) for x in depth_files]     # depth
        self.normal_filenames = [os.path.join(rgb_dir, normal_dir, x) for x in normal_files]  # noraml map
        self.img_options = img_options

        self.ps = ps

        if debug:
            self.tar_size = 100 
        else:
            self.tar_size = len(self.noisy_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        
        clean = np.float32(load_img(self.clean_filenames[tar_index]))
        noisy = np.float32(load_img(self.noisy_filenames[tar_index]))
        depth = np.float32(load_depth(self.depth_filenames[tar_index]))
        normal = np.float32(load_normal(self.normal_filenames[tar_index]))

        point = depthToPoint(60, depth)  

        normal = process_normal(normal)

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
        depth = torch.from_numpy(depth)
        point = torch.from_numpy(point)
        normal =  torch.from_numpy(normal)

        point = point / (2 * point[:,:,2].mean())

        clean  = clean.permute(2,0,1)
        noisy  = noisy.permute(2,0,1)
        point  = point.permute(2,0,1)
        normal  = normal.permute(2,0,1)

        H, W, _ = noisy.shape

        if H > 1000 or W > 1000:
            ps = self.ps # 目標尺寸
            img_transform = transforms.Resize((ps, ps), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            
            
            geo_transform = transforms.Resize((ps, ps), interpolation=transforms.InterpolationMode.NEAREST)

            clean = img_transform(clean)
            noisy = img_transform(noisy)
            point = geo_transform(point)
            normal = geo_transform(normal)



        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        depth_filename = os.path.split(self.depth_filenames[tar_index])[-1]
        normal_filename = os.path.split(self.normal_filenames[tar_index])[-1]


        augment.rotate = random.randint(-20,20)
        apply_trans = transforms_aug[random.randint(0, 2)]

        # [0, 1]
        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy) 
        point = getattr(augment, apply_trans)(point)
        normal = getattr(augment, apply_trans)(normal)


        #Crop Input and Target
        ps = self.img_options['patch_size']
        scale = 1#random.uniform(1, 1.5)

        H = noisy.shape[1]
        W = noisy.shape[2]
        scaled_ps = (int)(scale * ps)
        if H - scaled_ps != 0 or W - scaled_ps != 0:
            r = np.random.randint(0, H - scaled_ps + 1)
            c = np.random.randint(0, W - scaled_ps + 1)
            clean   = clean  [:, r:r + scaled_ps, c:c + scaled_ps]
            noisy   = noisy  [:, r:r + scaled_ps, c:c + scaled_ps]
            point   = point  [:, r:r + scaled_ps, c:c + scaled_ps]
            normal  = normal [:, r:r + scaled_ps, c:c + scaled_ps]

        # scale back to the patch_size
        if scale != 1:
            clean = F.interpolate(clean.unsqueeze(0), size=[ps, ps], mode='bilinear')
            noisy = F.interpolate(noisy.unsqueeze(0), size=[ps, ps], mode='bilinear')
            point = F.interpolate(point.unsqueeze(0), size=[ps, ps], mode='nearest')
            normal = F.interpolate(normal.unsqueeze(0), size=[ps, ps], mode='nearest')
            return clean.squeeze(0), noisy.squeeze(0), point.squeeze(0), normal.squeeze(0), noisy_filename

        return clean, noisy, point, normal, clean_filename, noisy_filename
    

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None, debug=False, ps=256):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'shadow_free'
        input_dir = 'origin'
        depth_dir = 'depth'
        normal_dir = 'normal'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        depth_files = sorted(os.listdir(os.path.join(rgb_dir, depth_dir)))
        normal_files = sorted(os.listdir(os.path.join(rgb_dir, normal_dir)))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files]
        self.depth_filenames = [os.path.join(rgb_dir, depth_dir, x) for x in depth_files]
        self.normal_filenames = [os.path.join(rgb_dir, normal_dir, x) for x in normal_files]

        self.ps = ps

        if debug:
            self.tar_size = 10
        else:
            self.tar_size = len(self.noisy_filenames)

    def __len__(self):
        return self.tar_size 

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean  = np.float32(load_img(self.clean_filenames[tar_index]))
        noisy  = np.float32(load_img(self.noisy_filenames[tar_index]))
        depth  = np.float32(load_depth(self.depth_filenames[tar_index]))
        normal = np.float32(load_normal(self.normal_filenames[tar_index]))

        point = depthToPoint(60, depth)   
        normal = process_normal(normal)
        point = point / (2 * point[:,:,2].mean())

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        # ==================== 新增尺寸判斷邏輯 ====================
        # 在轉換為 Tensor 前，先從 numpy 陣列取得原始尺寸 (H, W, C)
        H, W, _ = noisy.shape
        # ==========================================================

        clean  = torch.from_numpy(clean)
        noisy  = torch.from_numpy(noisy)
        point  = torch.from_numpy(point)
        normal = torch.from_numpy(normal)

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        point  = point.permute(2,0,1)
        normal = normal.permute(2,0,1)


        if H > 1000 or W > 1000:
            ps = self.ps # 目標尺寸

            img_transform = transforms.Resize((ps, ps), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            
            geo_transform = transforms.Resize((ps, ps), interpolation=transforms.InterpolationMode.NEAREST)

            clean = img_transform(clean)
            noisy = img_transform(noisy)
            point = geo_transform(point)
            normal = geo_transform(normal)

        return clean, noisy, point, normal, clean_filename, noisy_filename


class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None, debug=False, ps=256):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform
        
        input_dir = 'origin'
        depth_dir = 'depth'
        normal_dir = 'normal'
        
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        depth_files = sorted(os.listdir(os.path.join(rgb_dir, depth_dir)))
        normal_files = sorted(os.listdir(os.path.join(rgb_dir, normal_dir)))

        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files]
        self.depth_filenames = [os.path.join(rgb_dir, depth_dir, x) for x in depth_files]
        self.normal_filenames = [os.path.join(rgb_dir, normal_dir, x) for x in normal_files]

        self.ps = ps

        if debug:
            self.tar_size = 10
        else:
            self.tar_size = len(self.noisy_filenames)

    def __len__(self):
        return self.tar_size 

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        noisy  = np.float32(load_img(self.noisy_filenames[tar_index]))
        depth  = np.float32(load_depth(self.depth_filenames[tar_index]))
        normal = np.float32(load_normal(self.normal_filenames[tar_index]))

        point = depthToPoint(60, depth)   
        normal = process_normal(normal)
        point = point / (2 * point[:,:,2].mean())

        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        H, W, _ = noisy.shape

        noisy  = torch.from_numpy(noisy)
        point  = torch.from_numpy(point)
        normal = torch.from_numpy(normal)

        noisy = noisy.permute(2,0,1)
        point  = point.permute(2,0,1)
        normal = normal.permute(2,0,1)

        if H > 1000 or W > 1000:
            ps = self.ps

            img_transform = transforms.Resize((ps, ps), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            geo_transform = transforms.Resize((ps, ps), interpolation=transforms.InterpolationMode.NEAREST)

            noisy = img_transform(noisy)
            point = geo_transform(point)
            normal = geo_transform(normal)
        # ==========================================================
        
        return noisy, noisy, point, normal, noisy_filename, noisy_filename