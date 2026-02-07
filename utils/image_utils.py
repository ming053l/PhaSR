import torch
import numpy as np
import pickle
import cv2
from skimage.color import rgb2lab
import os
import math
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_SSAO(filepath):
    img = cv2.imread(filepath)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = img/255.
    return img
def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = img/255.
    return img

def load_val_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    resized_img = img.astype(np.float32)
    resized_img = resized_img/255.
    return resized_img

def load_mask(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # kernel = np.ones((8,8), np.uint8)
    # erosion = cv2.erode(img, kernel, iterations=1)
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # contour = dilation - erosion
    img = img.astype(np.float32)
    # contour = contour.astype(np.float32)
    # contour = contour/255.
    img = img/255.
    return img

def load_ssao(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    # contour = contour.astype(np.float32)
    # contour = contour/255.
    img = img/255.
    return img

def load_depth(filepath):
    img = np.load(filepath)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    # img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    # img = img / 255
    return img

def load_normal(filepath):
    img = np.load(filepath).transpose(1,2,0)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    return img

def load_val_mask(filepath):
    img = cv2.imread(filepath, 0)
    resized_img = img
    # resized_img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    resized_img = resized_img.astype(np.float32)
    resized_img = resized_img/255.
    return resized_img

def save_img(img, filepath):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

    # imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    # rmse = (imdff**2).mean()
    # ps = 10*torch.log10(1/rmse)
    # return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):

        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')

        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    # image_numpy =
    return np.clip(image_numpy, 0, 255).astype(imtype)

def calc_RMSE(real_img, fake_img):
    # convert to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    return real_lab - fake_lab

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def process_normal(normal):
    normal = normal * 2.0 - 1.0
    normal = normal[:,:,np.newaxis,:]
    normalizer = np.sqrt(normal @ normal.transpose(0,1,3,2))
    normalizer = np.squeeze(normalizer, axis=-2)
    normalizer = np.clip(normalizer, 1.0e-20, 1.0e10)
    normal = np.squeeze(normal, axis=-2)
    normal = normal / normalizer 
    return normal


def depthToPoint(fov, depth):
    # width = 512
    # height = 512
    height, width = depth.shape
    fov_radians = np.deg2rad(fov)

    focal_length = width / (2 * np.tan(fov_radians / 2))
    fx = focal_length  
    fy = focal_length 
    cx = (width - 1) / 2.0  
    cy = (height - 1) / 2.0  

    x, y = np.meshgrid(range(width), range(height))
    z = depth 
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    x_3d = x_3d.astype(np.float32)
    y_3d = y_3d.astype(np.float32)


    point_cloud = np.stack((x_3d, y_3d, z), axis=-1)

    return point_cloud

def grid_sample(input, img_size):
    x = torch.linspace(-1, 1, img_size[0])
    y = torch.linspace(-1, 1, img_size[1])
    meshx, meshy = torch.meshgrid((x, y))
    grid = torch.stack((meshy, meshx),2).unsqueeze(0).cuda()
    grid = grid.repeat(input.shape[0],1,1,1)
    input = torch.nn.functional.grid_sample(input, grid, mode="nearest", align_corners=False)
    return input


