import numpy as np
import os
import argparse
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import random
from utils.loader import get_validation_data
import utils
import cv2
import torch.distributed as dist
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss


from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as TM_LPIPS




parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='dataset/test', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default=None, type=str, help='Directory for results')
parser.add_argument('--weights', default=None ,type=str, help='Path to weights')
parser.add_argument('--arch', type=str, default='phasr', help='archtechture')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', default=False, help='Save denoised images in result directory')
parser.add_argument('--cal_metrics', action='store_true', default=True, help='Measure denoised images with GT')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=16, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')

parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
parser.add_argument("--local-rank", default=0, type=int)

args = parser.parse_args()

local_rank = args.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='gloo')
device = torch.device("cuda", local_rank)


utils.mkdir(args.result_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)

g = torch.Generator()
g.manual_seed(1234)

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
######### Model ###########
model_restoration = utils.get_arch(args)
model_restoration.to(device)
model_restoration.eval()
DINO_Net = torch.hub.load('./dinov2', 'dinov2_vitl14', source='local')
DINO_Net.to(device)
DINO_Net.eval()

# LPIPS model (VGG backbone)
tm_lpips = TM_LPIPS(net_type='vgg', reduction='mean').to(device).eval()



######### Load ###########
utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

######### DDP ###########

model_restoration = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_restoration).to(device)
model_restoration = DDP(model_restoration, device_ids=[local_rank], output_device=local_rank)
DINO_Net = DDP(DINO_Net, device_ids=[local_rank], output_device=local_rank)

######### Test ###########
img_multiple_of = 8 * args.win_size
DINO_patch_size = 14

def UpSample(img):
    upsample = nn.UpsamplingBilinear2d(
        size=((int)(img.shape[2] * (DINO_patch_size / 8)), 
            (int)(img.shape[3] * (DINO_patch_size / 8))))
    return upsample(img)

img_options_train = {'patch_size':args.train_ps}
test_dataset = get_validation_data(args.input_dir, False)
test_sampler = DistributedSampler(test_dataset, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=0, sampler=test_sampler, drop_last=False, worker_init_fn=worker_init_fn, generator=g)
with torch.no_grad():
    psnr_val_rgb_list = []
    psnr_val_mask_list = []
    ssim_val_rgb_list = []
    rmse_val_rgb_list = []
    lpips_val_rgb_list = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
            rgb_noisy = data_test[1].to(device)
            point = data_test[2].to(device)
            normal = data_test[3].to(device)
            filenames = data_test[4]

            height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]

            with torch.amp.autocast('cuda'):
                # DINO_V2
                input_DINO = UpSample(rgb_noisy)
                dino_mat_features = DINO_Net.module.get_intermediate_layers(input_DINO, 4, True)
                restored = model_restoration(rgb_noisy, dino_mat_features, point, normal)

        
            restored = torch.clamp(restored, 0.0, 1.0)
            restored = restored[:, : ,:height, :width]

            restored = restored.cpu().numpy().squeeze().transpose((1,2,0))

        
            if args.cal_metrics:
            # calculate SSIM in gray space
                gray_restored = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY)
                gray_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)
                ssim_val_rgb = ssim_loss(gray_restored, gray_gt, channel_axis=None, data_range=gray_restored.max() - gray_restored.min())
                ssim_val_rgb = torch.tensor(ssim_val_rgb).to(device)
                list = utils.distributed_concat(ssim_val_rgb, dist.get_world_size())
                ssim_val_rgb_list.extend(list)

                psnr_val_rgb = psnr_loss(restored, rgb_gt)
                psnr_val_rgb = torch.tensor(psnr_val_rgb).to(device)
                list = utils.distributed_concat(psnr_val_rgb, dist.get_world_size())
                psnr_val_rgb_list.extend(list)

                # calculate the RMSE in LAB space
                rmse_temp = np.abs(cv2.cvtColor(restored, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2LAB)).mean() * 3
                rmse_temp = torch.tensor(rmse_temp).to(device)
                list = utils.distributed_concat(ssim_val_rgb, dist.get_world_size())
                rmse_val_rgb_list.extend(list)

                rest_t = torch.from_numpy(restored.transpose(2,0,1)).unsqueeze(0).float().to(device)
                gt_t   = torch.from_numpy(rgb_gt.transpose(2,0,1)).unsqueeze(0).float().to(device)
                lpips_val = tm_lpips(rest_t, gt_t)  # already scalar
                list = utils.distributed_concat(lpips_val, dist.get_world_size())
                lpips_val_rgb_list.extend(list)

            if args.save_images:
                utils.save_img(restored * 255.0, os.path.join(args.result_dir, filenames[0]))
            
if args.cal_metrics:
    is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)

    ssim_val_rgb  = 0.0
    psnr_val_rgb  = 0.0
    rmse_val_rgb  = 0.0
    lpips_val_rgb = 0.0

    for ssim_ele, psnr_ele, rmse_ele, lpips_ele in zip(
        ssim_val_rgb_list, psnr_val_rgb_list, rmse_val_rgb_list, lpips_val_rgb_list
    ):
        ssim_val_rgb  += ssim_ele.item()
        psnr_val_rgb  += psnr_ele.item()
        rmse_val_rgb  += rmse_ele.item()
        lpips_val_rgb += lpips_ele.item()

    n = len(test_dataset) if len(test_dataset) > 0 else 1
    psnr_val_rgb  /= n
    ssim_val_rgb  /= n
    rmse_val_rgb  /= n
    lpips_val_rgb /= n

    if is_main:
        print("PSNR: %.4f, SSIM: %.4f, RMSE: %.4f, LPIPS: %.4f" %
              (psnr_val_rgb, ssim_val_rgb, rmse_val_rgb, lpips_val_rgb))


