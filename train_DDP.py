import os
import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from losses import CharbonnierLoss, TVLoss
import os
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
import torch.nn.functional as F
from utils.loader import get_validation_data, get_training_data
import time
import argparse
import options
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm


from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Single Image Shadow Removal')).parse_args()
local_rank = opt.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='gloo')
device = torch.device("cuda", local_rank)
if opt.debug == True:
    opt.eval_now = 2

######### Logs dir ###########
dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, opt.save_dir, opt.arch+opt.env+datetime.datetime.now().isoformat(timespec='minutes'))
logname = os.path.join(log_dir, datetime.datetime.now().isoformat(timespec='minutes')+'.txt') 

result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
tensorlog_dir  = os.path.join(log_dir, 'tensorlog')
if dist.get_rank() == 0:
    utils.mkdir(log_dir)
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)
    utils.mkdir(tensorlog_dir)
    utils.mknod(logname)
    tb_logger = SummaryWriter(log_dir=tensorlog_dir)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


####### just allow one process to print info to log
if dist.get_rank() == 0:
    logging.basicConfig(filename=logname,level=logging.INFO if dist.get_rank() in [-1, 0] else logging.WARN)
    torch.distributed.barrier()
else:
    torch.distributed.barrier()
    logging.basicConfig(filename=logname,level=logging.INFO if dist.get_rank() in [-1, 0] else logging.WARN)

logging.info(opt)
logging.info(f"Now time is : {datetime.datetime.now().isoformat()}")
########### Set Seeds ###########
random.seed(1234 + dist.get_rank())
np.random.seed(1234 + dist.get_rank())
torch.manual_seed(1234 + dist.get_rank())
torch.cuda.manual_seed(1234 + dist.get_rank())
torch.cuda.manual_seed_all(1234 + dist.get_rank())

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id + dist.get_rank())

g = torch.Generator()
g.manual_seed(1234 + dist.get_rank())

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


######### Model ###########
model_restoration = utils.get_arch(opt)
model_restoration.to(device)
DINO_Net = torch.hub.load('./dinov2', 'dinov2_vitl14', source='local')
logging.info(str(model_restoration) + '\n')


######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### Resume ###########
start_epoch = 1
ckpt = None

if opt.resume:
    path_chk_rest = opt.pretrain_weights
    ckpt = torch.load(path_chk_rest, map_location='cpu')

    utils.load_checkpoint(model_restoration, path_chk_rest)
    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    start_epoch = (ckpt.get('epoch', 0) or utils.load_start_epoch(path_chk_rest)) + 1

    logging.info("------------------------------------------------------------------------------")
    logging.info(f"==> Resuming Training from epoch {start_epoch}")
    logging.info("------------------------------------------------------------------------------")

######### DDP ###########
model_restoration = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_restoration).to(device)
model_restoration = DDP(model_restoration, 
                        device_ids=[local_rank], 
                        output_device=local_rank,
                        find_unused_parameters=True)

DINO_Net.to(device)
DINO_Net.eval()
DINO_Net = DDP(DINO_Net, device_ids=[local_rank], output_device=local_rank)


# ######### Scheduler ###########
scheduler = None

if opt.scheduler == 'reduce_on_plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.85,
        patience=15,
        threshold=1e-3,
        threshold_mode='rel',
        cooldown=0,
        min_lr=1e-7
    )
elif opt.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
elif opt.scheduler == 'step':
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
else:
    raise ValueError(f'Unknown scheduler: {opt.scheduler}')


if opt.resume and ckpt is not None:
    if 'scheduler' in ckpt and scheduler is not None:
        try:
            scheduler.load_state_dict(ckpt['scheduler'])
            logging.info("Scheduler state loaded from checkpoint.")
        except Exception as e:
            logging.warning(f"Scheduler state not fully compatible, continue fresh. reason={e}")
    # timm 的 NativeScaler 有 state_dict / load_state_dict
    if 'scaler' in ckpt:
        try:
            loss_scaler.load_state_dict(ckpt['scaler'])
            logging.info("AMP scaler state loaded from checkpoint.")
        except Exception as e:
            logging.warning(f"AMP scaler state not compatible, continue fresh. reason={e}")


######### Loss ###########
criterion_restore = CharbonnierLoss().to(device)

######### DataLoader ###########
logging.info('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train, opt.debug)
train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size // dist.get_world_size(), 
        num_workers=opt.train_workers, sampler=train_sampler, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn,
        generator=g )

val_dataset = get_validation_data(opt.val_dir, opt.debug)
val_sampler = DistributedSampler(val_dataset, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size= 3 // dist.get_world_size(),
        num_workers=opt.eval_workers, sampler=val_sampler, pin_memory=False, drop_last=False, worker_init_fn=worker_init_fn,
        generator=g)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
logging.info(f"Sizeof training set: {len_trainset} sizeof validation set: {len_valset}")

######### train ###########
logging.info("===> Start Epoch {} End Epoch {}".format(start_epoch,opt.nepoch))
best_psnr = 0
best_ssim = 0
best_epoch = 0
best_iter = 0
logging.info("\nEvaluation after every {} Iterations !!!\n".format(opt.eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()

index = 0
DINO_patch_size = 14
img_multiple_of = 8 * opt.win_size

# the train_ps must be the multiple of win_size
UpSample = nn.UpsamplingBilinear2d(
    size=((int)(opt.train_ps * DINO_patch_size / 8), 
        (int)(opt.train_ps * DINO_patch_size / 8))
    )

Charbonnier_weight = 0.999
SSIM_weight = 0.001

for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0

    epoch_direct_loss = 0
    epoch_indirect_loss = 0
    train_id = 1
    epoch_ssim_loss = 0
    epoch_tv_loss = 0
    epoch_Charbonnier_loss = 0
    epoch_perceptual_loss = 0


    train_loader.sampler.set_epoch(epoch)

    train_bar = tqdm(train_loader, disable=dist.get_rank() != 0)

    for i, data in enumerate(train_bar, 0): 
        # zero_grad
        index += 1
        optimizer.zero_grad()
        target = data[0].to(device)
        input_ = data[1].to(device)
        point = data[2].to(device)
        normal = data[3].to(device)

        with torch.amp.autocast('cuda'):
            dino_mat_features = None
            with torch.no_grad():
                input_DINO = UpSample(input_)
                dino_mat_features = DINO_Net.module.get_intermediate_layers(input_DINO, 4, True)

            restored = model_restoration(input_, dino_mat_features, point, normal)
            
            
            restored = restored.clamp(0.0, 1.0)
            target   = target.clamp(0.0, 1.0)

            restored = torch.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0)
            target   = torch.nan_to_num(target,   nan=0.0, posinf=1.0, neginf=0.0)
            
            loss_restore = criterion_restore(restored, target)


            ssim_loss = 1 - ssim_metric(restored, target)


            loss = Charbonnier_weight * loss_restore + SSIM_weight * ssim_loss



        # 避免AMP NaN
        if not torch.isfinite(loss):
            if dist.get_rank() == 0:
                logging.warning(f"[NaN detected] epoch={epoch}, iter={i}, lr={optimizer.param_groups[0]['lr']:.2e}")
            optimizer.zero_grad(set_to_none=True)
            loss_scaler._scaler = torch.amp.GradScaler('cuda', init_scale=2.0, growth_interval=100)
            continue


        loss_scaler(loss, 
                    optimizer,
                    parameters=model_restoration.parameters(),
                    clip_grad=1.0)

        loss_list = utils.distributed_concat(loss, dist.get_world_size())
        ssim_loss_list = utils.distributed_concat(ssim_loss, dist.get_world_size())
        Charbonnier_loss_list = utils.distributed_concat(loss_restore, dist.get_world_size())


        loss = 0
        for ele in loss_list:
            loss += ele.item()


        ssim_loss = sum(ele.item() for ele in ssim_loss_list) / dist.get_world_size()
        Charbonnier_loss = sum(ele.item() for ele in Charbonnier_loss_list) / dist.get_world_size()

        epoch_loss += loss
        epoch_ssim_loss += (SSIM_weight * ssim_loss) / len(train_loader)
        epoch_Charbonnier_loss += (Charbonnier_weight * Charbonnier_loss) / len(train_loader)


        if dist.get_rank() == 0:
            train_bar.set_description(f"Train Epoch: [{epoch}/{opt.nepoch}] Loss: {loss:.4f}")
            tb_logger.add_scalar("train/loss", epoch_loss, epoch+1)
            tb_logger.add_scalar("train/SSIM_Loss", epoch_ssim_loss, epoch+1)
            tb_logger.add_scalar("train/Charbonnier_loss", epoch_Charbonnier_loss, epoch+1)

            tb_logger.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch+1)
    train_bar.close()
  
    ################# Evaluation ########################
    if (epoch + 1) % opt.eval_now == 0:
        eval_shadow_rmse = 0
        eval_nonshadow_rmse = 0
        eval_rmse = 0
        with torch.no_grad():
            model_restoration.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            val_loss_rgb = []
            val_charbonnier_loss_rgb = []
            val_ssim_loss_rgb = []

            val_bar = tqdm(val_loader, disable=dist.get_rank()!=0)
            val_bar.set_description(f'Validation Epoch {epoch}')

            for _, data_val in enumerate(val_bar, 0):
                target = data_val[0].to(device)
                input_ = data_val[1].to(device)
                point = data_val[2].to(device)
                normal = data_val[3].to(device)
                filenames = data_val[4]

                height, width = input_.shape[2], input_.shape[3]
                H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of

                padh = H - height if height % img_multiple_of != 0 else 0
                padw = W - width if width % img_multiple_of != 0 else 0
                input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
                point = F.pad(point, (0, padw, 0, padh), 'reflect')
                normal = F.pad(normal, (0, padw, 0, padh), 'reflect')
                
                UpSample_val = nn.UpsamplingBilinear2d(
                    size=((int)(input_.shape[2] * (DINO_patch_size / 8)), 
                        (int)(input_.shape[3] * (DINO_patch_size / 8))))

                with torch.amp.autocast('cuda'):
                    # DINO_V2
                    input_DINO = UpSample_val(input_)
                    dino_mat_features = DINO_Net.module.get_intermediate_layers(input_DINO, 4, True)
                    restored = model_restoration(input_, dino_mat_features, point, normal)

                restored = torch.clamp(restored, 0.0, 1.0)
                restored = restored[:, :, :height, :width]

                restored = torch.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0)
                target   = torch.nan_to_num(target,   nan=0.0, posinf=1.0, neginf=0.0)

                charbonnier_loss = criterion_restore(restored, target)
                ssim_loss = 1 - ssim_metric(restored, target)
                total_loss = Charbonnier_weight * charbonnier_loss + SSIM_weight * ssim_loss

                val_loss_rgb.append(total_loss.item())
                val_charbonnier_loss_rgb.append(charbonnier_loss.item())
                val_ssim_loss_rgb.append(ssim_loss.item())

                psnr_val_rgb.append(utils.batch_PSNR(restored, target, True))
                ssim_val_rgb.append(ssim_loss.item())

                if dist.get_rank() == 0:
                    for idx, filename in enumerate(filenames):
                        rgb_restored = restored[idx].cpu().numpy().squeeze().transpose((1, 2, 0))
                        utils.save_img(rgb_restored * 255.0, os.path.join(result_dir, filename))

                val_bar.set_postfix(psnr=f'{sum(psnr_val_rgb)/len(psnr_val_rgb):.4f}', 
                                    ssim=f'{sum(ssim_val_rgb)/len(ssim_val_rgb):.4f}', 
                                    loss=f'{sum(val_loss_rgb)/len(val_loss_rgb):.4f}')


            psnr_val_rgb = sum(psnr_val_rgb) / len(val_loader)
            ssim_val_rgb = sum(ssim_val_rgb) / len(val_loader)
            val_loss_rgb = sum(val_loss_rgb) / len(val_loader)
            val_charbonnier_loss_rgb = sum(val_charbonnier_loss_rgb) / len(val_loader)
            val_ssim_loss_rgb = sum(val_ssim_loss_rgb) / len(val_loader)


            psnr_val_rgb = torch.tensor(psnr_val_rgb, dtype=torch.float32, device=device)
            ssim_val_rgb = torch.tensor(ssim_val_rgb, dtype=torch.float32, device=device)
            val_loss_rgb = torch.tensor(val_loss_rgb, dtype=torch.float32, device=device)
            val_charbonnier_loss_rgb = torch.tensor(val_charbonnier_loss_rgb, dtype=torch.float32, device=device)
            val_ssim_loss_rgb = torch.tensor(val_ssim_loss_rgb, dtype=torch.float32, device=device)


            val_loss_rgb_list = utils.distributed_concat(val_loss_rgb, dist.get_world_size())
            val_charbonnier_loss_rgb_list = utils.distributed_concat(val_charbonnier_loss_rgb, dist.get_world_size())
            val_ssim_loss_rgb_list = utils.distributed_concat(val_ssim_loss_rgb, dist.get_world_size())

            val_loss_rgb = sum(ele.item() for ele in val_loss_rgb_list) / len(val_loss_rgb_list)

            if opt.scheduler == 'reduce_on_plateau':
                metric_t = torch.tensor([val_loss_rgb], dtype=torch.float32, device=device)
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(metric_t, src=0)

                scheduler.step(metric_t.item())



            val_charbonnier_loss_rgb = sum(ele.item() for ele in val_charbonnier_loss_rgb_list) / len(val_charbonnier_loss_rgb_list)
            val_ssim_loss_rgb = sum(ele.item() for ele in val_ssim_loss_rgb_list) / len(val_ssim_loss_rgb_list)


            psnr_val_rgb_list = utils.distributed_concat(psnr_val_rgb, dist.get_world_size())
            ssim_val_rgb_list = utils.distributed_concat(ssim_val_rgb, dist.get_world_size())

            psnr_val_rgb = sum(ele.item() for ele in psnr_val_rgb_list) / len(psnr_val_rgb_list)
            ssim_val_rgb = sum(ele.item() for ele in ssim_val_rgb_list) / len(ssim_val_rgb_list)

            if dist.get_rank() == 0:
                tb_logger.add_scalar("val/psnr", psnr_val_rgb, epoch)
                tb_logger.add_scalar("val/ssim", ssim_val_rgb, epoch)
                tb_logger.add_scalar("val/loss", val_loss_rgb, epoch)
                tb_logger.add_scalar("val/Charbonnier_loss", val_charbonnier_loss_rgb, epoch)
                tb_logger.add_scalar("val/SSIM_loss", val_ssim_loss_rgb, epoch)

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model_restoration.module.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_best.pth"))

            logging.info("[Ep %d it %d\t PSNR: %.4f\t] ----  [Best_Epoch %d Best_Iter %d Best_PSNR %.4f] " \
                    % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
            logging.info("[Ep %d it %d\t SSIM: %.4f\t] ----  [Best_Epoch %d Best_Iter %d Best_SSIM %.4f] " \
                    % (epoch, i, ssim_val_rgb, best_epoch, best_iter, best_ssim))
            logging.info("[Ep %d it %d\t Validation Loss: %.4f\t]" % (epoch, i, val_loss_rgb))
            logging.info("[Ep %d it %d\t Charbonnier Loss: %.4f\t]" % (epoch, i, val_charbonnier_loss_rgb))
            logging.info("[Ep %d it %d\t SSIM Loss: %.4f\t]" % (epoch, i, val_ssim_loss_rgb))
            logging.info("Now time is : {}".format(datetime.datetime.now().isoformat()))

            model_restoration.train()
            torch.cuda.empty_cache()

        if opt.scheduler != 'reduce_on_plateau':
            scheduler.step()

    logging.info("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
    epoch, time.time()-epoch_start_time, epoch_loss, optimizer.param_groups[0]['lr']))
    if dist.get_rank() == 0:
        if opt.scheduler != 'reduce_on_plateau':
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.module.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_latest.pth"))   
        else:
            torch.save({
                'epoch': epoch,
                'state_dict': model_restoration.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': loss_scaler.state_dict(),
            }, os.path.join(model_dir, "model_latest.pth"))



        if epoch%opt.checkpoint == 0 and opt.scheduler != 'reduce_on_plateau':
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.module.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
        elif epoch%opt.checkpoint == 0 and opt.scheduler == 'reduce_on_plateau':
            torch.save({
                'epoch': epoch,
                'state_dict': model_restoration.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 
                'scaler': loss_scaler.state_dict(),
            }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))
logging.info("Now time is : {}".format(datetime.datetime.now().isoformat()))
