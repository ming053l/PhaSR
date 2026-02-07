import torch
import os
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights, strict=True):
    checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    try:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=strict)
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=strict)

def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights,  map_location=torch.device('cpu'))
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights,  map_location=torch.device('cpu'))
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from model import PhaSR
    arch = opt.arch

    print('You choose '+arch+'...')
    if arch == 'PhaSR':
        model_restoration = PhaSR(img_size=opt.train_ps,embed_dim=opt.embed_dim,
                                        win_size=opt.win_size,token_projection=opt.token_projection,
                                        token_mlp=opt.token_mlp)
    else:
        raise Exception("Arch error!")

    return model_restoration


def window_partition(x, win_size):
    B, C, H, W = x.shape
    x = x.permute(0,2,3,1)
    x = x.reshape(B, H // win_size, win_size, W // win_size, win_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, win_size, win_size, C)
    return x.permute(0,3,1,2)

# def distributed_concat(var, num_total):
#     var_list = [torch.zeros(1, dtype=var.dtype).cuda() for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(var_list, var)
#     # truncate the dummy elements added by SequentialDistributedSampler
#     return var_list[:num_total]

def distributed_concat(var, num_total):
    # 確保 var 是一個 1D tensor (shape: [1])
    var = var.view(1) if var.dim() == 0 else var

    var_list = [torch.zeros_like(var).cuda() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(var_list, var)

    # truncate the dummy elements added by SequentialDistributedSampler
    return var_list[:num_total]
