#!/usr/bin/env python

import os, sys, argparse, time
import numpy as np, cv2, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.backends.cudnn.benchmark = False
import utils
from utils import load_depth, load_normal, depthToPoint, process_normal
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as TM_LPIPS

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to checkpoint (.pth with 'state_dict')")
    p.add_argument("--data", required=True, help="test split dir with origin/ shadow_free/ depth/ normal/")
    p.add_argument("--win_size", type=int, default=16, help="window size; use 8 for Ambient6K/INS checkpoints, 16 for ISTD/WSRD")
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--eval_size", type=int, default=256, help="metric resolution")
    p.add_argument("--resize", choices=["cv2_area", "finterp_area"], default="cv2_area", help="256-downsample for the metric.")
    p.add_argument("--tta", action="store_true", help="8-way flip/rot self-ensemble")
    p.add_argument("--save_dir", default="", help="if set, save restored PNGs here")
    p.add_argument("--limit", type=int, default=0, help="evaluate only first N images (0 = all)")
    return p.parse_args()

class _Opt:
    pass

def main():
    a = build_args(); dev = "cuda"; DPS = 14; WIN = a.win_size
    opt = _Opt(); opt.arch="PhaSR"; opt.train_ps=a.eval_size; opt.embed_dim=a.embed_dim
    opt.win_size=WIN; opt.token_projection="linear"; opt.token_mlp="leff"
    model = utils.get_arch(opt).to(dev).eval()
    utils.load_checkpoint(model, a.weights)
    dino = torch.hub.load("./dinov2", "dinov2_vitl14", source="local").to(dev).eval()
    lpips_fn = TM_LPIPS(net_type='vgg', reduction='mean').to(dev).eval()
    print(f"[test] loaded {a.weights} | tta={a.tta} | eval@{a.eval_size}", flush=True)

    origin = os.path.join(a.data, "origin"); gtdir = os.path.join(a.data, "shadow_free")
    ddir = os.path.join(a.data, "depth"); ndir = os.path.join(a.data, "normal")
    has_gt = os.path.isdir(gtdir)
    if a.save_dir: os.makedirs(a.save_dir, exist_ok=True)

    def dfeat(x):
        xu = F.interpolate(x, size=(int(x.shape[2]*DPS/8), int(x.shape[3]*DPS/8)), mode="bilinear", align_corners=False)
        return dino.get_intermediate_layers(xu, 4, True)

    def infer(inp, pt, nm):
        m = 8*WIN; h, w = inp.shape[2], inp.shape[3]
        H = ((h+m)//m)*m; W = ((w+m)//m)*m
        xi = F.pad(inp,(0,W-w,0,H-h),'reflect'); pp = F.pad(pt,(0,W-w,0,H-h),'reflect'); nn_ = F.pad(nm,(0,W-w,0,H-h),'reflect')
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(xi, dfeat(xi), pp, nn_)
        return o.float().clamp(0,1)[:, :, :h, :w]

    def tta(inp, pt, nm):
        sp = ["id","h","v","r90","r180","r270","hr90","vr90"]
        A = lambda t,s: {"id":t,"h":torch.flip(t,[3]),"v":torch.flip(t,[2]),"r90":torch.rot90(t,1,[2,3]),
                         "r180":torch.rot90(t,2,[2,3]),"r270":torch.rot90(t,3,[2,3]),
                         "hr90":torch.rot90(torch.flip(t,[3]),1,[2,3]),"vr90":torch.rot90(torch.flip(t,[2]),1,[2,3])}[s]
        D = lambda t,s: {"id":t,"h":torch.flip(t,[3]),"v":torch.flip(t,[2]),"r90":torch.rot90(t,3,[2,3]),
                         "r180":torch.rot90(t,2,[2,3]),"r270":torch.rot90(t,1,[2,3]),
                         "hr90":torch.flip(torch.rot90(t,3,[2,3]),[3]),"vr90":torch.flip(torch.rot90(t,3,[2,3]),[2])}[s]
        acc = None
        for s in sp:
            o = D(infer(A(inp,s),A(pt,s),A(nm,s)), s); acc = o if acc is None else acc+o
        return acc/len(sp)

    def to_eval(x):
        s = a.eval_size
        if a.resize == "finterp_area":
            t = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
            return F.interpolate(t, (s, s), mode='area')[0].permute(1,2,0).numpy()
        return cv2.resize(x, (s, s), interpolation=cv2.INTER_AREA)
    def load_geo(stem):
        depth = load_depth(os.path.join(ddir, stem+".npy")); normal = process_normal(load_normal(os.path.join(ndir, stem+".npy")))
        point = depthToPoint(60, depth); point = point/(2*point[:,:,2].mean())
        return point, normal

    files = sorted(os.listdir(origin))
    if a.limit: files = files[:a.limit]
    gt_map = {}
    if has_gt:
        for f in os.listdir(gtdir):
            key = os.path.splitext(f)[0].rsplit('_', 1)[0]
            gt_map[key] = f
    PS, SS, LP = [], [], []; t0 = time.time()
    with torch.no_grad():
        for i, fn in enumerate(files):
            stem = os.path.splitext(fn)[0]
            inp = cv2.cvtColor(cv2.imread(os.path.join(origin,fn)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.
            point, normal = load_geo(stem)
            it = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).to(dev)
            pt = torch.from_numpy(point).permute(2,0,1).unsqueeze(0).float().to(dev)
            nm = torch.from_numpy(normal).permute(2,0,1).unsqueeze(0).float().to(dev)
            out = (tta(it,pt,nm) if a.tta else infer(it,pt,nm))[0].cpu().numpy().transpose(1,2,0)
            if a.save_dir:
                cv2.imwrite(os.path.join(a.save_dir, fn), cv2.cvtColor((out*255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
            if has_gt:
                gt_fn = gt_map.get(stem.rsplit('_', 1)[0], fn)
                g = cv2.cvtColor(cv2.imread(os.path.join(gtdir, gt_fn)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.
                r = to_eval(out); gg = to_eval(g)
                PS.append(psnr_sk(gg, r, data_range=1.0))
                SS.append(ssim_sk(cv2.cvtColor(gg,cv2.COLOR_RGB2GRAY), cv2.cvtColor(r,cv2.COLOR_RGB2GRAY), data_range=1.0))
                r_t  = torch.from_numpy(r).permute(2,0,1).unsqueeze(0).float().to(dev)
                gg_t = torch.from_numpy(gg).permute(2,0,1).unsqueeze(0).float().to(dev)
                LP.append(lpips_fn(r_t, gg_t).item())
            if (i+1) % 100 == 0: print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    if has_gt:
        print(f"  PSNR={np.mean(PS):.3f}  SSIM={np.mean(SS):.4f}  LPIPS={np.mean(LP):.4f}")
    else:
        print(f"[test] saved {len(files)} restored images to {a.save_dir} (no GT -> no metrics)")

if __name__ == "__main__":
    main()
