#!/usr/bin/env python

import os, sys, time, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.backends.cudnn.benchmark = False
import utils
from losses import CharbonnierLoss
from utils.loader import get_training_data, get_validation_data
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nepoch", type=int, default=1400)
    p.add_argument("--batch_size", type=int, default=9)
    p.add_argument("--train_ps", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--win_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--keep_topk", type=int, default=5, help="keep the K highest-PSNR checkpoints (never lose a good one)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--train_dir", default="dataset/train")
    p.add_argument("--val_dir", default="dataset/valid")
    p.add_argument("--save_dir", default="runs/phasr")
    p.add_argument("--resume", default="")
    return p.parse_args()

class _Opt: pass

def main():
    a = build_args(); dev = "cuda"; DPS = 14
    MIXUP = os.environ.get("PHASR_MIXUP", "0") == "1"
    STRONG_AUG = os.environ.get("PHASR_STRONG_AUG", "0") == "1"
    mdir = os.path.join(a.save_dir, "models"); os.makedirs(mdir, exist_ok=True)
    def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    opt = _Opt(); opt.arch="PhaSR"; opt.train_ps=a.train_ps; opt.embed_dim=a.embed_dim
    opt.win_size=a.win_size; opt.token_projection="linear"; opt.token_mlp="leff"
    model = utils.get_arch(opt).to(dev)
    dino = torch.hub.load("./dinov2", "dinov2_vitl14", source="local").to(dev).eval()
    for q in dino.parameters(): q.requires_grad_(False)
    crit = CharbonnierLoss().to(dev)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9,0.999), eps=1e-8, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, a.nepoch-a.warmup_epochs), eta_min=1e-6)
    log(f"model+dino ready | GPU {torch.cuda.get_device_name(0)}")

    start_ep, best_psnr = 1, 0.0
    if a.resume and os.path.isfile(a.resume):
        ck = torch.load(a.resume, map_location="cpu"); model.load_state_dict(ck["state_dict"])
        start_ep = ck.get("epoch",0)+1; best_psnr = ck.get("best_psnr",0.0)
        if "optimizer" in ck:
            try: optim.load_state_dict(ck["optimizer"]); log("restored optimizer state")
            except Exception as e: log(f"optimizer restore skipped: {e}")
        for _ in range(max(0, start_ep - 1 - a.warmup_epochs)): sched.step()
        log(f"resumed from {a.resume} @ ep{start_ep} | lr={optim.param_groups[0]['lr']:.2e}")

    tr = get_training_data(a.train_dir, {"patch_size":a.train_ps}, False)
    va = get_validation_data(a.val_dir, False)
    tl = torch.utils.data.DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                                     pin_memory=True, drop_last=True, persistent_workers=True)
    vl = torch.utils.data.DataLoader(va, batch_size=1, shuffle=False, num_workers=2)
    log(f"train={len(tr)} val={len(va)} | {len(tl)} it/epoch")

    def dfeat(x):
        xu = F.interpolate(x, size=(int(x.shape[2]*DPS/8), int(x.shape[3]*DPS/8)), mode="bilinear", align_corners=False)
        with torch.no_grad(): return dino.get_intermediate_layers(xu, 4, True)

    @torch.no_grad()
    def evaluate(stride=1):
        model.eval(); ps=[]; m_ = 8*a.win_size
        for i,d in enumerate(vl):
            if i % stride: continue
            tgt=d[0].to(dev); inp=d[1].to(dev); pt=d[2].to(dev); nm=d[3].to(dev)
            h,w = inp.shape[2], inp.shape[3]
            H=((h+m_)//m_)*m_; W=((w+m_)//m_)*m_
            ip=F.pad(inp,(0,W-w,0,H-h),'reflect'); pp=F.pad(pt,(0,W-w,0,H-h),'reflect'); np_=F.pad(nm,(0,W-w,0,H-h),'reflect')
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(ip, dfeat(ip), pp, np_)
            out = out.float().clamp(0,1)[:, :, :h, :w]
            o256=F.interpolate(out,(256,256),mode='area'); t256=F.interpolate(tgt.clamp(0,1),(256,256),mode='area')
            ps.append(utils.batch_PSNR(o256, t256, True).item())
        model.train(); torch.cuda.empty_cache(); return float(np.mean(ps))

    CW, SW = 0.999, 0.001
    import shutil, re
    topk = []
    for fn in os.listdir(mdir):
        mm = re.match(r"top_ep(\d+)_([\d.]+)\.pth", fn)
        if mm: topk.append((float(mm.group(2)), int(mm.group(1)), os.path.join(mdir, fn)))
    topk.sort(key=lambda x: -x[0]); topk = topk[:a.keep_topk]
    if topk: log(f"recovered top-{len(topk)} from disk: " + " ".join(f"{p:.3f}@{e}" for p,e,_ in topk))
    log(f"=== start training === | strong_aug(8way)={STRONG_AUG}  mixup={MIXUP}  keep_topk={a.keep_topk}")
    for ep in range(start_ep, a.nepoch+1):
        if ep <= a.warmup_epochs:
            for g in optim.param_groups: g["lr"] = a.lr*ep/max(1,a.warmup_epochs)
        model.train(); t0=time.time(); run=0.0; nit=0
        for i,d in enumerate(tl):
            tgt=d[0].to(dev,non_blocking=True); inp=d[1].to(dev,non_blocking=True)
            pt=d[2].to(dev,non_blocking=True); nm=d[3].to(dev,non_blocking=True)
            if MIXUP and ep > a.warmup_epochs:
                lam = float(np.random.beta(1.2, 1.2)); perm = torch.randperm(inp.size(0), device=dev)
                inp = lam*inp + (1-lam)*inp[perm]; tgt = lam*tgt + (1-lam)*tgt[perm]
                pt  = lam*pt  + (1-lam)*pt[perm];   nm  = lam*nm  + (1-lam)*nm[perm]
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(inp, dfeat(inp), pt, nm).clamp(0,1); tgt2 = tgt.clamp(0,1)
                loss = CW*crit(out, tgt2) + SW*(1-ssim_metric(out, tgt2))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); run += loss.item(); nit += 1
            if i % 50 == 49: log(f"ep{ep} it{i+1}/{len(tl)} loss={run/nit:.4f} {(i+1)/(time.time()-t0):.2f}it/s")
        if ep > a.warmup_epochs: sched.step()
        log(f"=== epoch {ep}/{a.nepoch} done | loss={run/max(1,nit):.4f} | {time.time()-t0:.1f}s | lr={optim.param_groups[0]['lr']:.2e}")
        torch.save({"epoch":ep,"state_dict":model.state_dict(),"best_psnr":best_psnr,"optimizer":optim.state_dict()}, os.path.join(mdir,"model_latest.pth"))
        if ep % a.eval_every == 0:
            pv = evaluate(); log(f"### EVAL ep{ep}: PSNR={pv:.4f} (best={best_psnr:.4f}) [correct test.py protocol @256]")
            cand = os.path.join(mdir, f"top_ep{ep}_{pv:.3f}.pth")
            torch.save({"epoch":ep,"state_dict":model.state_dict(),"best_psnr":pv,"optimizer":optim.state_dict()}, cand)
            topk.append((pv, ep, cand)); topk.sort(key=lambda x: -x[0])
            for _,_,p in topk[a.keep_topk:]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except OSError: pass
            topk = topk[:a.keep_topk]
            shutil.copyfile(topk[0][2], os.path.join(mdir,"model_best.pth"))
            if topk[0][1] == ep:
                best_psnr = pv
                log(f"### NEW BEST {best_psnr:.4f} @ ep{ep} | topK: " + " ".join(f"{p:.3f}@{e}" for p,e,_ in topk))
        if ep % a.save_every == 0:
            torch.save({"epoch":ep,"state_dict":model.state_dict(),"best_psnr":best_psnr,"optimizer":optim.state_dict()}, os.path.join(mdir,f"model_epoch_{ep}.pth"))
    log(f"=== training done, best PSNR {best_psnr:.4f} ===")

if __name__ == "__main__":
    main()
