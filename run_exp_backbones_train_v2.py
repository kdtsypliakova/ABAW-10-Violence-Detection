import argparse, json
from pathlib import Path

def load_preset(presets_path: Path, preset_name: str):
    if not presets_path.exists(): raise FileNotFoundError(f'Not found: {presets_path}')
    data = json.loads(presets_path.read_text(encoding='utf-8'))
    if preset_name not in data: raise KeyError(f'{preset_name!r} not in {list(data)}')
    return data[preset_name]

# ── Augmentation patches ──────────────────────────────────────────
def _patch_trivial_aug_wide_src(): return r'''
if bool(CFG.get("use_trivial_aug_wide", False)):
    try:
        import torch; import torchvision.transforms as T; import torchvision.transforms.functional as TF
        class TorchVideoTransformTrivialWide:
            def __init__(s,size,train):
                s.size=int(size);s.train=bool(train);s.aug=T.TrivialAugmentWide(num_magnitude_bins=int(CFG.get("trivial_aug_num_magnitude_bins",31)))
            def _f(s,x):
                if not isinstance(x,torch.Tensor):x=torch.from_numpy(x)
                if x.ndim==3 and x.shape[-1] in(1,3,4) and x.shape[0] not in(1,3,4):x=x.permute(2,0,1)
                if x.shape[0]==4:x=x[:3]
                if x.dtype==torch.uint8:x=x.float().div_(255.0)
                else:x=x.float();x=x/255.0 if x.max()>1.5 else x
                return x
            def __call__(s,x):
                x=s._f(x).clamp_(0,1);x=TF.resize(x,[s.size,s.size],antialias=True)
                if s.train:
                    xp=TF.to_pil_image((x*255).to(torch.uint8));xp=s.aug(xp);x=TF.pil_to_tensor(xp).float().div_(255.0)
                return TF.normalize(x,mean=IMAGENET_MEAN,std=IMAGENET_STD)
        train_tfms=TorchVideoTransformTrivialWide(CFG["img_size"],True);val_tfms=TorchVideoTransformTrivialWide(CFG["img_size"],False)
        globals()["train_tfms"]=train_tfms;globals()["val_tfms"]=val_tfms;print("[patch] TrivialAugWide enabled")
    except Exception as e: print(f"[patch][warn] TrivialAugWide failed: {e}")
'''

def _patch_strong_aug_src(): return r'''
if bool(CFG.get("strong_aug", False)):
    try:
        import torch; import torchvision.transforms.functional as TF
        class TorchVideoTransformStrong:
            def __init__(s,size,train):
                s.size=int(size);s.train=bool(train);s.smin=float(CFG.get("strong_aug_scale_min",0.7));s.smax=float(CFG.get("strong_aug_scale_max",1.0))
                s.cj=float(CFG.get("strong_aug_color_jitter",0.08));s.ep=float(CFG.get("strong_aug_erasing_prob",0.12))
            def _f(s,x):
                if not isinstance(x,torch.Tensor):x=torch.from_numpy(x)
                if x.ndim==3 and x.shape[-1] in(1,3,4) and x.shape[0] not in(1,3,4):x=x.permute(2,0,1)
                if x.shape[0]==4:x=x[:3]
                if x.dtype==torch.uint8:x=x.float().div_(255.0)
                else:x=x.float();x=x/255.0 if x.max()>1.5 else x
                return x
            def __call__(s,x):
                x=s._f(x)
                if s.train:
                    c,h,w=x.shape;sc=float(torch.empty(1).uniform_(s.smin,s.smax).item())
                    side=max(8,min(int((h*w*sc)**0.5),h,w));top=int(torch.randint(0,max(1,h-side+1),(1,)).item());left=int(torch.randint(0,max(1,w-side+1),(1,)).item())
                    x=TF.resized_crop(x,top,left,side,side,[s.size,s.size],antialias=True)
                    if float(torch.rand(1).item())<0.5:x=TF.hflip(x)
                    if s.cj>0:
                        x=TF.adjust_brightness(x,float(torch.empty(1).uniform_(1-s.cj,1+s.cj).item()))
                        x=TF.adjust_contrast(x,float(torch.empty(1).uniform_(1-s.cj,1+s.cj).item()))
                        x=TF.adjust_saturation(x,float(torch.empty(1).uniform_(1-s.cj,1+s.cj).item()));x=x.clamp_(0,1)
                    if s.ep>0 and float(torch.rand(1).item())<s.ep:
                        _,h2,w2=x.shape;eh=max(4,int(h2*float(torch.empty(1).uniform_(0.1,0.25).item())));ew=max(4,int(w2*float(torch.empty(1).uniform_(0.1,0.25).item())))
                        if eh<h2 and ew<w2:t=int(torch.randint(0,h2-eh+1,(1,)).item());l=int(torch.randint(0,w2-ew+1,(1,)).item());x[:,t:t+eh,l:l+ew]=0.0
                else: x=TF.resize(x,[s.size,s.size],antialias=True)
                return TF.normalize(x,mean=IMAGENET_MEAN,std=IMAGENET_STD)
        train_tfms=TorchVideoTransformStrong(CFG["img_size"],True);val_tfms=TorchVideoTransformStrong(CFG["img_size"],False)
        globals()["train_tfms"]=train_tfms;globals()["val_tfms"]=val_tfms;print("[patch] strong_aug enabled")
    except Exception as e: print(f"[patch][warn] strong_aug failed: {e}")
'''

def _patch_temporal_coherent_aug_src(): return r'''
if bool(CFG.get("temporal_coherent_aug", False)):
    try:
        import torch; import torchvision.transforms.functional as TF
        class TemporalCoherentAugTransform:
            def __init__(s,size,train):
                s.size=int(size);s.train=bool(train);s.smin=float(CFG.get("tc_aug_scale_min",0.75));s.smax=float(CFG.get("tc_aug_scale_max",1.0))
                s.cj=float(CFG.get("tc_aug_color_jitter",0.1));s.hp=float(CFG.get("tc_aug_hflip_prob",0.5));s.ep=float(CFG.get("tc_aug_erasing_prob",0.1))
                s._cs=None
            def set_clip_state(s):
                if not s.train:s._cs=None;return
                s._cs={'hf':float(torch.rand(1).item())<s.hp,'sc':float(torch.empty(1).uniform_(s.smin,s.smax).item()),
                       'br':float(torch.empty(1).uniform_(1-s.cj,1+s.cj).item()),'co':float(torch.empty(1).uniform_(1-s.cj,1+s.cj).item()),
                       'sa':float(torch.empty(1).uniform_(1-s.cj,1+s.cj).item()),'de':float(torch.rand(1).item())<s.ep,'cp':None,'erp':None}
            def _f(s,x):
                if not isinstance(x,torch.Tensor):x=torch.from_numpy(x)
                if x.ndim==3 and x.shape[-1] in(1,3,4) and x.shape[0] not in(1,3,4):x=x.permute(2,0,1)
                if x.shape[0]==4:x=x[:3]
                if x.dtype==torch.uint8:x=x.float().div_(255.0)
                else:x=x.float();x=x/255.0 if x.max()>1.5 else x
                return x
            def __call__(s,x):
                x=s._f(x)
                if s.train and s._cs is not None:
                    st=s._cs;c,h,w=x.shape
                    if st['cp'] is None:
                        side=max(8,min(int((h*w*st['sc'])**0.5),h,w));st['cp']=(int(torch.randint(0,max(1,h-side+1),(1,)).item()),int(torch.randint(0,max(1,w-side+1),(1,)).item()),side)
                    t,l,sd=st['cp'];x=TF.resized_crop(x,t,l,sd,sd,[s.size,s.size],antialias=True)
                    if st['hf']:x=TF.hflip(x)
                    if s.cj>0:x=TF.adjust_brightness(x,st['br']);x=TF.adjust_contrast(x,st['co']);x=TF.adjust_saturation(x,st['sa']);x=x.clamp_(0,1)
                    if st['de']:
                        _,h2,w2=x.shape
                        if st['erp'] is None:
                            eh=max(4,int(h2*float(torch.empty(1).uniform_(0.1,0.25).item())));ew=max(4,int(w2*float(torch.empty(1).uniform_(0.1,0.25).item())))
                            st['erp']=(int(torch.randint(0,h2-eh+1,(1,)).item()),int(torch.randint(0,w2-ew+1,(1,)).item()),eh,ew) if eh<h2 and ew<w2 else False
                        if st['erp'] and st['erp'] is not False:et,el,eh,ew=st['erp'];x[:,et:et+eh,el:el+ew]=0.0
                else: x=TF.resize(x,[s.size,s.size],antialias=True)
                return TF.normalize(x,mean=IMAGENET_MEAN,std=IMAGENET_STD)
        _og=DVDClipFrameLabelDataset.__getitem__
        def _gi(self,idx):
            if hasattr(self.transform,'set_clip_state'):self.transform.set_clip_state()
            return _og(self,idx)
        DVDClipFrameLabelDataset.__getitem__=_gi
        train_tfms=TemporalCoherentAugTransform(CFG["img_size"],True);val_tfms=TemporalCoherentAugTransform(CFG["img_size"],False)
        globals()["train_tfms"]=train_tfms;globals()["val_tfms"]=val_tfms;print("[patch] temporal_coherent_aug enabled")
    except Exception as e: print(f"[patch][warn] temporal_coherent_aug failed: {e}")
'''

# ── Sampling / Loss / Optimizer patches ───────────────────────────
def _patch_strict_pos_sampling_src(): return r'''
if bool(CFG.get("strict_pos_sampling", False)):
    try:
        import numpy as np; _osc=DVDClipFrameLabelDataset._choose_start
        def _cpc(y,s,cl,fs):
            ix=np.clip(s+np.arange(int(cl))*int(fs),0,len(y)-1).astype(int);ys=y[ix];v=ys!=-1
            return (int((ys[v]==1).sum()),int(v.sum())) if int(v.sum())>0 else (0,0)
        def _css(self,y,hp,idx,row,view=0):
            n=len(y)
            if n<=self.span:return 0
            if not self.train or self.clip_mode=="center":return _osc(self,y,hp,idx,row,view=view)
            mx=n-self.span
            if hp==1 and np.random.rand()<float(self.pos_sample_prob):
                pi=np.where(y==1)[0]
                if len(pi)>0:
                    tr=int(CFG.get("strict_pos_search_trials",24));mp=max(int(CFG.get("strict_pos_min_pos_frames",3)),int(np.ceil(float(CFG.get("strict_pos_min_pos_ratio",0.2))*self.clip_len)))
                    bs,bp=None,-1
                    for _ in range(max(1,tr)):
                        c=int(np.random.choice(pi));j=int(np.random.randint(-self.span//2,self.span//2+1));s=int(np.clip(c-self.span//2+j,0,mx))
                        pc,_=_cpc(y,s,self.clip_len,self.frame_step)
                        if pc>bp:bp,bs=pc,s
                        if pc>=mp:return s
                    for _ in range(max(4,tr//3)):
                        s=int(np.random.randint(0,mx+1));pc,_=_cpc(y,s,self.clip_len,self.frame_step)
                        if pc>bp:bp,bs=pc,s
                        if pc>=mp:return s
                    if bs is not None:return int(bs)
            return _osc(self,y,hp,idx,row,view=view)
        DVDClipFrameLabelDataset._choose_start=_css;print("[patch] strict_pos_sampling enabled")
    except Exception as e: print(f"[patch][warn] strict_pos_sampling failed: {e}")
'''

def _patch_train_n_views_src(): return r'''
try:
    _tv = max(1, int(CFG.get("train_n_views", 1)))
    if _tv > 1:
        _olen = DVDClipFrameLabelDataset.__len__
        _oget = DVDClipFrameLabelDataset.__getitem__

        def _len_tv(self):
            if getattr(self, "train", False):
                return len(self.meta) * _tv
            return _olen(self)

        def _get_tv(self, idx):
            if getattr(self, "train", False) and _tv > 1:
                idx = int(idx)
                vid_idx = idx // _tv
                return _oget(self, vid_idx)
            return _oget(self, idx)

        DVDClipFrameLabelDataset.__len__ = _len_tv
        DVDClipFrameLabelDataset.__getitem__ = _get_tv
        print(f"[patch] train_n_views enabled: {_tv}x clips/video/epoch")
    else:
        print("[patch] train_n_views=1 (default)")
except Exception as e:
    print(f"[patch][warn] train_n_views patch failed: {e}")
'''

def _patch_class_weights_src(): return r'''
if bool(CFG.get("enable_cb_loss", False)):
    try:
        import torch;beta=float(CFG.get("cb_beta",0.999))
        c0,c1=max(int(globals().get("tr0",0)),1),max(int(globals().get("tr1",0)),1)
        w0=(1-beta)/max(1-beta**c0,1e-12);w1=(1-beta)/max(1-beta**c1,1e-12);s=w0+w1;w0,w1=2*w0/s,2*w1/s
        w1*=float(CFG.get("cb_pos_extra_mult",1.0));s2=w0+w1;w0,w1=2*w0/s2,2*w1/s2
        class_weights=torch.tensor([w0,w1],dtype=torch.float32,device=device);globals()["class_weights"]=class_weights
        print(f"[patch] CB weights: {class_weights.tolist()}")
    except Exception as e: print(f"[patch][warn] CB weights failed: {e}")
'''

def _patch_advanced_losses_src(): return r'''
try:
    import torch,torch.nn as nn,torch.nn.functional as F;_oml=masked_ce_loss
    def _tc():return max(int(globals().get("tr0",1)),1),max(int(globals().get("tr1",1)),1)
    def _tp(d):c0,c1=_tc();p=torch.tensor([c0,c1],dtype=torch.float32,device=d);return p/p.sum()
    def _asl(l2,y2,cw=None):
        nc=l2.shape[1];tgt=F.one_hot(y2,nc).float();pr=torch.softmax(l2,1);xp=pr.clamp(1e-8,1);xn=(1-pr).clamp(1e-8,1)
        cl=float(CFG.get("asl_clip",0.05))
        if cl>0:xn=(xn+cl).clamp(max=1)
        gp,gn=float(CFG.get("asl_gamma_pos",0)),float(CFG.get("asl_gamma_neg",4))
        w=torch.pow((1-(xp*tgt+xn*(1-tgt))).clamp(1e-8),gp*tgt+gn*(1-tgt)) if gp>0 or gn>0 else 1.0
        loss=(-(tgt*torch.log(xp)+(1-tgt)*torch.log(xn))*w).sum(1)
        if cw is not None:loss=loss*cw[y2]
        return loss.mean()
    def masked_ce_loss(logits,y_seq,mask,class_weights=None,label_smoothing=0.0,loss_type="ce",focal_gamma=2.0):
        B,K,T=logits.shape;l2=logits.permute(0,2,1).reshape(B*T,K);y2=y_seq.reshape(B*T);m2=mask.reshape(B*T);l2=l2[m2];y2=y2[m2].long()
        if l2.numel()==0:return(logits*0).sum()
        lt=str(loss_type).lower()
        if lt in{"logit_adj_ce","logit_adjust","logit_adjustment"}:
            tau=float(CFG.get("logit_adj_tau",1));return nn.CrossEntropyLoss(weight=class_weights,label_smoothing=label_smoothing)(l2+tau*torch.log(_tp(l2.device).clamp(1e-8)).unsqueeze(0),y2)
        if lt in{"balanced_softmax","bsce"}:
            c0,c1=_tc();return nn.CrossEntropyLoss(label_smoothing=label_smoothing)(l2+torch.log(torch.tensor([c0,c1],dtype=l2.dtype,device=l2.device).clamp(1)).unsqueeze(0),y2)
        if lt in{"asl","asymmetric"}:return _asl(l2,y2,class_weights)
        return _oml(logits=logits,y_seq=y_seq,mask=mask,class_weights=class_weights,label_smoothing=label_smoothing,loss_type=loss_type,focal_gamma=focal_gamma)
    globals()["masked_ce_loss"]=masked_ce_loss;print("[patch] advanced losses enabled")
except Exception as e: print(f"[patch][warn] advanced losses failed: {e}")
'''

def _patch_optimizer_llrd_src(): return r'''
if CFG.get("backbone")!="resnet18":
    try:
        import re,torch;ld=float(CFG.get("layer_decay",1.0))
        if 0<ld<1:
            blr=float(CFG.get("base_lr",1e-4));blrm=float(CFG.get("backbone_lr_mult",0.1));wd=float(CFG.get("adamw_weight_decay",5e-4))
            ht=("head","lstm","drop","temporal_head","proj")
            def ih(n):return any(t in n for t in ht)
            def lid(n):
                m=re.search(r"encoder\.encoder\.layer\.(\d+)",n)
                if m:return int(m.group(1))+1
                m=re.search(r"(?:features|layers|stages|blocks)\.(\d+)",n)
                if m:return int(m.group(1))+1
                return 0
            bids=[lid(n) for n,p in model.named_parameters() if p.requires_grad and not ih(n)];ml=max(bids) if bids else 0
            gd={}
            for n,p in model.named_parameters():
                if not p.requires_grad:continue
                nd=p.ndim==1 or n.endswith(".bias");lr=blr if ih(n) else blr*blrm*(ld**(ml-lid(n)));w=0.0 if nd else wd
                gd.setdefault((float(lr),float(w)),[]).append(p)
            pg=[{"params":ps,"lr":lr,"weight_decay":w} for(lr,w),ps in sorted(gd.items())]
            optimizer=torch.optim.AdamW(pg,lr=blr);globals()["optimizer"]=optimizer
            print(f"[patch] LLRD: decay={ld}, max_layer={ml}, groups={len(pg)}")
    except Exception as e: print(f"[patch][warn] LLRD failed: {e}")
'''

# ── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--notebook', default='exp_backbones_v2_skeleton_ready.ipynb')
    ap.add_argument('--presets-path', default='improved_presets.json')
    ap.add_argument('--preset', default=None)
    ap.add_argument('--overrides', default='{}')
    ap.add_argument('--disable-patches', action='store_true')
    args = ap.parse_args()

    nb = json.loads(Path(args.notebook).read_text(encoding='utf-8'))
    merged = {}
    if args.preset:
        print(f'Preset: {args.preset}')
        merged.update(load_preset(Path(args.presets_path), args.preset))
        merged['train_preset'] = None
        merged['presets_path'] = str(Path(args.presets_path))
    merged.update(json.loads(args.overrides))

    cells = [6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24]
    g = {'__name__': '__main__'}

    for idx in cells:
        cell = nb['cells'][idx]
        if cell.get('cell_type') != 'code': continue
        src = ''.join(cell.get('source', []))

        if idx == 7 and merged:
            src += f"\nCFG.update({repr(merged)})\nprint('Applied overrides')\n"
            src += "print({k:CFG.get(k) for k in ['backbone','epochs','base_lr','backbone_lr_mult','frame_step','train_n_views','loss_type','temporal_coherent_aug','bilstm_hidden']})\n"
        if idx == 11 and not args.disable_patches:
            src += "\n" + _patch_trivial_aug_wide_src() + "\n" + _patch_strong_aug_src() + "\n" + _patch_temporal_coherent_aug_src()
        if idx == 12 and not args.disable_patches:
            src += "\n" + _patch_strict_pos_sampling_src() + "\n" + _patch_train_n_views_src()
        if idx == 18 and not args.disable_patches:
            src += "\n" + _patch_class_weights_src() + "\n" + _patch_advanced_losses_src()
        if idx == 20:
            src = "class _D:\n def update(s,*a,**k):return None\ndef _sd(*a,**k):return _D() if k.get('display_id',False) else None\ndisplay=_sd\n" + src
        if idx == 22 and not args.disable_patches:
            src += "\n" + _patch_optimizer_llrd_src()

        exec(compile(src, f'cell_{idx}', 'exec'), g, g)

    print('=== TRAIN FINISHED ===')
    print('run_dir:', g.get('run_dir'))
    print('best_f1:', g.get('best_f1'))

if __name__ == '__main__':
    main()
