# -*- coding: utf-8 -*-
"""
Fair comparison: ALL models use single-stage direct quantile (q10,q50,q90).
Only difference: SemiFrozen freezes CNN, baselines train everything.
"""
import os, sys, time, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
from copy import deepcopy
import warnings; warnings.filterwarnings("ignore")

# The interval-forecasting engine (all architectures + the single-stage direct
# quantile pipeline) lives in the sibling module; import it from this script's
# own directory so the repository is self-contained.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
from multimodel_base_residual_quantile import (
    ModelFactory, apply_feature_engineering, create_sequences,
    build_known_future_timefeats, SeqDataset, GRN, _metrics,
    train_model, predict_model, build_scheduler, lr_warmup_search,
    safe_torch_save, set_seed, _fmt_dur, ensure_dir,
    SinusoidalPositionalEncoding, CNNFrontTFT
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANTILES = [0.10, 0.50, 0.90]
OUT = _REPO / "results" / "interval_fair"
DATA = str(_REPO / "data" / "belgium.csv")

# ===================== CWC (standard Khosravi, matches paper Eq. 19) =====================
def cwc_standard(picp, mpiw, mu=0.80, eta=50.0):
    """CWC = MPIW * (1 + 1[PICP<mu] * exp(-eta*(PICP-mu))). No over-coverage penalty."""
    if picp < mu: return float(mpiw*(1.0+np.exp(-eta*(picp-mu))))
    return float(mpiw)

def interval_metrics(yt,yl,yu,tc=0.80,ytb=None):
    yt=np.asarray(yt).reshape(-1); lo=np.minimum(np.asarray(yl).reshape(-1),np.asarray(yu).reshape(-1)); hi=np.maximum(np.asarray(yl).reshape(-1),np.asarray(yu).reshape(-1))
    p=float(np.mean((yt>=lo)&(yt<=hi))); w=float(np.mean(hi-lo)); a=p-tc; c=cwc_standard(p,w,tc)
    dr=float(np.max(ytb)-np.min(ytb)) if ytb is not None else float(np.max(yt)-np.min(yt))
    return {"PICP(%)":p*100,"ACE(%)":a*100,"MPIW":w,"PINAW(%)":float(w/(dr+1e-8))*100,"CWC":c,"CrossRate(%)":0.0}

# ===================== Multi-Pinball Loss =====================
class MultiPinballLoss(nn.Module):
    def __init__(self,qs): super().__init__(); self.qs=[float(q) for q in qs]
    def forward(self,pred,target):
        target=target.view(-1,1); errors=target-pred
        return sum(torch.max(q*errors[:,[i]],(q-1)*errors[:,[i]]).mean() for i,q in enumerate(self.qs))/len(self.qs)

# ===================== Direct Quantile Wrapper =====================
class DirectQuantileWrapper(nn.Module):
    """Converts any base model to single-stage 3-quantile output."""
    def __init__(self, backbone, base_state_dict=None, freeze_cnn_only=False, freeze_lstm=False):
        super().__init__()
        self.backbone = backbone

        # Load base weights (partial matching)
        if base_state_dict is not None:
            own = self.backbone.state_dict()
            n = 0
            for k in base_state_dict:
                if k in own and own[k].shape == base_state_dict[k].shape:
                    own[k].copy_(base_state_dict[k]); n += 1
            self.backbone.load_state_dict(own)
            print(f"  Loaded {n} backbone params")

        # Freeze strategy
        if freeze_cnn_only and hasattr(self.backbone, 'convs'):
            for p in self.backbone.convs.parameters():
                p.requires_grad = False
            print(f"  CNN frozen", end="")
        if freeze_lstm and hasattr(self.backbone, 'enc_lstm'):
            for p in self.backbone.enc_lstm.parameters():
                p.requires_grad = False
            print(f" + LSTM frozen", end="")
        if freeze_cnn_only or freeze_lstm:
            print()

        d = getattr(self.backbone, 'feature_dim', 64)
        # Replace fc_out with 3-quantile output
        self.backbone.fc_out = nn.Linear(d, len(QUANTILES))

    def forward(self, enc_seq, dec_known):
        return self.backbone(enc_seq, dec_known)  # Returns [B, 3]

# ===================== Data =====================
def prepare_data(pred_step):
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d %H:%M", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(by="date").reset_index(drop=True)
    bc = ["humidity","speed","mintem","temperature","maxtem","load"]
    df[bc] = (df[bc].replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both").fillna(method="bfill").fillna(method="ffill"))
    df_fe, ec = apply_feature_engineering(df, "date")
    N = len(df_fe); ti = ec.index("load")
    te, ve = int(N*0.60), int(N*0.80)
    Xa = df_fe[ec].values.astype(np.float32); sc = MinMaxScaler((0,1)); sc.fit(Xa[:te])
    Xs = sc.transform(Xa).astype(np.float32)
    es, dk, lb, ld = create_sequences(Xs, df_fe["date"].values, 96, ti, pred_step)
    ms = len(lb); tr = min(max(0,te-96-pred_step+1),ms); vl = min(max(0,ve-96-pred_step+1),ms)
    Xt,Kt,yt = es[:tr],dk[:tr],lb[:tr]; Xv,Kv,yv = es[tr:vl],dk[tr:vl],lb[tr:vl]; Xe,Ke,ye = es[vl:],dk[vl:],lb[vl:]
    print(f"  Data: tr={len(Xt)} val={len(Xv)} te={len(Xe)}")
    bs, pin = 32, True
    tl = DataLoader(SeqDataset(Xt,Kt,yt), bs, shuffle=True, pin_memory=pin)
    vl2 = DataLoader(SeqDataset(Xv,Kv,yv), bs, shuffle=False, pin_memory=pin)
    el = DataLoader(SeqDataset(Xe,Ke,ye), bs, shuffle=False, pin_memory=pin)
    return tl, vl2, el, Xa, ti, sc

def inv(arr, Xa, ti, sc):
    arr = np.asarray(arr).reshape(-1); d = np.zeros((len(arr),Xa.shape[1]),dtype=np.float32); d[:,ti]=arr
    return sc.inverse_transform(d)[:,ti]

# ===================== Run one model =====================
def run_single(model_name, pred_step, freeze_cnn=False, freeze_lstm=False, lr=1e-3, wu=5, ep=50, pat=15):
    torch.manual_seed(42); np.random.seed(42)
    label = f"freeze_cnn={freeze_cnn},freeze_lstm={freeze_lstm}"
    print(f"\n{'='*50}\n{model_name} step={pred_step} {label}\n{'='*50}")
    tl, vl, el, Xa, ti, sc = prepare_data(pred_step)
    d_in, dk_in = tl.dataset.enc_seqs.shape[2], tl.dataset.dec_known.shape[1]

    # Build base model for weight init
    base_model = ModelFactory.create_base_model(model_name, d_in, dk_in, {
        "conv_out_each":12,"conv_kernels":[3,5,7],"conv_dilations":[1,2,4],
        "tft_d_model":64,"tft_n_heads":4,"tft_lstm_layers":1,"tft_dropout":0.2,
        "baseline_d_model":64,"baseline_n_heads":4,"baseline_dropout":0.2,
        "lstm_layers":2,"gru_layers":2,"tcn_layers":4,"transformer_layers":2,
        "seq_length":96, "conv_act":"relu"
    })
    base_sd = base_model.state_dict()

    # Wrap with Direct Quantile
    model = DirectQuantileWrapper(base_model, base_state_dict=base_sd, freeze_cnn_only=freeze_cnn, freeze_lstm=freeze_lstm).to(DEVICE)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {n_tr:,}")

    crit = MultiPinballLoss(QUANTILES)
    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=lr)
    sch = build_scheduler(opt, total_epochs=ep, warmup_epochs=wu, sched_type="cosine", min_lr=0.0)
    cp = OUT / f"{model_name}_{'semi' if freeze_cnn else 'full'}_step{pred_step}.pth"
    cp.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tr_h, va_h, sec, _ = train_model(model, tl, vl, crit, opt, ep, DEVICE, save_path=cp, scheduler=sch, early_stop_patience=pat, early_stop_min_delta=0.0, grad_clip_norm=1.0, tag=f"[{model_name}-{pred_step}] ")

    model.load_state_dict(torch.load(str(cp), map_location=DEVICE))
    tep, tet = predict_model(model, el, DEVICE)
    yt = inv(tet, Xa, ti, sc)
    yq = np.column_stack([inv(tep[:,i], Xa, ti, sc) for i in range(len(QUANTILES))])
    yq10, yq50, yq90 = yq[:,0], yq[:,1], yq[:,2]; yq90 = np.maximum(yq90, yq10+1e-4)

    pt = _metrics(yt, yq50)
    itv = interval_metrics(yt, yq10, yq90, tc=0.80, ytb=inv(tl.dataset.labels, Xa, ti, sc))
    if freeze_cnn and freeze_lstm: label = "Re-TFT (FullFreeze)"
    elif freeze_cnn: label = "Re-TFT (SemiFrozen)"
    else: label = model_name
    print(f"  => RMSE={pt['RMSE']:.1f} R2={pt['R2']:.4f} | PICP={itv['PICP(%)']:.1f}% MPIW={itv['MPIW']:.0f} CWC={itv['CWC']:.0f}")
    return {"model":label,"step":pred_step,**pt,**itv,"lr":lr,"time":sec}

if __name__ == "__main__":
    all_results = []

    MODELS = ["CNNFrontTFT","LSTM","GRU","TCN","Transformer","TFT","NBEATS","DLinear","PatchTST","iTransformer","TimesNet"]

    # ALL models: same architecture, same 20 epochs. Only difference: freeze scope.
    for step in [1,4,24,96]:
        for m in MODELS:
            # No-freeze baseline
            all_results.append(run_single(m, step, freeze_cnn=False, freeze_lstm=False, lr=1e-3, ep=50))
        # CNNFrontTFT: SemiFrozen (CNN freeze only)
        all_results.append(run_single("CNNFrontTFT", step, freeze_cnn=True, freeze_lstm=False, lr=1e-3, ep=50))
        # CNNFrontTFT: Full freeze (CNN+LSTM frozen, like original)
        all_results.append(run_single("CNNFrontTFT", step, freeze_cnn=True, freeze_lstm=True, lr=1e-3, ep=50))

    # Save
    df = pd.DataFrame(all_results)
    cols = ["model","step","RMSE","MAE","MAPE","R2","PICP(%)","ACE(%)","MPIW","PINAW(%)","CWC","time"]
    df = df[cols].sort_values(["step","model"])
    df.to_csv(str(OUT/"fair_comparison.csv"), index=False)

    print("\n" + "="*90)
    print("FAIR COMPARISON: All models = single-stage direct quantile. Same CWC.")
    print("="*90)
    for step in [1,4,24,96]:
        print(f"\n--- Step {step} ---")
        sub = df[df.step==step]
        print(f"{'Model':<25}{'RMSE':>8}{'R2':>8}{'PICP':>8}{'MPIW':>8}{'ACE':>8}{'CWC':>10}")
        print("-"*75)
        for _,r in sub.iterrows():
            print(f"{r['model']:<25}{r['RMSE']:>8.1f}{r['R2']:>8.4f}{r['PICP(%)']:>7.1f}%{r['MPIW']:>7.0f}{r['ACE(%)']:>7.1f}%{r['CWC']:>9.0f}")
