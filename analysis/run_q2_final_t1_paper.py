from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from q2_protocols import blocked_mixed_condition_split
from run_representation_conditioning_diagnostic import PairTCN
from run_sequence_representation_benchmark import (
    TCNEncoder,
    WindowDataset,
    count_params,
    load_sources,
    metric_dict,
    normalize_sources,
    predict_model,
    seed_everything,
    train_normalizer,
)

MODELS = (
    "CNN",
    "GRU",
    "LSTM",
    "Transformer",
    "VI-TCN",
    "VI+TF-TCN",
    "RA-FBG-TCN",
)


class SliceModel(nn.Module):
    def __init__(self, indices: tuple[int, ...]):
        super().__init__()
        self.indices = indices

    def select(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, self.indices]


class CNNBaseline(SliceModel):
    def __init__(self):
        super().__init__((0, 1, 2, 3))
        self.net = nn.Sequential(
            nn.Conv1d(4, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 48, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        z = self.select(x).transpose(1, 2)
        h = self.net(z)[:, :, -1]
        return self.head(h).squeeze(-1), None


class RNNBaseline(SliceModel):
    def __init__(self, kind: str):
        super().__init__((0, 1, 2, 3))
        cls = nn.GRU if kind == "GRU" else nn.LSTM
        self.rnn = cls(4, 32, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(32, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, x):
        z = self.select(x)
        out, _ = self.rnn(z)
        return self.head(out[:, -1]).squeeze(-1), None


class TransformerBaseline(SliceModel):
    def __init__(self, window: int = 64):
        super().__init__((0, 1, 2, 3))
        d = 32
        self.proj = nn.Linear(4, d)
        self.pos = nn.Parameter(torch.zeros(1, window, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=64, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, x):
        z = self.select(x)
        h = self.proj(z) + self.pos[:, : z.shape[1]]
        h = self.enc(h)
        return self.head(h[:, -1]).squeeze(-1), None


class VITCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = TCNEncoder(2, 24)
        self.head = nn.Sequential(nn.Linear(24, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, x):
        h = self.enc(x[:, :, (0, 1)])
        return self.head(h).squeeze(-1), None


def factory(name: str, window: int):
    if name == "CNN": return CNNBaseline()
    if name == "GRU": return RNNBaseline("GRU")
    if name == "LSTM": return RNNBaseline("LSTM")
    if name == "Transformer": return TransformerBaseline(window)
    if name == "VI-TCN": return VITCN()
    if name == "VI+TF-TCN": return PairTCN((4, 5), None)
    if name == "RA-FBG-TCN": return PairTCN((2, 3), None)
    raise ValueError(name)


def train_earlystop(model, train_loader, val_loader, device, *, max_epochs, min_epochs, patience, min_delta, lr):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_mae = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    anchor = float("inf")
    stale = 0
    curves = []
    for epoch in range(1, max_epochs + 1):
        model.train(); total = 0.0; n = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(y); n += len(y)
        yv, pv, _, _ = predict_model(model, val_loader, device)
        vm = metric_dict(yv, pv); mae = float(vm["MAE"])
        curves.append({"epoch": epoch, "train_mse": total/max(n,1), **{f"val_{k}":v for k,v in vm.items()}})
        if mae < best_mae:
            best_mae = mae; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict())
        if mae < anchor - min_delta:
            anchor = mae; stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    model.load_state_dict(best_state)
    return curves, best_epoch


def pred_frame(raw_sources, ds, y, pred):
    rows=[]
    for source_id, end in ds.index:
        src=raw_sources[source_id]
        rows.append({
            "source_name":src["name"], "profile":src["profile"], "rate":src["rate"],
            "current_A":float(src["x"][end,1]), "voltage_V":float(src["x"][end,0]),
            "y_true":float(y[len(rows)]), "y_pred":float(pred[len(rows)]),
        })
    return pd.DataFrame(rows)


def conformal(cal: pd.DataFrame, test: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    scores=np.sort(np.abs(cal.y_true.to_numpy()-cal.y_pred.to_numpy()))
    rows=[]; interval_frames=[]
    for alpha in (0.10,0.05):
        k=int(math.ceil((len(scores)+1)*(1-alpha))); k=min(max(k,1),len(scores)); q=float(scores[k-1])
        y=test.y_true.to_numpy(float); p=test.y_pred.to_numpy(float)
        lo=np.clip(p-q,0,1); hi=np.clip(p+q,0,1); cov=(y>=lo)&(y<=hi); width=hi-lo
        score=width+(2/alpha)*(np.maximum(lo-y,0)+np.maximum(y-hi,0))
        rows.append({"nominal_coverage":1-alpha,"alpha":alpha,"PICP":float(cov.mean()),"MPIW":float(width.mean()),"mean_interval_score":float(score.mean()),"q_abs_residual":q,"n_cal":len(cal),"n_test":len(test)})
        f=test.copy(); f["alpha"]=alpha; f["lower"]=lo; f["upper"]=hi; f["covered"]=cov; interval_frames.append(f)
    pd.DataFrame(rows).to_csv(out_dir/"uq_summary.csv",index=False)
    pd.concat(interval_frames,ignore_index=True).to_csv(out_dir/"test_intervals.csv",index=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/extracted/SiC-18"))
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--window",type=int,default=64)
    p.add_argument("--train-stride",type=int,default=2)
    p.add_argument("--eval-stride",type=int,default=1)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--max-epochs",type=int,default=50)
    p.add_argument("--min-epochs",type=int,default=10)
    p.add_argument("--patience",type=int,default=7)
    p.add_argument("--min-delta",type=float,default=5e-5)
    p.add_argument("--lr",type=float,default=1e-3)
    a=p.parse_args()
    seed_everything(a.seed)
    raw=load_sources(a.data); split=blocked_mixed_condition_split(raw,window=a.window)
    mean,std=train_normalizer(split.train)
    tr=normalize_sources(split.train,mean,std); va=normalize_sources(split.validation,mean,std); ca=normalize_sources(split.calibration,mean,std); te=normalize_sources(split.test,mean,std)
    trds=WindowDataset(tr,a.window,a.train_stride); vads=WindowDataset(va,a.window,a.eval_stride); cads=WindowDataset(ca,a.window,a.eval_stride); teds=WindowDataset(te,a.window,a.eval_stride)
    trl=DataLoader(trds,batch_size=a.batch_size,shuffle=True,num_workers=0); val=DataLoader(vads,batch_size=a.batch_size*2,shuffle=False,num_workers=0); cal=DataLoader(cads,batch_size=a.batch_size*2,shuffle=False,num_workers=0); test=DataLoader(teds,batch_size=a.batch_size*2,shuffle=False,num_workers=0)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=factory(a.model,a.window)
    curves,best=train_earlystop(model,trl,val,dev,max_epochs=a.max_epochs,min_epochs=a.min_epochs,patience=a.patience,min_delta=a.min_delta,lr=a.lr)
    yc,pc,_,_=predict_model(model,cal,dev); yt,pt,_,_=predict_model(model,test,dev)
    m={"protocol":split.name,"split_id":split.split_id,"model":a.model,"seed":a.seed,"best_epoch":best,"params":count_params(model),"n_train_windows":len(trds),"n_val_windows":len(vads),"n_cal_windows":len(cads),"n_test_windows":len(teds),**metric_dict(yt,pt)}
    a.out_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([m]).to_csv(a.out_dir/"point_metrics.csv",index=False); pd.DataFrame(curves).to_csv(a.out_dir/"validation_curve.csv",index=False)
    cf=pred_frame(split.calibration,cads,yc,pc); tf=pred_frame(split.test,teds,yt,pt); cf.to_csv(a.out_dir/"calibration_predictions.csv",index=False); tf.to_csv(a.out_dir/"test_predictions.csv",index=False)
    if a.model=="RA-FBG-TCN": conformal(cf,tf,a.out_dir/"uq")
    print(pd.DataFrame([m]).to_string(index=False))

if __name__=="__main__": main()
