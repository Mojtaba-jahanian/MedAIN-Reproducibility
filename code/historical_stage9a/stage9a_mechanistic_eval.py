from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import scipy.linalg
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HOME = Path.home()
PKG = HOME / "MedAIN_STAGE9A_PROSPECTIVE_PILOT_V1"
STAGE9 = HOME / "MedAIN_V19_STAGE9"
ROOT = HOME / "MedAIN_V10"
REPAIR = HOME / "MedAIN_V13_IDENTITY_REPAIR"
PROTOCOL = PKG / "STAGE9A_PILOT_PROTOCOL_LOCK_v1_0.json"

DOMAINS = [
    "BinRushed",
    "Magrabia",
    "MESSIDOR_Base1",
    "MESSIDOR_Base2",
    "MESSIDOR_Base3",
]
SEED = 1103
RANK = 2
ALPHA = 0.5
BETA = 1e-3
BASIS_N = 24
IGNORE_INDEX = 255

def load_anchor_module():
    p = PKG / "stage9_anchor_train.py"
    spec = importlib.util.spec_from_file_location("stage9_anchor", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

A = load_anchor_module()

def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def select_basis_rows(train_rows, target_domain):
    ranked = sorted(
        train_rows,
        key=lambda r: stable_hash(
            f"MedAIN_STAGE9A_V1|{target_domain}|{r['sample_uid']}"
        )
    )
    chosen = ranked[:BASIS_N]
    if len(chosen) != BASIS_N:
        raise RuntimeError("Insufficient basis rows")
    return chosen

def covnorm(C):
    C = 0.5 * (C + C.T)
    tr = float(np.trace(C))
    return C if tr <= 1e-12 else C / tr

def basis_metrics(U, A0, T0):
    A1 = covnorm(A0)
    T1 = covnorm(T0)
    capture = float(np.trace(U.T @ A1 @ U) / max(float(np.trace(A1)), 1e-12))
    leakage = float(np.trace(U.T @ T1 @ U) / max(float(np.trace(T1)), 1e-12))
    orth = float(np.linalg.norm(U.T @ U - np.eye(U.shape[1]), ord="fro"))
    efficiency = float(capture / (leakage + 1e-12))
    return {
        "acquisition_capture_ratio": capture,
        "task_leakage_ratio": leakage,
        "protection_efficiency": efficiency,
        "orthogonality_error": orth,
    }

def solve_tp(CA, CT):
    A1 = covnorm(CA)
    T1 = covnorm(CT)
    B = T1 + BETA * np.eye(T1.shape[0], dtype=np.float64)
    vals, vecs = scipy.linalg.eigh(A1, B, check_finite=True)
    order = np.argsort(vals)[::-1][:RANK]
    U = vecs[:, order]
    U, _ = np.linalg.qr(U)
    return U.astype(np.float32), vals[order].astype(np.float64)

def solve_ao(CA):
    A1 = covnorm(CA)
    vals, vecs = np.linalg.eigh(A1)
    order = np.argsort(vals)[::-1][:RANK]
    U = vecs[:, order]
    U, _ = np.linalg.qr(U)
    return U.astype(np.float32), vals[order].astype(np.float64)

def solve_rand(channels, target_domain, stage):
    seed_hex = stable_hash(f"MedAIN_STAGE9A_RAND_V1|{target_domain}|{stage}")[:16]
    rng = np.random.default_rng(int(seed_hex, 16))
    M = rng.standard_normal((channels, RANK))
    U, _ = np.linalg.qr(M)
    return U[:, :RANK].astype(np.float32)

def estimate_covariances(model, rows, class_weights, device):
    ds = A.RigaDataset(rows, augment=False, seed=SEED)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    CA1 = np.zeros((64,64), dtype=np.float64)
    CT1 = np.zeros((64,64), dtype=np.float64)
    CA2 = np.zeros((128,128), dtype=np.float64)
    CT2 = np.zeros((128,128), dtype=np.float64)
    nA = 0
    nT = 0
    perturbations = ["brightness","contrast","gamma","sharpness"]

    model.eval()
    for x,y,_ in loader:
        x=x.to(device); y=y.to(device)
        model.zero_grad(set_to_none=True)

        logits,f1,f2 = model(x, apply_medain=False, return_raw_features=True)
        loss,_,_ = A.loss_fn(logits,y,class_weights)
        g1,g2 = torch.autograd.grad(loss,[f1,f2],retain_graph=False,create_graph=False)

        t1=g1.mean(dim=(2,3)).detach().cpu().double().numpy()
        t2=g2.mean(dim=(2,3)).detach().cpu().double().numpy()
        CT1 += t1.T @ t1
        CT2 += t2.T @ t2
        nT += t1.shape[0]

        for kind in perturbations:
            xp,xm = A.perturb_pair(x,kind,eps=0.15)
            with torch.no_grad():
                _,p1,p2 = model.raw_early_features(xp)
                _,m1,m2 = model.raw_early_features(xm)
                d1=(p1.mean(dim=(2,3))-m1.mean(dim=(2,3)))/(2*0.15)
                d2=(p2.mean(dim=(2,3))-m2.mean(dim=(2,3)))/(2*0.15)
            d1=d1.cpu().double().numpy()
            d2=d2.cpu().double().numpy()
            CA1 += d1.T @ d1
            CA2 += d2.T @ d2
            nA += d1.shape[0]

    CA1 /= max(nA,1); CA2 /= max(nA,1)
    CT1 /= max(nT,1); CT2 /= max(nT,1)
    return CA1,CT1,CA2,CT2

@torch.no_grad()
def validation_outputs(model, val_rows, class_weights, device, apply_medain):
    ds=A.RigaDataset(val_rows,augment=False,seed=SEED)
    loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0)
    acc=A.MetricAccumulator()
    loss_sum=ce_sum=dl_sum=0.0
    n=0
    preds=[]
    uids=[]
    model.eval()
    for x,y,uid in loader:
        x=x.to(device); y=y.to(device)
        logits=model(x,apply_medain=apply_medain)
        loss,ce,dl=A.loss_fn(logits,y,class_weights)
        bs=x.shape[0]
        loss_sum += float(loss.item())*bs
        ce_sum += float(ce.item())*bs
        dl_sum += float(dl.item())*bs
        n += bs
        acc.update(logits,y)
        preds.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
        uids.extend(list(uid))
    m=acc.compute()
    m.update({
        "loss":loss_sum/n,
        "ce_loss":ce_sum/n,
        "dice_loss":dl_sum/n,
        "n_samples":n,
    })
    return m, np.concatenate(preds,axis=0), uids

def set_variant(model, U1, U2):
    model.med1.set_basis(torch.from_numpy(U1))
    model.med2.set_basis(torch.from_numpy(U2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target-domain",required=True,choices=DOMAINS)
    args=ap.parse_args()
    target=args.target_domain

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    A.seed_everything(SEED)
    device=torch.device("cpu")

    rows=A.load_rows(target)
    class_weights=A.compute_train_weights(rows["train"]).to(device)

    run_name=f"stage9a_anchor_erm_{target}_seed1103_r2_a0p5"
    ckpt=STAGE9/"anchors/checkpoints"/run_name/"best_model.pt"
    if not ckpt.exists():
        raise RuntimeError(f"Missing anchor checkpoint: {ckpt}")

    state=torch.load(ckpt,map_location=device,weights_only=False)

    model=A.ResNet18UNet(method="medain",rank=RANK,alpha=ALPHA).to(device)
    missing,unexpected=model.load_state_dict(state["model_state"],strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint state mismatch missing={missing} unexpected={unexpected}")

    chosen=select_basis_rows(rows["train"],target)
    chosen_uids=[r["sample_uid"] for r in chosen]

    CA1,CT1,CA2,CT2=estimate_covariances(model,chosen,class_weights,device)

    Utp1,evtp1=solve_tp(CA1,CT1)
    Utp2,evtp2=solve_tp(CA2,CT2)
    Uao1,evao1=solve_ao(CA1)
    Uao2,evao2=solve_ao(CA2)
    Ur1=solve_rand(64,target,"stage1")
    Ur2=solve_rand(128,target,"stage2")

    bases={
        "TP":(Utp1,Utp2),
        "AO":(Uao1,Uao2),
        "RAND":(Ur1,Ur2),
    }

    mechanism={
        "TP":{
            "stage1":basis_metrics(Utp1,CA1,CT1),
            "stage2":basis_metrics(Utp2,CA2,CT2),
            "generalized_eigenvalues_stage1":evtp1.tolist(),
            "generalized_eigenvalues_stage2":evtp2.tolist(),
        },
        "AO":{
            "stage1":basis_metrics(Uao1,CA1,CT1),
            "stage2":basis_metrics(Uao2,CA2,CT2),
            "acquisition_eigenvalues_stage1":evao1.tolist(),
            "acquisition_eigenvalues_stage2":evao2.tolist(),
        },
        "RAND":{
            "stage1":basis_metrics(Ur1,CA1,CT1),
            "stage2":basis_metrics(Ur2,CA2,CT2),
        }
    }

    ang1=np.degrees(scipy.linalg.subspace_angles(Utp1.astype(float),Uao1.astype(float))).tolist()
    ang2=np.degrees(scipy.linalg.subspace_angles(Utp2.astype(float),Uao2.astype(float))).tolist()

    # OFF reference
    off_metrics,off_pred,off_uids=validation_outputs(model,rows["val"],class_weights,device,apply_medain=False)

    val={"OFF":off_metrics}
    agreements={"OFF":1.0}
    for name,(u1,u2) in bases.items():
        set_variant(model,u1,u2)
        m,p,uids=validation_outputs(model,rows["val"],class_weights,device,apply_medain=True)
        if uids != off_uids:
            raise RuntimeError("Validation UID order changed")
        val[name]=m
        agreements[name]=float(np.mean(p==off_pred))

    for name in ("TP","AO","RAND"):
        val[name]["delta_mean_fg_dice_vs_OFF"]=float(val[name]["mean_fg_dice"]-val["OFF"]["mean_fg_dice"])
        val[name]["delta_loss_vs_OFF"]=float(val[name]["loss"]-val["OFF"]["loss"])
        val[name]["pixel_agreement_vs_OFF"]=agreements[name]

    out_dir=STAGE9/"stage9a_pilot"/target
    out_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        out_dir/"mechanism_matrices_and_bases.npz",
        CA1=CA1,CT1=CT1,CA2=CA2,CT2=CT2,
        U_TP_stage1=Utp1,U_TP_stage2=Utp2,
        U_AO_stage1=Uao1,U_AO_stage2=Uao2,
        U_RAND_stage1=Ur1,U_RAND_stage2=Ur2,
        selected_sample_uids=np.asarray(chosen_uids,dtype="U512"),
    )

    result={
        "stage":"MedAIN_Stage9A_Prospective_Pilot",
        "status":"PASS",
        "target_domain":target,
        "anchor_seed":SEED,
        "anchor_best_epoch":int(state["epoch"]),
        "anchor_best_val_mean_fg_dice":float(state["best_val_mean_fg_dice"]),
        "basis_selection_rule":"SHA256 row-order-invariant locked rule",
        "basis_sample_uids":chosen_uids,
        "basis_sample_uid_digest":hashlib.sha256(("\\n".join(chosen_uids)+"\\n").encode()).hexdigest(),
        "mechanism":mechanism,
        "principal_angles_deg_TP_vs_AO":{"stage1":ang1,"stage2":ang2},
        "validation":val,
        "target_test_evaluated":False,
        "external_data_used":False,
        "protocol_lock_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "anchor_checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    }
    (out_dir/"result.json").write_text(json.dumps(result,indent=2)+"\\n",encoding="utf-8")
    (out_dir/"PASS").write_text("PASS\\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
