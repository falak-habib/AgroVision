#!/usr/bin/env python3
"""
AgroVision - leaf disease classifier training (PyTorch).

Pipeline (follows the research brief):
  1. Scan PlantVillage-style folder, keep 1-2 crops, canonical labels "Crop__Disease"
  2. Group-aware train/val/test split (no near-duplicate leakage)
  3. Transfer learning: head warm-up -> full fine-tune (MobileNetV2 or EfficientNet-B0)
  4. Evaluate: macro F1, per-class recall, confusion matrix
  5. Calibrate (temperature scaling on VAL only) + pick abstain threshold on VAL only
  6. Optional external test (e.g. mapped PlantDoc subset) -> shows domain-shift gap
  7. Save versioned artifact: weights + class map + preprocessing + T + threshold

Example:
  python train_agrovision.py --data_dir PlantVillage --crops Tomato Potato \
      --arch mobilenet_v2 --out runs/mnv2_v1 \
      --external_dir PlantDoc_test --class_map plantdoc_map.csv
"""
import argparse, csv, json, random, re, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# Strip augmentation suffixes so copies of one leaf share a group id. CHECK this
# against your dataset copy (Kaggle "augmented" versions use different names).
AUG_SUFFIX = re.compile(
    r"(_flip(LR|TB)|_rotate_?\d+|_\d+deg|_scale|_crop|_contrast|_brightness|"
    r"_shear|_zoom|_final_masked|_copy\d*)+$", re.I)


# ----------------------------------------------------------------- data utils
def canon(name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato__Early_blight'."""
    return re.sub(r"_{2,}", "__", name.strip().replace(" ", "_"))


def load_class_map(path):
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["alias"].strip(): r["canonical"].strip() for r in csv.DictReader(f)}


def scan(root, crops=None, class_map=None):
    """Return list of (path, canonical_label, group_id)."""
    class_map = class_map or {}
    items = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        if d.name in class_map:
            label = class_map[d.name]
        else:
            label = canon(d.name)
            for c in (crops or []):  # 'Tomato_Early_blight' / 'Tomato___Early_blight' -> 'Tomato__Early_blight'
                if d.name.lower().startswith(c.lower()):
                    label = f"{c}__{canon(d.name[len(c):].lstrip('_ '))}"
                    break
        if crops and not any(label.lower().startswith(c.lower() + "__") for c in crops):
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                items.append((str(p), label, f"{label}/{AUG_SUFFIX.sub('', p.stem)}"))
    if not items:
        raise SystemExit(f"No images found in {root} for crops={crops}")
    return items


class LeafDS(Dataset):
    def __init__(self, paths, labels, tfm):
        self.paths, self.labels, self.tfm = paths, labels, tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), self.labels[i]


def make_tfms(size):
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.02),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    evalt = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train, evalt


# ---------------------------------------------------------------------- model
def build_model(arch, n_classes):
    if arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise ValueError(arch)
    return m


def set_backbone_trainable(m, flag):
    for p in m.features.parameters():
        p.requires_grad = flag


@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(outs), torch.cat(ys)


def fit_stage(model, tr_dl, va_dl, device, epochs, lr, class_w, patience, tag, ckpt_path=None):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x), y, weight=class_w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * len(y)
            n += len(y)
        sched.step()
        lg, yy = get_logits(model, va_dl, device)
        f1 = f1_score(yy, lg.argmax(1), average="macro")
        print(f"[{tag}] ep{ep:02d} loss={loss_sum/n:.4f} val_macroF1={f1:.4f} ({time.time()-t0:.0f}s)")
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path:  # saved at every improvement, so a disconnect does not lose everything
                torch.save(best_state, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[{tag}] early stop")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1


# ------------------------------------------------- calibration / abstention
def fit_temperature(logits, labels):
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.05), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=0.05))


def ece(probs, labels, bins=15):
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    edges, e = torch.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(e)


def pick_threshold(probs, labels, target_acc, min_cov=0.2):
    """Smallest confidence threshold whose accepted-set accuracy >= target (VAL only)."""
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    for t in np.linspace(0.30, 0.99, 70):
        m = conf >= t
        if m.float().mean() >= min_cov and correct[m].mean() >= target_acc:
            return float(t)
    return 0.99


def evaluate(name, logits, labels, class_names, T, thr, out):
    probs = F.softmax(logits / T, dim=1)
    conf, pred = probs.max(1)
    present = sorted(set(labels.tolist()))
    names = [class_names[i] for i in present]
    rep = classification_report(labels, pred, labels=present, target_names=names,
                                output_dict=True, zero_division=0)
    accept = conf >= thr
    res = {
        "n": int(len(labels)),
        "accuracy": float((pred == labels).float().mean()),
        "macro_f1": float(f1_score(labels, pred, labels=present, average="macro")),
        "ece": ece(probs, labels),
        "abstain_threshold": thr,
        "coverage": float(accept.float().mean()),
        "accepted_accuracy": float((pred[accept] == labels[accept]).float().mean()) if accept.any() else None,
        "per_class": {k: {"precision": v["precision"], "recall": v["recall"],
                          "f1": v["f1-score"], "support": v["support"]} for k, v in rep.items()
                      if k in names},
    }
    cm = confusion_matrix(labels, pred, labels=present)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8),) * 2)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} (row-normalised, counts shown)")
    fig.tight_layout()
    fig.savefig(out / f"confusion_{name}.png", dpi=150)
    plt.close(fig)
    print(f"\n== {name} == n={res['n']} acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f} "
          f"ECE={res['ece']:.4f} coverage@thr={res['coverage']:.2f} "
          f"accepted_acc={res['accepted_accuracy']}")
    for k, v in res["per_class"].items():
        print(f"   {k:45s} recall={v['recall']:.3f} f1={v['f1']:.3f} n={int(v['support'])}")
    return res


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder of class sub-folders (PlantVillage)")
    ap.add_argument("--crops", nargs="*", default=["Tomato", "Potato"],
                    help="crop name prefixes to keep, or 'all' for every class folder")
    ap.add_argument("--arch", default="mobilenet_v2", choices=["mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--target_acc", type=float, default=0.95,
                    help="accuracy wanted on accepted (non-abstained) val samples")
    ap.add_argument("--external_dir", default=None, help="field-style test set (e.g. PlantDoc)")
    ap.add_argument("--class_map", default=None, help="CSV with columns alias,canonical for external set")
    ap.add_argument("--limit_per_class", type=int, default=0,
                    help="smoke-test only: use at most N images per class (0 = use all)")
    ap.add_argument("--init_weights", default=None,
                    help="resume: path to checkpoint_best.pt from an interrupted run (skips warm-up)")
    ap.add_argument("--out", default="runs/agrovision_v1")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.crops and [c.lower() for c in a.crops] == ["all"]:
        a.crops = []  # empty = no crop filter, use every class folder found

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("device:", device)

    # ---- data + group-aware split
    items = scan(a.data_dir, a.crops)
    if a.limit_per_class:
        seen, kept = {}, []
        for it in items:
            seen[it[1]] = seen.get(it[1], 0) + 1
            if seen[it[1]] <= a.limit_per_class:
                kept.append(it)
        items = kept
        print(f"SMOKE TEST: limited to {a.limit_per_class} images/class (results are not meaningful)")
    classes = sorted({l for _, l, _ in items})
    cidx = {c: i for i, c in enumerate(classes)}
    paths = np.array([p for p, _, _ in items])
    y = np.array([cidx[l] for _, l, _ in items])
    groups = np.array([g for _, _, g in items])
    print(f"{len(items)} images, {len(classes)} classes")
    for c in classes:
        print(f"   {c:45s} {int((y == cidx[c]).sum())}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=a.seed)
    tr_i, rest_i = next(gss.split(paths, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=a.seed)
    va_r, te_r = next(gss2.split(paths[rest_i], y[rest_i], groups[rest_i]))
    va_i, te_i = rest_i[va_r], rest_i[te_r]
    with open(out / "splits.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "label", "split"])
        for idx, s in ((tr_i, "train"), (va_i, "val"), (te_i, "test")):
            for i in idx:
                w.writerow([paths[i], classes[y[i]], s])

    tr_t, ev_t = make_tfms(a.img_size)
    mk = lambda idx, t, sh: DataLoader(LeafDS(paths[idx], y[idx], t), batch_size=a.batch_size,
                                       shuffle=sh, num_workers=a.workers, pin_memory=device.type == "cuda")
    tr_dl, va_dl, te_dl = mk(tr_i, tr_t, True), mk(va_i, ev_t, False), mk(te_i, ev_t, False)

    counts = np.bincount(y[tr_i], minlength=len(classes)).astype(np.float32)
    class_w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), device=device)

    # ---- train: warm-up head, then fine-tune everything
    model = build_model(a.arch, len(classes)).to(device)
    ckpt = out / "checkpoint_best.pt"
    warmup_epochs = a.warmup_epochs
    if a.init_weights:
        model.load_state_dict(torch.load(a.init_weights, map_location=device))
        warmup_epochs = 0
        print(f"resumed from {a.init_weights}; skipping warm-up")
    if warmup_epochs > 0:
        set_backbone_trainable(model, False)
        fit_stage(model, tr_dl, va_dl, device, warmup_epochs, a.lr_head, class_w, a.patience, "warmup", ckpt)
    set_backbone_trainable(model, True)
    best = fit_stage(model, tr_dl, va_dl, device, a.epochs, a.lr_ft, class_w, a.patience, "finetune", ckpt)
    print(f"best val macro-F1: {best:.4f}")

    # ---- calibrate + threshold on VAL only
    va_lg, va_y = get_logits(model, va_dl, device)
    T = fit_temperature(va_lg, va_y)
    thr = pick_threshold(F.softmax(va_lg / T, 1), va_y, a.target_acc)
    print(f"temperature={T:.3f}  abstain threshold={thr:.2f}")

    # ---- final reports
    metrics = {"val": evaluate("val", va_lg, va_y, classes, T, thr, out)}
    te_lg, te_y = get_logits(model, te_dl, device)
    metrics["test_in_domain"] = evaluate("test_in_domain", te_lg, te_y, classes, T, thr, out)

    if a.external_dir:
        ext = scan(a.external_dir, a.crops, load_class_map(a.class_map))
        keep = [(p, l) for p, l, _ in ext if l in cidx]
        dropped = sorted({l for _, l, _ in ext if l not in cidx})
        print(f"external: {len(keep)} usable images; classes not in model (excluded): {dropped}")
        if keep:
            ext_dl = DataLoader(LeafDS([p for p, _ in keep], np.array([cidx[l] for _, l in keep]), ev_t),
                                batch_size=a.batch_size, num_workers=a.workers)
            ex_lg, ex_y = get_logits(model, ext_dl, device)
            metrics["external_field"] = evaluate("external_field", ex_lg, ex_y, classes, T, thr, out)
            metrics["external_field"]["excluded_classes"] = dropped

    # ---- versioned artifact (everything inference needs)
    torch.save(model.state_dict(), out / "model.pt")
    artifact = {
        "model_version": f"agrovision-{a.arch}-0.1.0", "arch": a.arch,
        "classes": classes, "input_size": a.img_size,
        "normalize_mean": MEAN, "normalize_std": STD,
        "resize": "direct_resize_to_square", "temperature": T,
        "abstain_threshold": thr, "target_accepted_acc_on_val": a.target_acc,
        "train_args": vars(a), "torch": torch.__version__,
    }
    (out / "artifact.json").write_text(json.dumps(artifact, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved to {out}/ : model.pt, artifact.json, metrics.json, splits.csv, confusion_*.png")


if __name__ == "__main__":
    main()
