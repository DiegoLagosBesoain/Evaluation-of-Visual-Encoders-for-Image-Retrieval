import numpy as np
import skimage.io as io
import skimage.transform as transform
import matplotlib.pyplot as plt
import os
from sklearn.metrics import average_precision_score, precision_recall_curve

DATASET   = 'simple1K'
MODELS    = ['resnet18', 'resnet34', 'DINO', 'CLIP']
data_dir  = DATASET
image_dir = os.path.join(data_dir, 'images')
val_file  = os.path.join(data_dir, 'list_of_images.txt')

with open(val_file, "r") as f:
    files = [line.strip().split('\t') for line in f]
labels = [lab for (_, lab) in files]
n_imgs = len(files)

def evaluate_model(model_name):
    feat_file = os.path.join('data', f'feat_{model_name}_{DATASET}.npy')
    feats     = np.load(feat_file)
    feats_n   = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    sim       = feats_n @ feats_n.T
    sim_idx   = np.argsort(-sim, axis=1)

    APs = []
    all_true, all_scores = [], []
    for q in range(n_imgs):
        row    = sim_idx[q]
        ranked = row[row != q]
        y_true  = np.array([1 if labels[i] == labels[q] else 0 for i in ranked])
        scores  = sim[q, ranked]
        AP      = average_precision_score(y_true, scores)
        APs.append(AP)
        all_true .append(y_true)
        all_scores.append(scores)

    APs = np.array(APs)
    mAP = APs.mean()
    print(f"{model_name}: mAP = {mAP:.4f}")

    y_true_glob  = np.concatenate(all_true)
    scores_glob  = np.concatenate(all_scores)
    prec, rec, _ = precision_recall_curve(y_true_glob, scores_glob)

    plt.figure(figsize=(6,4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall ({model_name})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ranking = np.argsort(-APs)
    return sim_idx, ranking, APs

results = {}
for m in MODELS:
    sim_idx, ranking, APs = evaluate_model(m)
    results[m] = (sim_idx, ranking, APs)

def show_queries(sim_idx, ranking, APs, model_name, N=5):
    for kind, qs in [("Top", ranking[:N]), ("Bottom", ranking[-N:])]:
        fig, axes = plt.subplots(N, 6, figsize=(6*1.5, N*1.5))
        for row, q in enumerate(qs):
            # índices de la query + top-5
            imgs = sim_idx[q, :6]
            for col, idx in enumerate(imgs):
                ax = axes[row, col]
                im = io.imread(os.path.join(image_dir, files[idx][0]))
                # redimensiona manteniendo proporción
                h, w = im.shape[:2]
                size = 64
                im2 = transform.resize(im, (int(size*h/w), size)) if h>w else transform.resize(im, (size, int(size*w/h)))
                ax.imshow(im2)
                ax.axis('off')
                if col == 0:
                    ax.set_title(f"Q={q}\nAP={APs[q]:.2f}", fontsize=8)
                    ax.patch.set(lw=3, ec='b')
        plt.suptitle(f"{model_name} {kind}-5")
        plt.tight_layout()
        plt.show()


for m in MODELS:
    sim_idx, ranking, APs = results[m]
    show_queries(sim_idx, ranking, APs, m, N=5)
