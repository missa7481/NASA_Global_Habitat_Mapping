#!/usr/bin/env python3
"""Vision-language concept module: RemoteCLIP zero-shot habitat concepts on Sentinel-2 chips.

1. Concept scores: cosine(image, "a satellite image of <concept>") with RemoteCLIP ViT-B/32
   (Liu et al. 2024), z-scored per concept across chips.
2. Concept profiles: mean concept z-score among report cells with each vector class, minus
   the mean among other-Culicidae-only report cells.
3. Added value: 5-fold spatial-block CV (10-degree blocks), logistic regression per class on
   AlphaEarth only / concepts only / AlphaEarth + concepts, among report cells with chips.
4. Distillation: ridge from all 105 cell features -> each concept (block CV R^2); concepts with
   R^2 >= 0.5 are mapped globally from features alone (no imagery needed at inference).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicofuse.train import ART, Store  # noqa: E402

V = ART / "vlm"
CONCEPTS = [
    "dense urban residential buildings", "informal settlement with tightly packed small roofs",
    "suburban houses with gardens and swimming pools", "industrial area with warehouses",
    "port with shipping containers", "scrapyard or used tire dump", "cemetery", "highway interchange",
    "airport", "golf course or urban park", "rice paddies", "irrigated cropland", "agricultural fields",
    "wetland marsh", "river with riparian vegetation", "lake or reservoir", "coastal mangroves",
    "dense tropical forest", "temperate deciduous forest", "coniferous forest", "dry savanna grassland",
    "desert with bare soil", "snow and ice",
]
CL = ["Ae_albopictus", "Ae_aegypti", "Culex", "Anopheles"]


def score_chips():
    rows = np.load(V / "chip_store_rows.npy")
    have = [r for r in rows if (V / "chips" / f"{r}.png").exists()]
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32")
    model.load_state_dict(torch.load(hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt"), map_location="cpu"))
    model = model.cuda().eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        t = model.encode_text(tok([f"a satellite image of {c}" for c in CONCEPTS]).cuda())
        t = t / t.norm(dim=-1, keepdim=True)
        sims, keep = [], []
        for i in range(0, len(have), 128):
            ims, ok = [], []
            for r in have[i:i + 128]:
                im = Image.open(V / "chips" / f"{r}.png").convert("RGB")
                if np.asarray(im).std() < 2:        # blank / no-data chip
                    continue
                ims.append(pre(im))
                ok.append(r)
            if not ims:
                continue
            v = model.encode_image(torch.stack(ims).cuda())
            v = v / v.norm(dim=-1, keepdim=True)
            sims.append((v @ t.T).cpu().numpy())
            keep += ok
    S = np.concatenate(sims)
    Z = (S - S.mean(0)) / S.std(0)
    np.save(V / "concept_rows.npy", np.array(keep))
    np.save(V / "concept_z.npy", Z.astype(np.float32))
    return np.array(keep), Z


def main():
    keep, Z = score_chips()
    st = Store()
    d = st.d
    pos = st.pos[keep]
    feats = st.X[pos]
    ae = feats[:, st.cols["AlphaEarth"]]
    blocks = (st.rc[pos, 0] // 200).astype(np.int64) * 100 + st.rc[pos, 1] // 200
    obs_map = {o: i for i, o in enumerate(d["obs_idx"])}
    is_obs = np.array([k in obs_map for k in keep])
    Yc = np.zeros((len(keep), 5))
    Yc[is_obs] = d["Y"][[obs_map[k] for k in keep[is_obs]]]
    out = {"n_chips_scored": int(len(keep)), "n_report_cells": int(is_obs.sum())}

    # 2. profiles
    other_only = is_obs & (Yc[:, :4].sum(1) == 0)
    prof = pd.DataFrame({c: Z[is_obs & (Yc[:, s] > 0)].mean(0) - Z[other_only].mean(0) for s, c in enumerate(CL)}, index=CONCEPTS)
    prof.to_csv(V / "concept_profiles.csv")

    # 3. added value among report cells
    rows = []
    zo, ao, bo, yo = Z[is_obs], ae[is_obs], blocks[is_obs], Yc[is_obs]
    for s, c in enumerate(CL):
        y = (yo[:, s] > 0).astype(int)
        for name, F in (("AlphaEarth", ao), ("VLM concepts", zo), ("AlphaEarth + VLM concepts", np.c_[ao, zo])):
            p = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(F, y, bo):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=4000)).fit(F[tr], y[tr])
                p[te] = m.decision_function(F[te])
            rows.append({"class": c, "features": name, "AUC": roc_auc_score(y, p), "positives": int(y.sum())})
    av = pd.DataFrame(rows)
    av.to_csv(V / "added_value.csv", index=False)

    # 4. distillation from cell features (all scored chips incl. random land cells)
    r2 = []
    for j, c in enumerate(CONCEPTS):
        p = np.zeros(len(Z))
        for tr, te in GroupKFold(5).split(feats, Z[:, j], blocks):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 9))).fit(feats[tr], Z[tr, j])
            p[te] = m.predict(feats[te])
        r2.append({"concept": c, "block_cv_R2": r2_score(Z[:, j], p)})
    r2 = pd.DataFrame(r2).sort_values("block_cv_R2", ascending=False)
    r2.to_csv(V / "distillation_r2.csv", index=False)
    mappable = r2[r2.block_cv_R2 >= 0.5].concept.tolist()
    glob = {}
    for c in mappable[:6]:
        j = CONCEPTS.index(c)
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 9))).fit(feats, Z[:, j])
        glob[c] = np.concatenate([m.predict(st.X[i:i + 500_000]) for i in range(0, len(st.X), 500_000)]).astype(np.float16)
    np.savez_compressed(V / "global_concepts.npz", **{c.replace(" ", "_"): v for c, v in glob.items()})
    out.update({"mappable_concepts": mappable})
    (V / "summary.json").write_text(json.dumps(out, indent=1))
    with pd.option_context("display.width", 200):
        print(prof.round(2))
        print(av.pivot(index="class", columns="features", values="AUC").round(3))
        print(r2.round(3).to_string())


if __name__ == "__main__":
    main()
