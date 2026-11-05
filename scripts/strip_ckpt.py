#!/usr/bin/env python
"""Strip dual-only parameters (pert_encoder_inv.*) from a DualState checkpoint so the stock
StateTransitionPerturbationModel can load it for predict/eval.
Usage: python scripts/strip_ckpt.py <in.ckpt> <out.ckpt>"""
import sys
import torch

src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
drop = [k for k in sd if "pert_encoder_inv" in k or "dual_classifier" in k or "dual_pert_mask" in k]
for k in drop:
    del sd[k]
print(f"dropped {len(drop)} dual-only params")
torch.save(ckpt, dst)
print("wrote", dst)
