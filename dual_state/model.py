"""DualState: State Transition model with a dual-learning objective.

Primal task  f: (control cell set, perturbation p)     -> perturbed cell set   (original ST)
Dual task    g: (perturbed cell set, inverse-pert p̄)   -> control cell set     (new)

Loss: L = L_primal + lambda_dual * L_dual + lambda_cycle * L_cycle
  L_primal = MMD(f(x_ctrl, p),  x_pert)              (original energy-MMD)
  L_dual   = MMD(g(x_pert, p̄),  x_ctrl)              (inverse mapping, energy-MMD)
  L_cycle  = MMD(f(g(x_pert, p̄), p), x_pert)
           + MMD(g(f(x_ctrl, p), p̄), x_ctrl)         (reconstruction duality, optional)

The inverse perturbation embedding uses a separate encoder (pert_encoder_inv)
initialized as a deepcopy of the primal perturbation encoder, so the two
directions start from the same point and specialize during fine-tuning —
mirroring dual supervised learning's joint training of f and g.

DualCrossStateTransitionPerturbationModel: aligns the DualCross1.0 variant
(github.com/Linezolid2011/DualCross1.0) with the State ST model — the dual
task is *classifying the perturbation* from the hidden-state difference
(transformer hidden − basal-encoded control), pooled over the cell set,
cross-entropy. No inverse generative mapping, no cycle loss.
"""
import copy
import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from state.tx.models.state_transition import StateTransitionPerturbationModel

logger = logging.getLogger(__name__)


class DualStateTransitionPerturbationModel(StateTransitionPerturbationModel):
    def __init__(self, *args, **kwargs):
        self.lambda_dual = kwargs.get("lambda_dual", 1.0)
        self.lambda_cycle = kwargs.get("lambda_cycle", 0.0)
        # "wrong-drug" control: shuffle perturbation labels within the batch for
        # the dual branch, so g is asked to map x_pert(A) back to control while
        # conditioned on a random wrong drug. Ablates whether the dual gain is
        # drug-specific signal or mere regularization.
        self.dual_shuffle_labels = bool(kwargs.get("dual_shuffle_labels", False))
        super().__init__(*args, **kwargs)
        # inverse perturbation encoder: same architecture as the primal one
        self.pert_encoder_inv = copy.deepcopy(self.pert_encoder)
        self._dual_mode = False

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        if self._dual_mode:
            return self.pert_encoder_inv(pert)
        return self.pert_encoder(pert)

    def _dual_forward(self, batch: dict, basal_key: str, padded: bool = True) -> torch.Tensor:
        """g: basal cells (key=basal_key, typically the perturbed cells) + inv-pert emb -> control cells."""
        dual_batch = dict(batch)
        dual_batch["ctrl_cell_emb"] = batch[basal_key]
        if self.dual_shuffle_labels:
            # shuffle pert labels across cell sets in the batch (per-set shuffle)
            pe = batch["pert_emb"]
            if padded:
                B = pe.shape[0] // self.cell_sentence_len
                pe3 = pe.reshape(B, self.cell_sentence_len, -1)
                perm = torch.randperm(B, device=pe.device)
                dual_batch["pert_emb"] = pe3[perm].reshape(pe.shape)
            # non-padded path has a single set; shuffling is a no-op there
        self._dual_mode = True
        try:
            out = self.forward(dual_batch, padded=padded)
        finally:
            self._dual_mode = False
        return out

    def _reshape(self, x, padded):
        if padded:
            return x.reshape(-1, self.cell_sentence_len, self.output_dim)
        return x.reshape(1, -1, self.output_dim)

    def _dual_losses(self, batch, padded: bool) -> torch.Tensor:
        # L_dual: g(x_pert, p̄) vs x_ctrl
        pred_ctrl = self._reshape(self._dual_forward(batch, "pert_cell_emb", padded), padded)
        target_ctrl = self._reshape(batch["ctrl_cell_emb"], padded)
        dual_loss = torch.nanmean(self._compute_distribution_loss(pred_ctrl, target_ctrl))
        self.log("train/dual_loss", dual_loss)
        total = self.lambda_dual * dual_loss

        if self.lambda_cycle > 0.0:
            # cycle 1: f(g(x_pert, p̄), p) vs x_pert   (chain keeps gradients)
            cyc_batch = dict(batch)
            cyc_batch["ctrl_cell_emb"] = pred_ctrl.reshape(-1, self.output_dim) if padded else pred_ctrl
            pred_pert_cyc = self._reshape(self.forward(cyc_batch, padded=padded), padded)
            target_pert = self._reshape(batch["pert_cell_emb"], padded)
            cyc1 = torch.nanmean(self._compute_distribution_loss(pred_pert_cyc, target_pert))

            # cycle 2: g(f(x_ctrl, p), p̄) vs x_ctrl
            pred_pert = self.forward(batch, padded=padded)
            cyc2_pred_ctrl = self._reshape(
                self._dual_forward(dict(batch, pert_cell_emb=pred_pert), "pert_cell_emb", padded), padded
            )
            cyc2 = torch.nanmean(self._compute_distribution_loss(cyc2_pred_ctrl, target_ctrl))

            cycle_loss = 0.5 * (cyc1 + cyc2)
            self.log("train/cycle_loss", cycle_loss)
            total = total + self.lambda_cycle * cycle_loss

        return total

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        base_loss = super().training_step(batch, batch_idx, padded=padded)
        return base_loss + self._dual_losses(batch, padded)


class DualOutClsStateTransitionPerturbationModel(StateTransitionPerturbationModel):
    """Output-space dual classifier: classify the perturbation from the
    *predicted expression delta* (pred − control), mean-pooled over the cell set.

    DualCross reads the transformer hidden-state difference, which is an
    open-book exam: the perturbation embedding is added to the input tokens
    (state_transition.py: combined_input = pert_embedding + control_cells),
    so the hidden states carry the answer for free and the CE loss saturates
    at ~0 without constraining generative quality. Here the classifier only
    sees the decoded output delta, so it can only succeed if the model's
    *predictions* genuinely encode the perturbation's effect — the true
    dual constraint of DSL. Watermarking the answer into the output is
    suppressed by the primal MMD loss.
    Assumes input_dim == output_dim (hvg gene space).
    """

    def __init__(self, *args, **kwargs):
        self.dual_cls_weight = kwargs.get("dual_cls_weight", 0.1)
        self.var_anchor_weight = kwargs.get("var_anchor_weight", 0.0)
        super().__init__(*args, **kwargs)
        G = self.output_dim
        self.dual_classifier = nn.Sequential(
            nn.Linear(G, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.pert_dim),
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        total = super().training_step(batch, batch_idx, padded=padded)

        # one shared forward for the dual classifier and the variance anchor
        out = self.forward(batch, padded=padded)
        pred = out[0] if isinstance(out, tuple) else out
        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            ctrl = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.output_dim)
            pert_oh = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            ctrl = batch["ctrl_cell_emb"].reshape(1, -1, self.output_dim)
            pert_oh = batch["pert_emb"].reshape(1, -1, self.pert_dim)

        delta = (pred - ctrl).mean(dim=1)  # [B, G] predicted effect
        dual_logits = self.dual_classifier(delta)
        pert_target = pert_oh[:, 0, :].argmax(dim=-1)
        dual_loss = F.cross_entropy(dual_logits, pert_target)
        self.log("train/dual_cls_loss", dual_loss, prog_bar=True)
        total = total + self.dual_cls_weight * dual_loss

        if self.var_anchor_weight > 0.0:
            target_v = batch["pert_cell_emb"]
            if padded:
                target_v = target_v.reshape(-1, self.cell_sentence_len, self.output_dim)
            else:
                target_v = target_v.reshape(1, -1, self.output_dim)
            eps = 1e-4
            v_pred = pred.var(dim=1) + eps    # [B, G] cross-cell variance per sentence
            v_real = target_v.var(dim=1) + eps
            var_loss = (v_pred.log() - v_real.log()).abs().mean()
            self.log("train/var_anchor_loss", var_loss, prog_bar=True)
            total = total + self.var_anchor_weight * var_loss
        return total


class DualOutCls2StateTransitionPerturbationModel(DualOutClsStateTransitionPerturbationModel):
    """Hardened output-delta dual ("加固"): bottleneck head without LayerNorm.

    The clsout head (512-wide + LayerNorm) still saturates within ~1k steps:
    LayerNorm erases amplitude information, so the classifier can exploit a
    weak, high-dimensional watermark in the output delta. The hardened head
    uses a narrow 64-d bottleneck and no LayerNorm — the dual signal must
    travel through the dominant low-dimensional structure at amplitudes the
    primal MMD loss is willing to pay for, which is exactly where the real
    perturbation biology lives.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        G = self.output_dim
        self.dual_classifier = nn.Sequential(
            nn.Linear(G, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.pert_dim),
        )


class DualCrossStateTransitionPerturbationModel(StateTransitionPerturbationModel):
    """State ST + DualCross1.0's hidden-diff dual classifier, aligned onto State.

    Dual task: classify which perturbation was applied from the hidden-state
    difference (transformer hidden − basal-encoded control), mean-pooled over
    the cell set. Loss: L = L_primal(official) + dual_cls_weight * CE.
    The classifier is unused at inference (strip it for eval).
    """

    def __init__(self, *args, **kwargs):
        self.dual_cls_weight = kwargs.get("dual_cls_weight", 0.1)
        self.var_anchor_weight = kwargs.get("var_anchor_weight", 0.0)
        super().__init__(*args, **kwargs)
        h = self.hidden_dim
        self.dual_classifier = nn.Sequential(
            nn.Linear(h, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.pert_dim),
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        total = super().training_step(batch, batch_idx, padded=padded)

        # hidden-state difference: transformer output (cached by forward) − basal encoding
        if padded:
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
            pert_oh = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
        else:
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)
            pert_oh = batch["pert_emb"].reshape(1, -1, self.pert_dim)
        control_cells = self.encode_basal_expression(basal)  # [B, S, H]
        hidden_diff = self._token_features - control_cells   # [B, S, H]
        dual_logits = self.dual_classifier(hidden_diff.mean(dim=1))  # [B, n_pert]

        pert_target = pert_oh[:, 0, :].argmax(dim=-1)
        dual_loss = F.cross_entropy(dual_logits, pert_target)
        self.log("train/dual_cls_loss", dual_loss, prog_bar=True)
        total = total + self.dual_cls_weight * dual_loss

        if self.var_anchor_weight > 0.0:
            # variance anchor: keep cross-cell variance realistic (counter over-smoothing
            # from the dual objective, which inflates DE test power -> nsig overprediction)
            out = self.forward(batch, padded=padded)
            pred_v = out[0] if isinstance(out, tuple) else out
            target_v = batch["pert_cell_emb"]
            if padded:
                pred_v = pred_v.reshape(-1, self.cell_sentence_len, self.output_dim)
                target_v = target_v.reshape(-1, self.cell_sentence_len, self.output_dim)
            else:
                pred_v = pred_v.reshape(1, -1, self.output_dim)
                target_v = target_v.reshape(1, -1, self.output_dim)
            eps = 1e-4
            v_pred = pred_v.var(dim=1) + eps    # [B, G] cross-cell variance per sentence
            v_real = target_v.var(dim=1) + eps
            var_loss = (v_pred.log() - v_real.log()).abs().mean()
            self.log("train/var_anchor_loss", var_loss, prog_bar=True)
            total = total + self.var_anchor_weight * var_loss
        return total


class DualInvSplitStateTransitionPerturbationModel(DualStateTransitionPerturbationModel):
    """Inverse-mapping generative dual with a perturbation-level split.

    A fixed seeded subset of perturbations (fraction `dual_pert_frac`) is
    *dual-only*: their batches train only the inverse map g(x_pert, p̄)->x_ctrl.
    The rest are primal-only. The two tasks therefore never share a perturbation
    at the loss level — g must invert perturbations that f was never fit on
    forward, forcing the perturbation embedding to carry invertible structure
    instead of per-pert memorization (the DSL low-resource analogy: each
    direction has its own exclusive data).
    """

    def __init__(self, *args, **kwargs):
        self.dual_pert_frac = kwargs.get("dual_pert_frac", 0.2)
        self.dual_split_seed = kwargs.get("dual_split_seed", 42)
        super().__init__(*args, **kwargs)
        g = torch.Generator().manual_seed(self.dual_split_seed)
        perm = torch.randperm(self.pert_dim, generator=g)
        n_dual = max(1, int(round(self.pert_dim * self.dual_pert_frac)))
        mask = torch.zeros(self.pert_dim, dtype=torch.bool)
        mask[perm[:n_dual]] = True
        self.register_buffer("dual_pert_mask", mask, persistent=True)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        batch = self._normalize_count_keys(batch)
        if padded:
            pert_oh = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
        else:
            pert_oh = batch["pert_emb"].reshape(1, -1, self.pert_dim)
        pert_ids = pert_oh[:, 0, :].argmax(dim=-1)  # [B] one pert per cell sentence
        is_dual = self.dual_pert_mask[pert_ids]
        is_primal = ~is_dual

        total = 0.0
        if is_primal.any():
            pred = self.forward(batch, padded=padded)
            if isinstance(pred, tuple):
                pred = pred[0]
            target = batch["pert_cell_emb"]
            pred = self._reshape(pred, padded)
            target = self._reshape(target, padded)
            pl = self._compute_distribution_loss(pred, target)  # per-set [B]
            main_loss = torch.nanmean(pl[is_primal])
            self.log("train_loss", main_loss)
            total = total + main_loss
        if is_dual.any():
            pred_ctrl = self._reshape(self._dual_forward(batch, "pert_cell_emb", padded), padded)
            target_ctrl = self._reshape(batch["ctrl_cell_emb"], padded)
            dl = self._compute_distribution_loss(pred_ctrl, target_ctrl)
            dual_loss = torch.nanmean(dl[is_dual])
            self.log("train/dual_loss", dual_loss)
            total = total + self.lambda_dual * dual_loss
        return total
