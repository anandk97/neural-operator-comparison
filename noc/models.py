"""Model registry: 5 families x (base, newer variant). PINN and PirateNet live in pinn.py (they solve one instance
at a time rather than learning an operator)."""

from .deeponet import DeepONetFamily
from .spectral import FNOWrap, KANOWrap
from .transolver import Transolver

FAMILIES = {
    "DeepONet": ("deeponet", "shift_deeponet"),
    "Fourier NO": ("fno", "local_fno"),
    "KAN": ("deepokan", "kano"),
    "Transformer": ("transolver", "transolver_pp"),
    "PINN": ("pinn", "piratenet"),
}

GRID_ONLY = {"fno", "local_fno", "kano"}


def build(name, task):
    if name in GRID_ONLY and task.kind == "points":
        raise ValueError(f"{name} needs a grid or structured mesh; {task.name} is a point cloud")
    if name == "deeponet":
        return DeepONetFamily(task, "deeponet")
    if name == "shift_deeponet":
        return DeepONetFamily(task, "shift")
    if name == "shift_deeponet_shared":  # ablation: Lanthaler et al.'s shared-trunk form
        return DeepONetFamily(task, "shift_shared")
    if name == "deepokan":
        return DeepONetFamily(task, "deepokan")
    if name == "fno":
        return FNOWrap(task)
    if name == "local_fno":
        return FNOWrap(task, local=True)
    if name == "kano":
        return KANOWrap(task)
    if name == "transolver":
        return Transolver(task)
    if name == "transolver_pp":
        return Transolver(task, plus=True)
    raise KeyError(name)
