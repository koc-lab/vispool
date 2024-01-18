from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from functools import partial

import numpy as np
import torch
from vector_vis_graph import WeightMethod, horizontal_vvg, natural_vvg

from vispool import USE_THREADPOOL


class VVGType(Enum):
    NATURAL = auto()
    HORIZONTAL = auto()


def get_vvg(
    multivariate: torch.Tensor,
    *,
    vvg_type: VVGType = VVGType.NATURAL,
    timeline: np.ndarray | None = None,
    weight_method: WeightMethod = WeightMethod.UNWEIGHTED,
    penetrable_limit: int = 0,
    directed: bool = False,
) -> torch.Tensor:
    device = multivariate.device
    tensor_np = multivariate.to("cpu").detach().numpy()
    vvg_fn = natural_vvg if vvg_type == VVGType.NATURAL else horizontal_vvg
    vvg = vvg_fn(
        tensor_np,
        timeline=timeline,
        weight_method=weight_method,
        penetrable_limit=penetrable_limit,
        directed=directed,
    ).astype(np.float32)
    return torch.from_numpy(vvg).to(device)


def get_vvgs(
    tensor: torch.Tensor,
    *,
    vvg_type: VVGType = VVGType.NATURAL,
    timeline: np.ndarray | None = None,
    weight_method: WeightMethod = WeightMethod.UNWEIGHTED,
    penetrable_limit: int = 0,
    directed: bool = False,
) -> torch.Tensor:
    if USE_THREADPOOL:
        get_vvg_partial = partial(
            get_vvg,
            vvg_type=vvg_type,
            timeline=timeline,
            weight_method=weight_method,
            penetrable_limit=penetrable_limit,
            directed=directed,
        )

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(get_vvg_partial, tensor))
    else:
        results = [get_vvg(matrix) for matrix in tensor]
    return torch.stack(results)
