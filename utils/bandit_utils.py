import numpy as np
import numba as nb

from utils.constants import HOEFFDING


#@nb.njit
def get_ci(
    delta: np.ndarray,
    num_samples: int,
    var_proxy: np.ndarray = None,  # Variance proxy of sub-gaussian r.v. # Can't have default None argument because of numba
    sum_var_proxy: float = None, # Can't have default None argument because of numba
    ci_bound: str = HOEFFDING,
    with_replacement: bool = True,
    pop_size: int = 10**10,
) -> np.ndarray:
    assert (var_proxy is not None and sum_var_proxy is None) or (var_proxy is None and sum_var_proxy is not None), "Should only pass one of var_proxy and sum_var_proxy"

    if with_replacement:
        # see https://arxiv.org/pdf/2006.06856.pdf#page=20
        if ci_bound == HOEFFDING:
            # TODO: Should we also be using sum_var_proxy here?
            assert var_proxy is not None, "Need to pass var_proxy"
            return np.sqrt(-2 * np.log(delta) * var_proxy / num_samples)
    else:
        # see Corollary 1 in https://arxiv.org/pdf/1812.06360.pdf
        if ci_bound == HOEFFDING:
            rho = min(
                1 - (num_samples - 1) / pop_size,
                (1 - num_samples / pop_size) * (1 + 1 / num_samples),
            )
            if sum_var_proxy is not None:
                return np.sqrt(
                    -2 * rho * np.log(delta) * sum_var_proxy / num_samples**2
                )

            assert var_proxy is not None, "Need to pass var_proxy"
            return np.sqrt(-2 * rho * np.log(delta) * var_proxy / num_samples)
