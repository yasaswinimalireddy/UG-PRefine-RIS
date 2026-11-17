import torch
import numpy as np
import time
from src.channel.system_model import SystemModel
from src.evaluation.canonical_evaluator import CanonicalEvaluator

class AlternatingOptimization:
    """
    Vectorized, Mathematically Rigorous Alternating Optimization (AO) for RIS Phase Optimization.
    Evaluates candidate phase updates across batch dimension B simultaneously.
    Guarantees monotonic non-decreasing sum-rate convergence per iteration.
    """

    def __init__(
        self,
        system_model: SystemModel,
        max_iters: int = 3,
        num_starts: int = 5,
        grid_resolution: int = 16
    ):
        self.sys = system_model
        self.evaluator = CanonicalEvaluator(system_model)
        self.max_iters = max_iters
        self.num_starts = num_starts
        self.grid_resolution = grid_resolution

    def solve_batch(
        self,
        H_BR: torch.Tensor,
        H_RU: torch.Tensor,
        H_BU: torch.Tensor,
        discrete: bool = True
    ) -> tuple:
        """
        Solves AO optimization across batch B.
        """
        device = H_BR.device
        B, N, M = H_BR.shape
        start_time = time.perf_counter()

        if discrete and self.sys.phase_alphabet_torch is not None:
            candidates = self.sys.phase_alphabet_torch.to(device)
        else:
            candidates = torch.linspace(0, 2 * np.pi, self.grid_resolution, device=device)[:-1]

        best_global_rates = torch.full((B,), -1.0, device=device)
        best_global_thetas = torch.zeros(B, N, device=device)

        for s in range(self.num_starts):
            # Random initial phase
            theta = torch.rand(B, N, device=device) * 2 * np.pi
            if discrete:
                theta = self.sys.quantize_phase(theta)

            # Evaluate initial rate
            res = self.evaluator.evaluate(H_BR, H_RU, H_BU, theta)
            curr_rates = res["sum_rate"] # [B]

            for it in range(self.max_iters):
                for n in range(N):
                    best_n_rates = curr_rates.clone()
                    best_n_states = theta[:, n].clone()

                    for cand in candidates:
                        theta_test = theta.clone()
                        theta_test[:, n] = cand

                        res_test = self.evaluator.evaluate(H_BR, H_RU, H_BU, theta_test)
                        r_test = res_test["sum_rate"] # [B]

                        mask = r_test > best_n_rates
                        best_n_rates[mask] = r_test[mask]
                        best_n_states[mask] = cand

                    theta[:, n] = best_n_states
                    curr_rates = best_n_rates

            mask_start = curr_rates > best_global_rates
            best_global_rates[mask_start] = curr_rates[mask_start]
            best_global_thetas[mask_start] = theta[mask_start]

        exec_time = time.perf_counter() - start_time
        return best_global_thetas, best_global_rates.cpu().numpy(), exec_time
