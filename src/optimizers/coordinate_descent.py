import torch
import time
from src.channel.system_model import SystemModel
from src.evaluation.canonical_evaluator import CanonicalEvaluator

class CoordinateDescent:
    """
    Coordinate Descent (CD) optimizer baseline for RIS Phase Shifts.
    Optimizes one element theta_n at a time while preserving non-decreasing rate.
    """

    def __init__(self, system_model: SystemModel, max_passes: int = 2):
        self.sys = system_model
        self.evaluator = CanonicalEvaluator(system_model)
        self.passes = max_passes

    def solve_batch(
        self,
        H_BR: torch.Tensor,
        H_RU: torch.Tensor,
        H_BU: torch.Tensor,
        discrete: bool = True
    ) -> tuple:
        device = H_BR.device
        B, N, M = H_BR.shape
        start_time = time.perf_counter()

        theta = torch.rand(B, N, device=device) * 2 * torch.pi
        if discrete:
            theta = self.sys.quantize_phase(theta)

        alphabet = self.sys.phase_alphabet_torch if discrete and self.sys.phase_alphabet_torch is not None else torch.linspace(0, 2 * torch.pi, 8, device=device)

        # Evaluate current rate
        res = self.evaluator.evaluate(H_BR, H_RU, H_BU, theta)
        current_rates = res["sum_rate"] # [B]

        for p in range(self.passes):
            for n in range(N):
                best_rates = current_rates.clone()
                best_states = theta[:, n].clone()

                for state in alphabet:
                    theta_test = theta.clone()
                    theta_test[:, n] = state

                    res_test = self.evaluator.evaluate(H_BR, H_RU, H_BU, theta_test)
                    rate_test = res_test["sum_rate"]

                    mask = rate_test > best_rates
                    best_rates[mask] = rate_test[mask]
                    best_states[mask] = state

                theta[:, n] = best_states
                current_rates = best_rates

        final_res = self.evaluator.evaluate(H_BR, H_RU, H_BU, theta)
        exec_time = time.perf_counter() - start_time
        return theta, final_res["sum_rate"].cpu().numpy(), exec_time
