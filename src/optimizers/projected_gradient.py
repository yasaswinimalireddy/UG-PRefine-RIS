import torch
import time
from src.channel.system_model import SystemModel
from src.evaluation.canonical_evaluator import CanonicalEvaluator

class ProjectedGradientDescent:
    """
    Projected Gradient Descent (PGD) optimizer baseline for RIS Phase Shifts.
    Uses canonical sum-rate evaluation and Adam gradient ascent steps.
    """

    def __init__(
        self,
        system_model: SystemModel,
        num_steps: int = 50,
        step_size: float = 0.05
    ):
        self.sys = system_model
        self.evaluator = CanonicalEvaluator(system_model)
        self.num_steps = num_steps
        self.eta = step_size

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
        theta = theta.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=self.eta)

        for step in range(self.num_steps):
            optimizer.zero_grad()
            res = self.evaluator.evaluate(H_BR, H_RU, H_BU, theta)
            sum_rate = res["sum_rate"]

            loss = -torch.sum(sum_rate)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if discrete:
                    theta.copy_(self.sys.quantize_phase(theta))
                else:
                    theta.copy_(torch.remainder(theta, 2 * torch.pi))

        with torch.no_grad():
            final_res = self.evaluator.evaluate(H_BR, H_RU, H_BU, theta)

        exec_time = time.perf_counter() - start_time
        return theta.detach(), final_res["sum_rate"].cpu().numpy(), exec_time
