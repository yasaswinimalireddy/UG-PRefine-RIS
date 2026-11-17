import torch
import numpy as np
import time

def calculate_metrics(
    model,
    dataloader,
    system_model,
    device: str = "cpu",
    uncertainty_threshold: float = None
) -> dict:
    """
    Comprehensive metric evaluation for a model on a DataLoader dataset.
    """
    model.eval()
    rates = []
    latencies = []
    violations = []
    
    dev = torch.device(device)
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(dev)
            h_br = batch["h_br"].to(dev)
            h_ru = batch["h_ru"].to(dev)
            h_bu = batch["h_bu"].to(dev)
            
            start_t = time.perf_counter()
            outputs = model(
                x,
                H_BR=h_br,
                H_RU=h_ru,
                H_BU=h_bu,
                uncertainty_threshold=uncertainty_threshold
            )
            lat = (time.perf_counter() - start_t) / x.size(0)
            latencies.append(lat)
            
            theta = outputs["theta"]
            
            # Rate evaluation
            H_eff = system_model.compute_effective_channel(h_br, h_ru, h_bu, theta)
            W = system_model.compute_active_beamformers_zf(H_eff)
            rate = system_model.compute_achievable_sum_rate(H_eff, W)
            
            rates.extend(rate.cpu().numpy())
            
            # Quantization constraint violation check
            if system_model.phase_alphabet_torch is not None:
                # Check distance to nearest discrete state
                diffs = torch.abs(theta.unsqueeze(-1) - system_model.phase_alphabet_torch.to(dev))
                min_diffs = torch.min(diffs, dim=-1)[0]
                viol = (min_diffs > 1e-2).float().mean().item()
                violations.append(viol)

    rates = np.array(rates)
    return {
        "mean_sum_rate": float(np.mean(rates)),
        "std_sum_rate": float(np.std(rates)),
        "min_sum_rate": float(np.min(rates)),
        "max_sum_rate": float(np.max(rates)),
        "mean_latency_ms": float(np.mean(latencies) * 1000.0),
        "constraint_violation_rate": float(np.mean(violations)) if violations else 0.0
    }

def compute_model_flops(model, input_shape: tuple) -> int:
    """
    Estimates floating point operations (FLOPs) for forward pass.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Approximate 2 * params FLOPs for linear layers
    return 2 * total_params
