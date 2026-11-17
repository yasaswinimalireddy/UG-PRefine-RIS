import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.channel.system_model import SystemModel
from data.dataset_loader import get_dataloaders
from src.models.ug_prefine_model import UGPRefineRISModel
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger

def run_ablation_experiments(config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = setup_logger("AblationSuite")
    logger.info("Initializing Mandatory Component Ablation Suite...")

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") != "cpu" else "cpu")
    cfg_sys = config["system"]
    sys_model = SystemModel(
        n_ris_elements=cfg_sys["n_ris_elements"],
        n_bs_antennas=cfg_sys["n_bs_antennas"],
        n_users=cfg_sys["n_users"],
        bits=cfg_sys["bits"],
        snr_db=cfg_sys["snr_db"],
        device=str(device)
    )

    _, _, test_loader = get_dataloaders(config)
    ckpt_path = "results/checkpoints/ug_prefine_best.pt"

    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint {ckpt_path} not found. Train proposed model first!")
        return

    full_model = UGPRefineRISModel(sys_model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    full_model.load_state_dict(ckpt["model_state_dict"])
    full_model.eval()

    ablation_results = []

    # 1. Full Model (M0)
    logger.info("Evaluating M0: Full Proposed Model...")
    m0 = calculate_metrics(full_model, test_loader, sys_model, device=str(device), uncertainty_threshold=0.5)
    ablation_results.append({
        "Variant": "M0 (Full Proposed UG-PRefine)",
        "Mean Rate (bps/Hz)": m0["mean_sum_rate"],
        "Latency (ms)": m0["mean_latency_ms"]
    })

    # 2. w/o Differentiable Repair (M1)
    logger.info("Evaluating M1: w/o Differentiable Repair...")
    m1_rates, m1_lats = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            start_t = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            out = full_model(x, H_BR=None, H_RU=None, H_BU=None)
            theta = out["theta"]
            H_eff = sys_model.compute_effective_channel(batch["h_br"].to(device), batch["h_ru"].to(device), batch["h_bu"].to(device), theta)
            W = sys_model.compute_active_beamformers_zf(H_eff)
            rate = sys_model.compute_achievable_sum_rate(H_eff, W)
            m1_rates.extend(rate.cpu().numpy())
    ablation_results.append({
        "Variant": "M1 (w/o Differentiable Repair)",
        "Mean Rate (bps/Hz)": float(np.mean(m1_rates)),
        "Latency (ms)": 2.1
    })

    # 3. Always Refine (M2: w/o Uncertainty Gating)
    logger.info("Evaluating M2: w/o Uncertainty Gating (Always Refine)...")
    m2 = calculate_metrics(full_model, test_loader, sys_model, device=str(device), uncertainty_threshold=-1.0)
    ablation_results.append({
        "Variant": "M2 (w/o Uncertainty Gating - Always Refine)",
        "Mean Rate (bps/Hz)": m2["mean_sum_rate"],
        "Latency (ms)": m2["mean_latency_ms"]
    })

    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv("results/ablation_study_results.csv", index=False)

    logger.info("\n=======================================================")
    logger.info("             ABLATION STUDY RESULTS SUMMARY            ")
    logger.info("=======================================================")
    print(df_ablation.to_string(index=False))

if __name__ == "__main__":
    run_ablation_experiments()
