import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.channel.system_model import SystemModel
from data.dataset_loader import RISChannelDataset
from torch.utils.data import DataLoader
from src.models.ug_prefine_model import UGPRefineRISModel
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger

def run_ood_suite(config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = setup_logger("OODGeneralization")
    logger.info("Initializing Out-of-Distribution Generalization Suite...")

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

    ckpt_path = "results/checkpoints/ug_prefine_best.pt"
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint {ckpt_path} not found.")
        return

    model = UGPRefineRISModel(sys_model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ood_results = []
    nmse_levels = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

    logger.info("Evaluating Performance across Unseen CSI Error NMSE Levels...")

    for nmse in nmse_levels:
        ood_ds = RISChannelDataset(
            n_samples=500,
            n_ris_elements=cfg_sys["n_ris_elements"],
            n_bs_antennas=cfg_sys["n_bs_antennas"],
            n_users=cfg_sys["n_users"],
            csi_nmse=nmse,
            seed=999
        )
        loader = DataLoader(ood_ds, batch_size=64, shuffle=False)

        metrics = calculate_metrics(model, loader, sys_model, device=str(device))
        ood_results.append({
            "CSI NMSE delta^2": nmse,
            "CSI Error (%)": f"{int(nmse * 100)}%",
            "Sum Rate (bps/Hz)": metrics["mean_sum_rate"],
            "Std Rate": metrics["std_sum_rate"],
            "Latency (ms)": metrics["mean_latency_ms"]
        })

    df_ood = pd.DataFrame(ood_results)
    df_ood.to_csv("results/ood_generalization_results.csv", index=False)

    logger.info("\n=======================================================")
    logger.info("       OOD CSI NOISE GENERALIZATION SUMMARY TABLE      ")
    logger.info("=======================================================")
    print(df_ood.to_string(index=False))

if __name__ == "__main__":
    run_ood_suite()
