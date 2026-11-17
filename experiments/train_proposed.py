import os
import sys
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.channel.system_model import SystemModel
from data.dataset_loader import get_dataloaders
from src.models.ug_prefine_model import UGPRefineRISModel
from src.losses.composite_loss import CompositeRISLoss
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger

def train(config_path: str = "configs/default_config.yaml"):
    # Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = setup_logger("TrainProposed")
    logger.info("Initializing UG-PRefine Training Pipeline...")

    # Set seeds
    seed = config["training"]["seed"]
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") != "cpu" else "cpu")
    logger.info(f"Using compute device: {device}")

    # Build System Model
    cfg_sys = config["system"]
    sys_model = SystemModel(
        n_ris_elements=cfg_sys["n_ris_elements"],
        n_bs_antennas=cfg_sys["n_bs_antennas"],
        n_users=cfg_sys["n_users"],
        bits=cfg_sys["bits"],
        snr_db=cfg_sys["snr_db"],
        coupled_amplitude=cfg_sys["coupled_amplitude"],
        beta_min=cfg_sys["beta_min"],
        phi_rad=cfg_sys["phi_rad"],
        gamma_exp=cfg_sys["gamma_exp"],
        device=str(device)
    )

    # DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Model & Loss
    cfg_train = config["training"]
    model = UGPRefineRISModel(
        system_model=sys_model,
        feature_dim=256,
        unrolled_pgd_steps=cfg_train.get("unrolled_pgd_steps", 3),
        pgd_step_size=cfg_train.get("pgd_step_size", 0.05)
    ).to(device)

    loss_weights = cfg_train["loss_weights"]
    loss_fn = CompositeRISLoss(
        system_model=sys_model,
        lambda_rate=loss_weights["lambda_rate"],
        lambda_uncertainty=loss_weights["lambda_uncertainty"],
        lambda_phase_circular=loss_weights["lambda_phase_circular"],
        lambda_quantization=loss_weights["lambda_quantization"]
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg_train["learning_rate"]),
        weight_decay=float(cfg_train["weight_decay"])
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_train["epochs"])

    gumbel_temp = cfg_train["gumbel_temp_init"]
    temp_decay = cfg_train["gumbel_temp_decay"]
    temp_min = cfg_train["gumbel_temp_min"]

    best_val_rate = 0.0
    os.makedirs("results/checkpoints", exist_ok=True)
    checkpoint_path = "results/checkpoints/ug_prefine_best.pt"

    logger.info(f"Starting training for {cfg_train['epochs']} epochs...")

    for epoch in range(1, cfg_train["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_rate = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg_train['epochs']}"):
            x = batch["x"].to(device)
            h_br = batch["h_br_noisy"].to(device)
            h_ru = batch["h_ru_noisy"].to(device)
            h_bu = batch["h_bu_noisy"].to(device)

            target_rate = batch["target_rate"].to(device) if "target_rate" in batch else None

            optimizer.zero_grad()
            outputs = model(x, H_BR=h_br, H_RU=h_ru, H_BU=h_bu, gumbel_temp=gumbel_temp)

            total_loss, loss_dict = loss_fn(outputs, h_br, h_ru, h_bu, target_rate=target_rate)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item()
            running_rate += loss_dict["mean_sum_rate"]

        scheduler.step()
        gumbel_temp = max(temp_min, gumbel_temp * temp_decay)

        epoch_loss = running_loss / len(train_loader)
        epoch_rate = running_rate / len(train_loader)

        # Validation evaluation
        val_metrics = calculate_metrics(model, val_loader, sys_model, device=str(device))
        val_rate = val_metrics["mean_sum_rate"]

        logger.info(
            f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f} | Train Rate: {epoch_rate:.2f} bps/Hz | "
            f"Val Rate: {val_rate:.2f} bps/Hz | Temp: {gumbel_temp:.2f}"
        )

        if val_rate > best_val_rate:
            best_val_rate = val_rate
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rate": val_rate,
                    "config": config
                },
                checkpoint_path
            )
            logger.info(f"--> Saved best model checkpoint (Val Rate: {best_val_rate:.2f} bps/Hz)")

    logger.info("Training complete. Evaluating best checkpoint on Test set...")
    best_ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    test_metrics = calculate_metrics(model, test_loader, sys_model, device=str(device))
    logger.info(f"=== FINAL TEST RESULTS ===")
    logger.info(f"Test Sum Rate: {test_metrics['mean_sum_rate']:.2f} +/- {test_metrics['std_sum_rate']:.2f} bps/Hz")
    logger.info(f"Test Inference Latency: {test_metrics['mean_latency_ms']:.3f} ms")
    logger.info(f"Test Constraint Violation Rate: {test_metrics['constraint_violation_rate'] * 100:.2f}%")

if __name__ == "__main__":
    train()
