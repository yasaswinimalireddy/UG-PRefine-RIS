import torch
import numpy as np
import pytest
from src.channel.system_model import SystemModel
from src.channel.csi_noise import CSINoiseModel
from src.evaluation.canonical_evaluator import CanonicalEvaluator
from src.models.ug_prefine_model import UGPRefineRISModel
from src.losses.composite_loss import CompositeRISLoss

@pytest.fixture
def setup_sys():
    sys_model = SystemModel(n_ris_elements=64, n_bs_antennas=8, n_users=2, bits=2, snr_db=15.0)
    return sys_model

# 1. Phase Alphabet Validity Test
def test_phase_alphabet_validity(setup_sys):
    alphabet = setup_sys.phase_alphabet
    assert len(alphabet) == 4
    expected = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    np.testing.assert_allclose(alphabet, expected, atol=1e-5)

# 2. Hardware Amplitude Model Test
def test_hardware_amplitude_model(setup_sys):
    theta = torch.tensor([0.0, np.pi/2, np.pi, 3*np.pi/2])
    beta = setup_sys.compute_amplitude(theta)
    assert (beta >= setup_sys.beta_min).all()
    assert (beta <= 1.0).all()

# 3. CSI Noise Model Test
def test_csi_noise_model():
    noise_model = CSINoiseModel(nmse_delta_sq=0.05, seed=42)
    H = np.ones((64, 8), dtype=np.complex64)
    H_noisy = noise_model.add_noise_numpy(H)
    nmse = np.mean(np.abs(H_noisy - H)**2)
    assert np.isclose(nmse, 0.05, atol=0.02)

# 4. Power Constraint Test
def test_power_constraint(setup_sys):
    H_eff = torch.randn(4, 2, 8, dtype=torch.complex64)
    W = setup_sys.compute_active_beamformers_zf(H_eff, p_max=1.0)
    power = torch.sum(torch.abs(W) ** 2, dim=(1, 2))
    torch.testing.assert_close(power, torch.ones_like(power), atol=1e-4, rtol=1e-4)

# 5. ZF Beamforming Residual Test
def test_zf_beamforming_residual(setup_sys):
    H_eff = torch.randn(4, 2, 8, dtype=torch.complex64)
    W = setup_sys.compute_active_beamformers_zf(H_eff, p_max=1.0)
    # H_eff: [4, 2, 8], W^T: [4, 8, 2]
    HW = torch.bmm(H_eff, W.transpose(-2, -1))
    # Off-diagonal interference should be minimal under Zero-Forcing
    off_diag = torch.abs(HW[:, 0, 1]) + torch.abs(HW[:, 1, 0])
    assert (off_diag < 1e-4).all()

# 6. Achievable Rate Positive Test
def test_achievable_rate_positive(setup_sys):
    H_eff = torch.randn(4, 2, 8, dtype=torch.complex64)
    W = setup_sys.compute_active_beamformers_zf(H_eff, p_max=1.0)
    rate = setup_sys.compute_achievable_sum_rate(H_eff, W)
    assert (rate > 0).all()

# 7. Gumbel Output Format & STE Test
def test_gumbel_output_format(setup_sys):
    model = UGPRefineRISModel(setup_sys)
    logits = torch.randn(2, 64, 4)
    theta, probs = model.quantizer(logits, hard=True)
    assert theta.shape == (2, 64)
    assert probs.shape == (2, 64, 4)

# 8. Uncertainty Positivity Test
def test_uncertainty_positivity(setup_sys):
    model = UGPRefineRISModel(setup_sys)
    x = torch.randn(2, 2, 16, 64)
    out = model(x)
    var = out["uncertainty_var"]
    assert (var > 0).all()

# 9. Router Selection Mask Logic Test
def test_router_selection_logic(setup_sys):
    model = UGPRefineRISModel(setup_sys)
    x = torch.randn(4, 2, 16, 64)
    H_BR = torch.randn(4, 64, 8, dtype=torch.complex64)
    H_RU = torch.randn(4, 2, 64, dtype=torch.complex64)
    H_BU = torch.randn(4, 2, 8, dtype=torch.complex64)
    
    out = model(x, H_BR=H_BR, H_RU=H_RU, H_BU=H_BU, uncertainty_threshold=0.0)
    assert out["theta"].shape == (4, 64)

# 10. PGD Phase Update Step Test
def test_pgd_phase_update_step(setup_sys):
    model = UGPRefineRISModel(setup_sys, unrolled_pgd_steps=2)
    theta_init = torch.zeros(2, 64)
    H_BR = torch.randn(2, 64, 8, dtype=torch.complex64)
    H_RU = torch.randn(2, 2, 64, dtype=torch.complex64)
    H_BU = torch.randn(2, 2, 8, dtype=torch.complex64)
    
    theta_refined = model.unrolled_pgd_refine(theta_init, H_BR, H_RU, H_BU)
    assert theta_refined.shape == (2, 64)

# 11. PGD Gradient Flow Test
def test_pgd_gradient_flow(setup_sys):
    model = UGPRefineRISModel(setup_sys, unrolled_pgd_steps=1)
    x = torch.randn(2, 2, 16, 64)
    H_BR = torch.randn(2, 64, 8, dtype=torch.complex64)
    H_RU = torch.randn(2, 2, 64, dtype=torch.complex64)
    H_BU = torch.randn(2, 2, 8, dtype=torch.complex64)
    
    loss_fn = CompositeRISLoss(setup_sys)
    outputs = model(x, H_BR=H_BR, H_RU=H_RU, H_BU=H_BU, force_refine=True)
    loss, _ = loss_fn(outputs, H_BR, H_RU, H_BU)
    loss.backward()
    assert model.backbone.stem[0].weight.grad is not None

# 12. Final Quantization Constraint Compliance Test
def test_final_quantization_constraint(setup_sys):
    evaluator = CanonicalEvaluator(setup_sys)
    theta = torch.tensor([[0.0, np.pi/2, np.pi, 3*np.pi/2] * 16])
    H_BR = torch.randn(1, 64, 8, dtype=torch.complex64)
    H_RU = torch.randn(1, 2, 64, dtype=torch.complex64)
    H_BU = torch.randn(1, 2, 8, dtype=torch.complex64)
    
    res = evaluator.evaluate(H_BR, H_RU, H_BU, theta)
    assert res["violation_rate"] == 0.0

# 13. Model Train/Eval Consistency Test
def test_model_train_eval_consistency(setup_sys):
    model = UGPRefineRISModel(setup_sys)
    x = torch.randn(2, 2, 16, 64)
    model.train()
    out_train = model(x)
    model.eval()
    out_eval = model(x)
    assert out_train["theta"].shape == out_eval["theta"].shape

# 14. Deterministic Inference Repeatability Test
def test_deterministic_inference_repeatability(setup_sys):
    torch.manual_seed(42)
    model = UGPRefineRISModel(setup_sys)
    model.eval()
    x = torch.randn(2, 2, 16, 64)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1["theta"], out2["theta"])

# 15. Batch-vs-Single-Sample Inference Consistency Test
def test_batch_vs_single_sample_consistency(setup_sys):
    torch.manual_seed(42)
    model = UGPRefineRISModel(setup_sys)
    model.eval()
    x_batch = torch.randn(4, 2, 16, 64)
    with torch.no_grad():
        out_batch = model(x_batch)
        out_single = model(x_batch[:1])
    torch.testing.assert_close(out_batch["theta"][:1], out_single["theta"])
