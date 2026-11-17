import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.channel.system_model import SystemModel
from src.evaluation.canonical_evaluator import CanonicalEvaluator
from src.evaluation.frozen_test_set import FrozenTestSet

def investigate_random_draws():
    sys_model = SystemModel(n_ris_elements=64, n_bs_antennas=8, n_users=2, bits=2)
    evaluator = CanonicalEvaluator(sys_model)

    test_set_gen = FrozenTestSet(n_samples=100, n_ris_elements=64, n_bs_antennas=8, n_users=2)
    data = test_set_gen.load_or_create()

    H_BR = data["h_br_true"]
    H_RU = data["h_ru_true"]
    H_BU = data["h_bu_true"]
    B = H_BR.size(0)

    draw_counts = [1, 10, 100, 1000]
    summary_results = []

    print("=======================================================================")
    print("        INVESTIGATING RANDOM PHASE BASELINE MULTI-DRAW BEHAVIOR         ")
    print("=======================================================================")

    for D in draw_counts:
        sample_best_rates = []
        sample_mean_rates = []

        for i in range(B):
            h_br_i = H_BR[i:i+1].expand(D, -1, -1)
            h_ru_i = H_RU[i:i+1].expand(D, -1, -1)
            h_bu_i = H_BU[i:i+1].expand(D, -1, -1)

            # Draw D random phase configurations
            rand_theta = torch.rand(D, sys_model.N) * 2 * np.pi
            rand_theta_quant = sys_model.quantize_phase(rand_theta)

            res = evaluator.evaluate(h_br_i, h_ru_i, h_bu_i, rand_theta_quant)
            rates_d = res["sum_rate"].numpy()

            sample_best_rates.append(np.max(rates_d))
            sample_mean_rates.append(np.mean(rates_d))

        summary_results.append({
            "Random Draws per Sample (D)": D,
            "Mean of Random Draws (bps/Hz)": float(np.mean(sample_mean_rates)),
            "Std of Random Draws": float(np.std(sample_mean_rates)),
            "Best-of-D Draw Mean (bps/Hz)": float(np.mean(sample_best_rates)),
            "Best-of-D Draw Max (bps/Hz)": float(np.max(sample_best_rates))
        })

    df_res = pd.DataFrame(summary_results)
    print(df_res.to_string(index=False))

    os.makedirs("results/audit", exist_ok=True)
    df_res.to_csv("results/audit/random_baseline_investigation.csv", index=False)

if __name__ == "__main__":
    investigate_random_draws()
