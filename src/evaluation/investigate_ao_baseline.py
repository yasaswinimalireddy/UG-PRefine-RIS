import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.channel.system_model import SystemModel
from src.evaluation.canonical_evaluator import CanonicalEvaluator
from src.evaluation.frozen_test_set import FrozenTestSet
from src.optimizers.alternating_optimization import AlternatingOptimization

def investigate_ao_multi_start():
    sys_model = SystemModel(n_ris_elements=64, n_bs_antennas=8, n_users=2, bits=2)
    evaluator = CanonicalEvaluator(sys_model)

    test_set_gen = FrozenTestSet(n_samples=50, n_ris_elements=64, n_bs_antennas=8, n_users=2)
    data = test_set_gen.load_or_create()

    H_BR = data["h_br_true"]
    H_RU = data["h_ru_true"]
    H_BU = data["h_bu_true"]

    start_counts = [1, 5, 10]
    ao_summary = []

    print("=======================================================================")
    print("        CORRECTED ALTERNATING OPTIMIZATION (AO) BENCHMARK               ")
    print("=======================================================================")

    for S in start_counts:
        ao_continuous = AlternatingOptimization(sys_model, max_iters=3, num_starts=S, grid_resolution=16)
        ao_quantized = AlternatingOptimization(sys_model, max_iters=3, num_starts=S, grid_resolution=4)

        # 1. Continuous AO
        _, r_cont, t_cont = ao_continuous.solve_batch(H_BR, H_RU, H_BU, discrete=False)

        # 2. Quantized 2-Bit AO
        _, r_quant, t_quant = ao_quantized.solve_batch(H_BR, H_RU, H_BU, discrete=True)

        ao_summary.append({
            "Multi-Starts (S)": S,
            "Continuous AO Mean (bps/Hz)": float(np.mean(r_cont)),
            "Continuous AO Std": float(np.std(r_cont)),
            "Continuous Latency (ms)": float(t_cont / 50.0 * 1000.0),
            "Quantized 2-bit AO Mean (bps/Hz)": float(np.mean(r_quant)),
            "Quantized 2-bit AO Std": float(np.std(r_quant)),
            "Quantized Latency (ms)": float(t_quant / 50.0 * 1000.0)
        })

    df_ao = pd.DataFrame(ao_summary)
    print(df_ao.to_string(index=False))

    os.makedirs("results/audit", exist_ok=True)
    df_ao.to_csv("results/audit/ao_baseline_investigation.csv", index=False)

if __name__ == "__main__":
    investigate_ao_multi_start()
