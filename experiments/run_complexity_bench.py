import os
import sys
import yaml
import torch
import pandas as pd

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.channel.system_model import SystemModel
from src.models.ug_prefine_model import UGPRefineRISModel
from src.optimizers.alternating_optimization import AlternatingOptimization
from src.utils.metrics import compute_model_flops
from src.utils.logger import setup_logger

def run_complexity_benchmark():
    logger = setup_logger("ComplexityBench")
    logger.info("Initializing FLOPs and Computational Complexity Benchmark...")

    ris_sizes = [32, 64, 128, 256]
    bench_results = []

    for N in ris_sizes:
        sys_model = SystemModel(n_ris_elements=N, n_bs_antennas=8, n_users=2, bits=2)
        model = UGPRefineRISModel(sys_model)
        
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        flops = compute_model_flops(model, (2, 16, N))
        
        # Estimate theoretical AO complexity: O(I_iter * (N^3 + M^3 + K*N^2))
        ao_flops = 20 * (N**3 + 8**3 + 2 * (N**2))
        speedup = ao_flops / max(1, flops)

        bench_results.append({
            "RIS Elements (N)": N,
            "Parameters": f"{params:,}",
            "Neural FLOPs": f"{flops:,}",
            "AO Classical FLOPs": f"{ao_flops:,}",
            "Theoretical Speedup Factor": f"{speedup:.1f}x"
        })

    df_bench = pd.DataFrame(bench_results)
    df_bench.to_csv("results/complexity_benchmark_results.csv", index=False)

    logger.info("\n=======================================================")
    logger.info("        COMPUTATIONAL COMPLEXITY BENCHMARK SUMMARY    ")
    logger.info("=======================================================")
    print(df_bench.to_string(index=False))

if __name__ == "__main__":
    run_complexity_benchmark()
