import itertools
import importlib.util
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pickle

from pyprojroot import here

_spec = importlib.util.spec_from_file_location("opinion_functions", here() / "src" / "opinion_functions.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fun = _mod

homophilyvec = [0, 0.25, 0.5, 0.75, 1]
num_agents_vec = [10**3, 10**4]
m_vec = [2, 5]
num_sim = 1000
minority_fraction = 0.33333

if __name__ == "__main__":
    output_dir = here() / "04_homophily_simulations" / "output"
    output_dir.mkdir(exist_ok=True)

    param_grid = list(itertools.product(homophilyvec, m_vec, num_agents_vec))

    with ProcessPoolExecutor(max_workers=50) as executor:
        futures = [
            executor.submit(fun.run_simulation_wrapper, (homophily, m, num_agents, num_sim, minority_fraction))
            for homophily, m, num_agents in param_grid
        ]

    for future in futures:
        future.result()
