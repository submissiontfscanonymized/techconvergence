import numpy as np
import json
import torch
import psutil

def load_json(filename: str) -> dict:
    with open(filename, 'r') as fp:
        data = json.load(fp)
    data = {k: {int(k2): v2 for k2, v2 in v.items()} for k, v in data.items()}
    return data

def find_total_usable_memory_device(device):    
    if torch.cuda.is_available():
        total = 0.9 * torch.cuda.get_device_properties(device).total_memory / 1024**3
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    else:
        mem = psutil.virtual_memory()
        memory_allocated = 0.9 * mem.available / 1024 ** 3
        return memory_allocated

    return total - memory_allocated

# The following methods are strongly inspired from the Textwiz library from Cyril Vallez
def find_total_usable_memory():
    if torch.cuda.is_available():
        total = 0.9 * torch.cuda.get_device_properties(0).total_memory / 1024**3
        memories = []
        for i in range(torch.cuda.device_count()):
            memories.append(torch.cuda.memory_allocated(torch.cuda.device(i)) / 1024**3)
    else:
        mem = psutil.virtual_memory()
        memories = 0.9 * mem.available / 1024 ** 3
        return memories

    return total - max(memories)


# Inspired by Cyrill Vallez textwiz
def estimate_memory(path_memory_estimate, input_size, max_new_tokens):
    memory_footprints = load_json(path_memory_estimate)
    # Convert keys to int
    memory_footprints = {k1: {int(k2): v2 for k2, v2 in v1.items()} for k1, v1 in memory_footprints.items()}

    passes_r2_test = True
    fit_results = {}

    # Fit the curves
    for key in memory_footprints.keys():
        x = np.array(list(memory_footprints[key].keys()))
        y = np.array(list(memory_footprints[key].values()))
        # Make sure vectors are sorted correctly (dics should stay ordered but you never know)
        sorting = np.argsort(x)
        x = x[sorting]
        y = y[sorting]

        # Memory usage of forward pass without cache is linear when using flash attention implementation, else quadratic.
        if key == 'without cache':
            # First try linear
            fit, stats = np.polynomial.Polynomial.fit(x, y, deg=1, full=True)
            r2 = r_squared(stats[0], y)
            # If bad fit, fallback to quadratic
            if r2 < 0.95:
                fit, stats = np.polynomial.Polynomial.fit(x, y, deg=2, full=True)
        # Memory usage of cache and forward pass using cache is always linear.
        else:
            fit, stats = np.polynomial.Polynomial.fit(x, y, deg=1, full=True)

        r2 = r_squared(stats[0], y)
        # This should always be the case, but check it if for some reason the behavior is not sufficiently linear (or quadratic)
        if r2 < 0.95:
            passes_r2_test = False
        fit_results[key] = fit


    memory_needed_without_cache = fit_results['without cache'](input_size)
    memory_needed_with_cache = fit_results['cache size'](input_size + max_new_tokens) + fit_results['with cache'](input_size + max_new_tokens)
    memory_needed = max(memory_needed_without_cache, memory_needed_with_cache)

    return memory_needed, passes_r2_test

def r_squared(residual: float, y: np.array) -> float:
    """Compute the coefficient of determination (R^2) of a numpy fit."""
    SS_tot = sum((y - np.mean(y))**2)
    return 1 - residual / SS_tot


def infer_best_batch_size_by_heuristics(parameters: int, available_memory: float, input_size: int, max_new_tokens: int) -> int:
        total_size = input_size + max_new_tokens
        chunks = (total_size // 2048) + 1 if total_size % 2048 != 0 else total_size // 2048
        if parameters < 5:
            batch = int(available_memory // (1 * chunks))
        elif parameters < 10:
            batch = int(available_memory // (2 * chunks))
        elif parameters < 20:
            batch = int(available_memory // (3 * chunks))
        else:
            batch = int(available_memory // (4 * chunks))
        
        return max(batch, 1)

def parameters_count(model) -> float:
        return sum(map(torch.numel, model.parameters())) / 1e9
