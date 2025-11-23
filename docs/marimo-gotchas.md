# Marimo Gotchas

Common pitfalls and solutions when working with marimo notebooks.

## Variable Names Must Be Unique Across Cells

**Problem:** Marimo requires all variables to be unique across all cells in a notebook.

**Error:**
```
critical[multiple-definitions]: Variable 'np' is defined in multiple cells
```

**Wrong:**
```python
@app.cell
def _():
    import numpy as np
    return np,

@app.cell
def _():
    import numpy as np  # ❌ Error: np already defined
    return np,
```

**Right:**
```python
@app.cell
def _():
    # Import once at the top
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    return np, plt, torch,

@app.cell
def _(np, plt, torch):
    # Use imported modules
    data = np.array([1, 2, 3])
    return data,
```

**Solution:** Consolidate all imports into a single cell at the top of your notebook.

## Importing Local Project Modules

**Problem:** When importing from local project modules (e.g., `from src.types import TrainConfig`), marimo can't find the modules by default.

**Error:**
```
ModuleNotFoundError: No module named 'src.types'
```

**Solution:** Add pythonpath configuration to `pyproject.toml`:

For project structure:
```
.
├── notebooks/
│   └── my_notebook.py
├── pyproject.toml
└── src/
    ├── __init__.py
    └── types.py
```

**If importing src as a package** (`from src.types import ...`):
```toml
[tool.marimo.runtime]
pythonpath = ["."]
```

**If importing modules directly from src** (`from types import ...`):
```toml
[tool.marimo.runtime]
pythonpath = ["src"]
```

Restart marimo after changing `pyproject.toml` for changes to take effect.

## PyTorch Device Mismatches

**Problem:** Model loaded from checkpoint might be on different device than input tensors.

**Error:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
```

**Wrong:**
```python
device = "cpu"
model = load_model(checkpoint_path)  # Model might be on MPS/CUDA
inputs = torch.tensor([1, 2, 3], device=device)  # ❌ Device mismatch
outputs = model(inputs)
```

**Right:**
```python
model = load_model(checkpoint_path)
device = next(model.parameters()).device  # Auto-detect model's device
inputs = torch.tensor([1, 2, 3], device=device)  # ✅ Same device
outputs = model(inputs)
```

**Solution:** Auto-detect the model's device instead of hardcoding it.

## Checking for Errors

Run `marimo check` before launching the notebook to catch errors early:

```bash
uv run marimo check notebooks/my_notebook.py
```

This will show all variable conflicts and other issues.

## Reference

- [Marimo docs: Importing local modules](https://docs.marimo.io/guides/package_management/importing_packages.html#importing-local-modules)
- [Marimo docs: Multiple definitions](https://docs.marimo.io/guides/lint_rules/#multiple-definitions)
