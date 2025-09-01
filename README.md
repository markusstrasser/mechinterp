# TODO

* Robust checkpointing + loading (for AI agent to interrupt execution and add hooks)
* Make hooks and logging modular and pluggable
* What should be filmed? At what point? 
* 3 Models: "no", "almost" "yes" , ... maybe "yes, easily"
* 

```
├── configs/
│   └── grokking_p113.yaml
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py         # Dataset generation
│   ├── model.py        # Model definition
│   ├── probes.py       # Probe definitions & registry
│   └── runner.py       # Training loop & ProbeRunner
└── scripts/
    ├── train.py        # Executes training runs
    └── analyze.py      # Consumes model artifacts for analysis

```