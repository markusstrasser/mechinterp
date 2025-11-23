import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pathlib import Path
    return Path, go, json, make_subplots, mo, pd, px


@app.cell
def _(Path, json, mo, pd):
    # Load Data
    _data_path = Path("notebooks/data/grokking_dynamics.json")
    
    if not _data_path.exists():
        # Fallback if running from root
        _data_path = Path("notebooks/data/grokking_dynamics.json")
        
    with open(_data_path, "r") as _f:
        _data = json.load(_f)
        
    df = pd.DataFrame(_data)
    df = df.sort_values("step")
    
    # Narrative Helper
    def get_phase(step):
        if step < 2000: return "Memorization"
        if step < 3500: return "Compression"
        return "Generalization"
        
    df["Phase"] = df["step"].apply(get_phase)

    mo.md(r"""
    # 🕵️‍♀️ The Grokking Detective Story

    **The Case:** A neural network is learning to add numbers ($a+b \pmod{{113}}$).
    
    **The Evidence:** For the first 2,000 steps, it fails miserably (Acc < 20%). Then, almost instantly at step 3,500, it becomes a genius (Acc > 99%).
    
    **The Question:** What happened in the dark?
    
    We placed "wiretaps" (checkpoints) every 20 steps inside the model to find out.
    """)
    return df, get_phase, _data_path


@app.cell
def _(df, mo, px):
    mo.md("""
    ## Clue #1: The "Sudden" Insight
    
    First, let's look at what the outside world sees: **Test Accuracy**.
    
    It looks like a miracle. Nothing happens, and then *everything* happens.
    If you only looked at this chart, you'd think the model just "got lucky" or "had an epiphany."
    
    *Hover over the line to see the step number.*
    """)
    
    _fig = px.line(
        df, x="step", y="acc", 
        title="The Miracle Curve (Test Accuracy)",
        labels={"step": "Training Step", "acc": "Accuracy (0-1)"},
        template="plotly_white"
    )
    _fig.update_traces(line_color="#00CC96", line_width=3)
    _fig.add_vline(x=3500, line_dash="dash", annotation_text="The Jump")
    
    mo.ui.plotly(_fig)
    return


@app.cell
def _(df, mo, px):
    mo.md("""
    ## Clue #2: The Hidden Cost (L2 Norm)
    
    Now let's look at the "energy bill" of the model—the **L2 Norm** of its weights.
    Think of this as **How complex is the solution?** or **How much memory is it using?**
    
    *   **Phase 1 (Memorization):** The model panics. It tries to memorize every answer. This is "expensive" (High Norm).
    *   **Phase 2 (Compression):** The "Weight Decay" tax bill arrives. The model realizes it can't afford the memorization strategy. It starts **deleting** information.
    
    Notice: The norm *drops* (Step 2000-3500) **before** the accuracy jumps.
    """)
    
    _fig = px.line(
        df, x="step", y="norm", 
        title="The Energy Bill (Model Complexity)",
        labels={"step": "Training Step", "norm": "L2 Weight Norm"},
        template="plotly_white"
    )
    _fig.update_traces(line_color="#EF553B", line_width=3)
    _fig.add_shape(type="rect", x0=2000, x1=3500, y0=df["norm"].min(), y1=df["norm"].max(), 
                   fillcolor="red", opacity=0.1, layer="below", line_width=0)
    _fig.add_annotation(x=2750, y=36, text="The Squeeze (Compression)", showarrow=False)
    
    mo.ui.plotly(_fig)
    return


@app.cell
def _(df, go, make_subplots, mo):
    mo.md("""
    ## Clue #3: The "Trash Compactor" (Variance)
    
    This is the smoking gun. We measure **Within-Class Variance**.
    Basically: *How confused is the model about numbers that should be the same?*
    
    *   **High Variance:** "I think 5+7 is totally different from 1+11." (Chaos)
    *   **Low Variance:** "All ways to make 12 are the same thing." (Order)
    
    Watch the purple line. In the "Squeeze" zone, the variance **crashes**. 
    The model is being forced to crush its messy memorized data into a tiny, efficient format.
    
    **The Grokking Point (Step 3500)** happens exactly when the "trash compactor" finishes its job.
    """)
    
    _fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    _fig.add_trace(
        go.Scatter(x=df["step"], y=df["norm"], name="Complexity (Norm)", line=dict(color="#EF553B", width=2)),
        secondary_y=False
    )
    
    _fig.add_trace(
        go.Scatter(x=df["step"], y=df["rnc1"], name="Confusion (Variance)", line=dict(color="#AB63FA", width=3)),
        secondary_y=True
    )
    
    _fig.update_layout(title="The Trash Compactor: Crushing Confusion", template="plotly_white")
    _fig.update_yaxes(title_text="Complexity (Norm)", secondary_y=False)
    _fig.update_yaxes(title_text="Confusion (Variance)", secondary_y=True, type="log") # Log scale for variance
    
    _fig.add_vline(x=3500, line_dash="dash", annotation_text="Grokking!", line_color="green")

    mo.ui.plotly(_fig)
    return


@app.cell
def _(df, mo, px):
    mo.md("""
    ## The Reveal: The "S" Curve of Learning
    
    If we plot **Complexity (Norm)** vs **Accuracy**, we see the full story in one loop.
    
    1.  **The Trap:** The model runs *up* (High Complexity) to get a little bit of accuracy. It gets stuck.
    2.  **The Escape:** It slides *left* (Reducing Complexity) while staying bad at the task.
    3.  **The Discovery:** Once it's simple enough, it shoots *up* (High Accuracy).
    
    **Press Play** below to watch the trajectory.
    """)
    
    _fig = px.scatter(
        df, x="norm", y="acc", animation_frame="step", 
        animation_group="step",
        color="Phase", 
        size_max=20,
        range_x=[27, 38], range_y=[-0.05, 1.05],
        title="The Learning Trajectory",
        labels={"norm": "Complexity (L2 Norm)", "acc": "Performance (Accuracy)"},
        template="plotly_white",
        color_discrete_map={
            "Memorization": "#EF553B",
            "Compression": "#FFA15A",
            "Generalization": "#00CC96"
        }
    )
    
    # Add a "tail" trace to show history in the background? 
    # Plotly animation frames replace data, so to show history we'd need cumulative frames.
    # Instead, let's show the static path in grey and the dot moving on top.
    
    # Note: Doing that in pure px with animation_frame is tricky. 
    # Let's keep it simple for the "Competent but not deep" reader.
    # The moving dot is engaging.
    
    mo.ui.plotly(_fig)
    return


if __name__ == "__main__":
    app.run()