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

    # Phase definition for visualization
    def get_phase(step):
        if step < 2000: return "Phase I: Memorization"
        if step < 3500: return "Phase II: Compression"
        return "Phase III: Generalization"

    df["Phase"] = df["step"].apply(get_phase)

    mo.md(fr"""
    # Analysis of Grokking Dynamics

    **Dataset:** High-resolution checkpoint analysis ($N={len(df)}$) from Run `1z2q8rx3`.
    **Sampling Rate:** $\Delta t = 20$ steps.

    This notebook visualizes the trajectory of internal model metrics during the grokking phase transition.
    The data highlights the role of **L2 Norm Regularization** and **Representation Collapse** (Variance reduction)
    as precursors to generalization.
    """)
    return df, get_phase, _data_path


@app.cell
def _(df, mo, px):
    mo.md("""
    ## 1. Test Accuracy Evolution

    The evolution of test accuracy reveals the characteristic delayed generalization curve.
    Note the sharp inflection point at $t \approx 3500$.
    """)

    _fig = px.line(
        df, x="step", y="acc",
        title="Test Accuracy vs. Training Step",
        labels={"step": "Training Step", "acc": "Test Accuracy"},
        template="plotly_white"
    )
    _fig.update_traces(line_color="#00CC96", line_width=2.5)
    _fig.add_vline(x=3500, line_dash="dash", annotation_text="Grokking Point")

    mo.ui.plotly(_fig)
    return


@app.cell
def _(df, mo, px):
    mo.md("""
    ## 2. L2 Weight Norm Dynamics

    The $L_2$ norm of the parameter vector |$\theta||_2$.

    *   **Phase I:** Norm increases as the model minimizes training loss via high-complexity solutions.
    *   **Phase II:** Norm decreases due to weight decay dominance, constraining model capacity.

    This "compression" creates the bottleneck necessary to destabilize the memorization circuit.
    """)

    _fig = px.line(
        df, x="step", y="norm",
        title="L2 Weight Norm Evolution",
        labels={"step": "Training Step", "norm": "L2 Norm ||$\theta$||"},
        template="plotly_white"
    )
    _fig.update_traces(line_color="#EF553B", line_width=2.5)

    # Highlight the compression phase
    _fig.add_shape(type="rect", x0=2000, x1=3500, y0=df["norm"].min(), y1=df["norm"].max(),
                   fillcolor="red", opacity=0.05, layer="below", line_width=0)
    _fig.add_annotation(x=2750, y=36, text="Compression Phase", showarrow=False)

    mo.ui.plotly(_fig)
    return


@app.cell
def _(df, go, make_subplots, mo):
    mo.md(r"""
    ## 3. Within-Class Variance (RNC1)

    The average variance of residual stream representations for inputs belonging to the same output class.

    $$ \text{RNC1} \propto \sum_c \text{Var}(X_c) $$

    The drastic reduction in variance (Phase II) indicates a **collapse** of the representation space,
    aligning representations of identical mathematical results (e.g., $5+7$ and $1+11$).
    This collapse coincides with the norm reduction and immediately precedes generalization.
    """)

    _fig = make_subplots(specs=[[{"secondary_y": True}]])

    _fig.add_trace(
        go.Scatter(x=df["step"], y=df["norm"], name="L2 Norm", line=dict(color="#EF553B", width=2)),
        secondary_y=False
    )

    _fig.add_trace(
        go.Scatter(x=df["step"], y=df["rnc1"], name="Within-Class Variance", line=dict(color="#AB63FA", width=2.5)),
        secondary_y=True
    )

    _fig.update_layout(title="Representation Collapse: Norm and Variance Correlation", template="plotly_white")
    _fig.update_yaxes(title_text="L2 Norm", secondary_y=False)
    _fig.update_yaxes(title_text="Within-Class Variance (Log Scale)", secondary_y=True, type="log")

    _fig.add_vline(x=3500, line_dash="dash", annotation_text="Transition", line_color="green")

    mo.ui.plotly(_fig)
    return


@app.cell
def _(df, mo, px):
    mo.md("""
    ## 4. Phase Space Trajectory

    Visualizing the training dynamics in the **Complexity (Norm) vs. Performance (Accuracy)** plane.

    The trajectory exhibits a hysteresis-like loop:
    1.  **Expansion:** Increasing norm, low accuracy.
    2.  **Compression:** Decreasing norm, low accuracy.
    3.  **Generalization:** Low norm, rapidly increasing accuracy.
    """)

    _fig = px.scatter(
        df, x="norm", y="acc", animation_frame="step",
        animation_group="step",
        color="Phase",
        size_max=15,
        range_x=[27, 38], range_y=[-0.05, 1.05],
        title="Phase Space Trajectory: Norm vs. Accuracy",
        labels={"norm": "L2 Norm (Complexity)", "acc": "Test Accuracy (Performance)"},
        template="plotly_white",
        color_discrete_map={
            "Phase I: Memorization": "#EF553B",
            "Phase II: Compression": "#FFA15A",
            "Phase III: Generalization": "#00CC96"
        }
    )

    mo.ui.plotly(_fig)
    return


if __name__ == "__main__":
    app.run()
