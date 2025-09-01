import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(train):
    import wandb
    import config

    # To run without wandb, you can comment out the `with wandb.init(...)` block and de-indent the line below it.
    wandb.init(project="grokking-interactive-full-code", config=config)
    # Capture both the model and the history object from the train function
    model, history = train(wandb.config)
    wandb.finish()
    print("--- Training Finished ---")

    print("History object is available:", history is not None)
    print("Available history keys:", list(history.keys()))

    return


if __name__ == "__main__":
    app.run()
