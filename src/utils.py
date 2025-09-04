

def checkpoint_name_from_config(config, with_extension=False):
    return (
        f"d{config.d_model}-wd{config.weight_decay}-h{config.n_heads}-ffn{config.d_ffn}"
        f"{'.pt' if with_extension else ''}"
    )