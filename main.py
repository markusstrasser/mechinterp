import wandb
from environs import Env



def main():
    from environs import Env
    env = Env()
    env.read_env()  # loads .env
    wandb_api_key = env.str("WANDB")
    print("Hello from marim!")
    wandb.login(key=wandb_api_key)


if __name__ == "__main__":
    main()
