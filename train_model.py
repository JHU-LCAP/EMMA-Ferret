from trainer import Trainer

if __name__ == "__main__":
    wandb_api_key = ""
    train_class = Trainer(wandb_api_key, "config-128.yaml", "EMMA_Ferret_3M_shifted", 0, False)
    train_class.run_model_pipeline(layer = 1, stream = 2)
    #train_class.run_model_pipeline(layer = 1, stream = 2)
    #train_class.run_model_pipeline(layer = 1, stream = 3)
