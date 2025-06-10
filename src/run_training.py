from config import get_domain_adaptation_config, print_config_summary
from trainer import DANNTrainer
import time


def main():
    # Get the fully configured training settings
    config = get_domain_adaptation_config()
    config.debug_augmentations_visualisation = True
    # Print configuration summary
    print_config_summary(config)

    # Create trainer and start training
    print("Starting domain adversarial training...")
    start_time = time.time()

    trainer = DANNTrainer(config)
    trainer.train()

    # Print training summary
    end_time = time.time()
    training_time = end_time - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best validation Dice score: {trainer.best_dice:.4f}")
    print(f"Model saved to: {config.model_path}")


if __name__ == "__main__":
    main()
