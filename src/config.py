# config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union


@dataclass
class DANNTrainingConfig:
    # Required parameters
    data_dir: str
    target_data_dir: str

    # Model saving
    model_path: str  # For saving/loading model

    # Model input/output
    in_channels: int = 1
    out_channels: int = 12  # Number of rib pairs/segments

    # Train-validation split
    train_val_split: float = 0.9  # 80% training, 20% validation

    # Model architecture (matching nnUNet 2D)
    features_per_stage: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 320, 320])
    n_stages: int = 6
    kernel_size: int = 3
    strides: List[int] = field(default_factory=lambda: [1, 2, 2, 2, 2, 2])
    n_conv_per_stage: int = 2

    # Traditional loss weights (fallback when not using combo loss)
    dice_loss_weight: float = 0.5
    ce_loss_weight: float = 0.5

    # --- NEW Loss settings ---
    use_combo_loss: bool = False  # Flag to use the combo loss
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "cldice": 0.8,
        "focal": 0.2,
    })
    channel_weights: List[float] = field(default_factory=lambda: [1.0] * 12)  # Weight for each output channel

    # Domain adaptation settings
    domain_loss_weight: float = 0.1  # Weight for bottleneck domain classification loss
    skip_domain_loss_weight: float = 0.0  # Set to 0 for testing - will activate skip DANN but with no effect
    max_dann_alpha: float = 0.035  # Maximum alpha for gradient reversal layer

    # --- NEW Skip Connection Architecture ---
    skip_bottleneck_factor: int = 4  # Channel reduction factor in skip connections (higher = more compression)
    use_canonical_style: bool = True  # Whether to use learned canonical style in skip connections

    # Optimizer settings
    optimizer_type: str = "SGD"  # 'SGD' or 'AdamW'
    learning_rate: float = 0.0003  # Initial learning rate
    domain_classifier_lr: float = 0.00001  # NEW: Separate LR for domain classifiers
    min_learning_rate: float = 1e-4  # Minimum learning rate
    momentum: float = 0.95  # SGD momentum
    weight_decay: float = 3e-5  # L2 regularization
    nesterov: bool = True  # Use Nesterov momentum
    grad_clip: float = 12.0  # Gradient clipping value

    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # 'poly' or 'cosine'
    lr_patience: int = 10  # Patience for ReduceLROnPlateau

    # Training settings
    num_epochs: int = 1000
    save_checkpoint_every: int = 100  # Save a checkpoint every N epochs

    # Data loading
    train_batch_size: int = 16  # nnUNet's 2D batch size
    val_batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True

    # Dynamic patch sizing
    patch_size: Tuple[int, int] = (512, 512)  # Fixed patch size

    # Spacing settings
    target_spacing: Tuple[float, float] = field(default_factory=lambda: (1.0, 1.0))  # Target voxel spacing

    # Debug settings
    debug_augmentations_visualisation: bool = False  # Whether to save debug visualizations

    # ===== AUGMENTATION PARAMETERS =====
    # Zoom
    p_zoom: float = 0.3  # Probability of applying zoom
    zoom_range: Tuple[float, float] = field(default_factory=lambda: (0.8, 1.2))  # More aggressive zoom range

    # Rotation
    p_rotate: float = 0.2  # Probability of applying rotation
    rotate_range: Tuple[float, float] = field(default_factory=lambda: (-10.0, 10.0))  # Rotation range in degrees

    # Gaussian noise
    p_gaussian_noise: float = 0.15  # Probability of applying Gaussian noise
    gaussian_noise_std: float = 0.01  # Standard deviation of Gaussian noise

    p_scoliosis: float = 0.5

    # Flipping
    p_flip: float = 0.5  # Probability of flipping

    p_adjust_contrast: float = 0.5  # Probability of histogram shift

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.data_dir:
            raise ValueError("data_dir must be provided for source domain (DRR) data")

        # Validate skip connection parameters
        if self.skip_bottleneck_factor < 1:
            raise ValueError("skip_bottleneck_factor must be >= 1")

        if self.skip_domain_loss_weight < 0:
            raise ValueError("skip_domain_loss_weight must be >= 0")

        if self.optimizer_type not in ["SGD", "AdamW"]:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

        if self.lr_scheduler not in ["poly", "cosine", "plateau", None]:
            raise ValueError(f"Unsupported lr_scheduler: {self.lr_scheduler}")

        if self.target_data_dir is None:
            print("Warning: No target domain data directory provided. Running without domain adaptation.")
        else:
            if not Path(self.target_data_dir).exists():
                raise ValueError(f"Target domain data directory does not exist: {self.target_data_dir}")

        # Validate loss weights
        if self.use_combo_loss:
            total_weight = sum(self.loss_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                print(f"Warning: Combo loss weights sum to {total_weight:.3f}, not 1.0")
        else:
            total_traditional = self.dice_loss_weight + self.ce_loss_weight
            if abs(total_traditional - 1.0) > 0.01:
                print(f"Warning: Traditional loss weights sum to {total_traditional:.3f}, not 1.0")


# Default configuration for domain adversarial rib segmentation
def get_domain_adaptation_config() -> DANNTrainingConfig:
    """Get the configuration for domain adversarial rib segmentation training."""
    # Set up paths
    model_dir = "../models"

    # Ensure model directory exists
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    model_path = str(Path(model_dir) / "best_model.pt")  # Changed name slightly

    # Create config with all parameters set
    config = DANNTrainingConfig(
        data_dir="../data/source_domain",
        target_data_dir="../data/target_domain",
        model_path=model_path,

        # --- Enable and configure custom loss ---
        use_combo_loss=False,
        # loss_weights={
        #     "cldice": 0.8,
        #     "focal": 0.2,
        # },
        # channel_weights=[1.0] * 12,  # Using your out_channels for rib segments

        # Standard configuration parameters
        domain_loss_weight=0.25,
        skip_domain_loss_weight=1,  # Activate skip connection domain adaptation
        max_dann_alpha=0.05,

        # Optimizer settings
        optimizer_type="AdamW",
        learning_rate=0.0002,
        domain_classifier_lr=0.0002,

        # Training settings
        num_epochs=500,

        debug_augmentations_visualisation=True,
    )

    # Validate the configuration
    config.validate()

    return config


def print_config_summary(config: DANNTrainingConfig) -> None:
    """Print a summary of the domain adversarial training configuration."""
    print("\n=== Domain Adversarial Training Configuration ===")
    print(f"Source domain data directory: {config.data_dir}")

    if config.target_data_dir:
        print(f"Target domain data directory: {config.target_data_dir}")
        print(f"Domain adaptation enabled with:")
        print(f"- Bottleneck domain loss weight: {config.domain_loss_weight}")
        print(f"- Skip domain loss weight: {config.skip_domain_loss_weight}")  # Will show 0.0
        print(f"- Max GRL alpha: {config.max_dann_alpha}")
        print(f"- Skip bottleneck factor: {config.skip_bottleneck_factor}")  # NEW
        print(f"- Use canonical style: {config.use_canonical_style}")  # NEW
    else:
        print("No target domain data provided. Running without domain adaptation.")

    print(f"Model save path: {config.model_path}")
    print(f"Train/val split: {config.train_val_split:.1f}/{1 - config.train_val_split:.1f} ")
    print(f"Batch sizes: {config.train_batch_size} (train), {config.val_batch_size} (val)")
    print(f"Max epochs: {config.num_epochs}")
    print(f"Optimizer: {config.optimizer_type} (lr: {config.learning_rate})")
    print(f"Domain classifier LR: {config.domain_classifier_lr}")  # NEW

    # --- Print loss configuration ---
    if getattr(config, 'use_combo_loss', False):
        print(f"Using custom combo loss with weights:")
        for loss_type, weight in config.loss_weights.items():
            print(f"- {loss_type}: {weight}")
    else:
        print(f"Using standard losses: Dice {config.dice_loss_weight}, CE {config.ce_loss_weight}")

    print(f"LR scheduler: {config.lr_scheduler}")

    print("\nAugmentation parameters:")
    print(f"- Zoom: prob={config.p_zoom}, range={config.zoom_range}")
    print(f"- Rotation: prob={config.p_rotate}, range={config.rotate_range}°")
    print(f"- Gaussian noise: prob={config.p_gaussian_noise}, std={config.gaussian_noise_std}")
    print(f"- Flipping: prob={config.p_flip}")
    print("==============================\n")