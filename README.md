# Adversarial-Ribcage-Segmentation-Network

This repository implements a Domain Adversarial Neural Network (DANN) for robust rib segmentation across different imaging domains. The system uses adversarial training to adapt from synthetic DRR (Digitally Reconstructed Radiograph) images to real X-ray images without requiring labeled target domain data.

## Repository Structure

```
Adversarial-Ribcage-Segmentation-Network/
├── data/
│   ├── source_domain/          # DRR training data (images + labels)
│   └── target_domain/          # Real X-ray data (unlabeled)
├── src/
│   ├── config.py              # Training configuration
│   ├── dataset.py             # Data loading and augmentation
│   ├── gradientReversalFunction.py  # Gradient reversal layer
│   ├── inference.py           # Model inference pipeline
│   ├── losses.py              # Custom loss functions
│   ├── model.py               # DANN architecture
│   ├── run_training.py        # Training script
│   └── trainer.py             # Training loop
├── test_data/                 # Test images for inference
├── models/                    # Saved checkpoints
└── predictions/               # Inference outputs
```

## Installation

To install the required packages:

```bash
pip install -r requirements.txt
```

This installs all necessary dependencies:
- PyTorch: Deep learning framework
- MONAI: Medical imaging toolkit
- NumPy/SciPy: Numerical computations
- scikit-image: Image processing
- Matplotlib: Visualization
- nibabel: Medical image I/O
- Pillow: PNG image handling

## Features

### Domain Adversarial Training
- Gradient reversal layers for domain-invariant feature extraction
- Skip connection adaptation for enhanced domain transfer
- Unified adversarial training without target labels

### Advanced Architecture
- U-Net backbone with MONAI ResidualUnit blocks
- Multi-scale domain adaptation (bottleneck + skip connections)
- Enhanced skip connections with channel attention

### Comprehensive Loss Functions
- Standard: Dice + Binary Cross-Entropy
- Advanced: clDice (centerline) + Focal loss
- Domain: Adversarial losses for domain confusion

## Usage

### Training

1. **Prepare data**:
   - Source domain PNG images: `data/source_domain/images/`
   - Segmentation masks: `data/source_domain/labels/` (NIfTI format)
   - Target domain PNG images: `data/target_domain/images/`

2. **Configure training** in `src/config.py`:
```python
config = DANNTrainingConfig(
    data_dir="../data/source_domain",
    target_data_dir="../data/target_domain",
    num_epochs=500,
    train_batch_size=16,
    learning_rate=0.0002,
    domain_loss_weight=0.25,
    use_combo_loss=False  # Set True for clDice + Focal
)
```

3. **Start training**:
```bash
cd src
python run_training.py
```

### Inference

Run inference on new images:

```bash
cd src
python inference.py
```

Outputs:
- NIfTI segmentation masks in `predictions/`
- Visualization overlays in `predictions/visualizations/`

## Model Architecture

- **Encoder**: 6-stage ResidualUnit blocks [32, 64, 128, 256, 320, 320]
- **Decoder**: Symmetric upsampling with enhanced skip connections
- **Domain adaptation**: Bottleneck + skip connection classifiers with gradient reversal
- **Output**: 12-channel segmentation (one per rib pair)

## Training Monitoring

The system provides real-time monitoring:
- Training/validation Dice scores and losses
- Domain classification accuracies
- t-SNE feature space visualization
- Comprehensive training plots in `models/figures/`

## Configuration Options

### Advanced Losses
```python
config.use_combo_loss = True
config.loss_weights = {"cldice": 0.8, "focal": 0.2}
```

### Domain Adaptation
```python
config.domain_loss_weight = 0.25      # Main adversarial loss
config.skip_domain_loss_weight = 1.0  # Skip connection adaptation
config.max_dann_alpha = 0.05          # GRL strength
```

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.