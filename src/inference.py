import torch
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk

matplotlib.use('Agg')

# Add os.environ to disable MONAI warnings about deprecated features
os.environ["MONAI_HIDE_FUTURE_WARNINGS"] = "1"

# Import the DANN model instead of the regular model
from model import DomainAdversarialModel
from dataset import LoadPNGd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    Lambdad,
    SpatialPadd,
    CenterSpatialCropd, NormalizeIntensityd,
)
from monai.data import Dataset
from torch.utils.data import DataLoader


def post_process_segmentation(binary_mask, closing_kernel_size=3, size_threshold_ratio=0.1):
    """
    Apply post-processing to segmentation mask:
    1. Channel-wise morphological closing to fill small holes
    2. Keep only components larger than a percentage of the largest component per channel

    Args:
        binary_mask: Binary segmentation mask [C, H, W]
        closing_kernel_size: Size of the disk kernel for morphological closing
        size_threshold_ratio: Keep components larger than this ratio of the largest component (0.1 = 10%)

    Returns:
        processed_mask: Post-processed binary mask [C, H, W]
    """
    print(
        f"Applying post-processing: closing kernel size={closing_kernel_size}, keeping components >{size_threshold_ratio * 100}% of largest")

    processed_mask = np.zeros_like(binary_mask)

    # Create morphological kernel
    kernel = disk(closing_kernel_size)

    for channel_idx in range(binary_mask.shape[0]):
        channel_mask = binary_mask[channel_idx].astype(bool)

        # Skip empty channels
        if not np.any(channel_mask):
            continue

        # Step 1: Morphological closing to fill small holes
        closed_mask = binary_closing(channel_mask, kernel)

        # Step 2: Keep only components larger than threshold
        if np.any(closed_mask):
            # Label connected components
            labeled_mask = label(closed_mask)

            # Get properties of all regions
            regions = regionprops(labeled_mask)

            if len(regions) > 0:
                # Find the largest component size
                largest_area = max(region.area for region in regions)
                area_threshold = largest_area * size_threshold_ratio

                # Keep components above threshold
                final_mask = np.zeros_like(closed_mask, dtype=bool)
                kept_components = []

                for region in regions:
                    # if region.area >= area_threshold:
                    final_mask |= (labeled_mask == region.label)
                    kept_components.append(region.area)

                processed_mask[channel_idx] = final_mask.astype(binary_mask.dtype)

                print(
                    f"  Channel {channel_idx}: {len(regions)} -> {len(kept_components)} components "
                    f"(largest: {largest_area}, threshold: {area_threshold:.0f}, kept: {kept_components})")
            else:
                print(f"  Channel {channel_idx}: No components found after closing")
        else:
            print(f"  Channel {channel_idx}: Empty after closing")

    return processed_mask


def create_colored_overlay(image, segmentation_mask, output_path, use_probs=True, threshold=0.5):
    """Create a colored overlay of all segmentation channels using plasma colormap.

    Args:
        image: The background image
        segmentation_mask: Either probabilities [C,H,W] or binary mask [C,H,W]
        output_path: Where to save the visualization
        use_probs: If True, segmentation_mask contains probability values that need thresholding
        threshold: Threshold to apply to probability maps
    """
    # Get number of channels
    num_channels = segmentation_mask.shape[0]

    # If using probability maps, apply thresholding
    if use_probs:
        binary_mask = (segmentation_mask > threshold)
    else:
        binary_mask = segmentation_mask > 0

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=100)

    # First subplot: X-ray with segmentation overlay
    # ---------------------------------------------
    # Display the grayscale image as background
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Image with Segmentation Overlay', pad=20)
    ax1.axis('off')

    # Second subplot: White background for segmentation only view
    # ---------------------------------------------------------
    white_bg = np.ones_like(image)
    ax2.imshow(white_bg, cmap='gray')
    ax2.set_title('Segmentation Only', pad=20)
    ax2.axis('off')

    # Add segmentation overlays directly to the plots
    # Process in reverse order so smaller segments appear on top
    cmap = plt.cm.plasma
    mask_added = False  # Flag to track if any mask was added

    for i in range(num_channels - 1, -1, -1):
        mask = binary_mask[i]
        if np.any(mask):
            mask_added = True
            # Get color from colormap
            color = cmap(i / max(num_channels - 1, 1))

            # Create a colored mask for this channel
            colored_mask = np.zeros((*image.shape, 4))
            colored_mask[mask, 0] = color[0]  # R
            colored_mask[mask, 1] = color[1]  # G
            colored_mask[mask, 2] = color[2]  # B

            # Set alpha (transparency)
            colored_mask[mask, 3] = 0.6  # For image overlay

            # Add this channel's overlay to both plots
            ax1.imshow(colored_mask)

            # For the segmentation-only view, use full opacity
            colored_mask[mask, 3] = 1.0
            ax2.imshow(colored_mask)

    # If no mask was added, add a note to the image
    if not mask_added:
        ax1.text(image.shape[1] / 2, image.shape[0] / 2, "No segments detected",
                 color='red', fontsize=20, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7))
        ax2.text(image.shape[1] / 2, image.shape[0] / 2, "No segments detected",
                 color='red', fontsize=20, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7))

    # Save with reduced quality for speed
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close(fig)


def fix_image_shape(image):
    """Ensure grayscale image has the channel dimension at the front (C, H, W)."""
    # If image is already 3D, check if it needs conversion to grayscale
    if len(image.shape) == 3:
        # Convert RGB/RGBA to grayscale if needed
        if image.shape[2] > 1:
            image = np.mean(image, axis=2).astype(np.float32)
        # If it's [H, W, 1], transpose to [1, H, W]
        else:
            image = np.transpose(image, (2, 0, 1))
            return image  # Already in correct format

    # For [H, W] format, add channel dimension to make it [1, H, W]
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]  # Add channel dimension at front

    return image


def get_inference_transforms(patch_size):
    """Get transforms for inference handling different image formats."""
    return Compose([
        # Use LoadPNGd for PNG images
        LoadPNGd(keys=["image"]),

        # Use our custom function to explicitly handle channel dimensions
        # This matches exactly what the dataset class does
        Lambdad(keys=["image"], func=fix_image_shape),

        # Spatial operations
        SpatialPadd(
            keys=["image"],
            spatial_size=patch_size,
            mode="constant"
        ),

        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True
        ),

        CenterSpatialCropd(
            keys=["image"],
            roi_size=patch_size
        ),

        EnsureTyped(keys=["image"])
    ])


def run_inference(model_path, data_dir, output_dir, threshold=0.5, img_extension='.png',
                  apply_postprocessing=True, closing_kernel_size=3, size_threshold_ratio=0.1):
    # Create directories
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True, parents=True)

    # Get images directly from the data directory
    data_dir = Path(data_dir)
    image_dir = data_dir / "images" if (data_dir / "images").exists() else data_dir
    image_files = list(sorted(image_dir.glob(f"*{img_extension}")))
    print(f"Found {len(image_files)} {img_extension} images in {image_dir}")

    if not image_files:
        raise ValueError(f"No {img_extension} images found in {image_dir}")

    # Print detected files for debugging
    print("Images for inference:")
    for idx, image_path in enumerate(image_files):
        print(f"  {idx + 1}. {str(image_path)}")

    # Use a fixed patch size or calculate dynamically
    patch_size = (512, 512)
    print(f"Using patch size for inference: {patch_size}")

    if apply_postprocessing:
        print(
            f"Post-processing enabled: closing kernel size={closing_kernel_size}, keeping components >{size_threshold_ratio * 100}% of largest")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("Successfully loaded model checkpoint")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Extract model configuration
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Found model_state_dict in checkpoint")
        model_state = checkpoint['model_state_dict']
        # Check if there's additional information in the checkpoint
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch'] + 1} epochs")
        if 'dice' in checkpoint:
            print(f"Best validation Dice score: {checkpoint['dice']:.4f}")
    else:
        print("Checkpoint does not contain model_state_dict, assuming it is the state dict itself")
        model_state = checkpoint

    # Default model parameters
    in_channels = 1
    out_channels = 12
    features_per_stage = [32, 64, 128, 256, 320, 320]
    n_stages = 6
    strides = [1, 2, 2, 2, 2, 2]
    n_conv_per_stage = 2

    # Create DANN model
    model = DomainAdversarialModel(
        in_channels=in_channels,
        out_channels=out_channels,
        features_per_stage=features_per_stage[:n_stages],
        n_stages=n_stages,
        strides=strides[:n_stages],
        n_conv_per_stage=n_conv_per_stage,
    ).to(device)
    print(f"Created DANN model with {n_stages} stages, features: {features_per_stage[:n_stages]}")

    # Load state dictionary with error checking
    try:
        # Check if there are any missing or unexpected keys
        incompatible_keys = model.load_state_dict(model_state, strict=False)
        if incompatible_keys.missing_keys:
            print(f"Warning: Missing keys in state dict: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {incompatible_keys.unexpected_keys}")

        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        # Print model structure and state dict keys for debugging
        print("\nModel structure:")
        for name, _ in model.named_parameters():
            print(f"  {name}")

        print("\nState dict keys:")
        for key in model_state.keys():
            print(f"  {key}")
        raise

    # Set model to evaluation mode
    model.eval()

    # Get inference transforms
    transforms = get_inference_transforms(patch_size)

    # Run inference on each image
    print(f"Running inference on {len(image_files)} images")

    with torch.no_grad():
        for idx, image_path in enumerate(image_files):
            print(f"Processing image {idx + 1}/{len(image_files)}: {image_path}")
            filename = image_path.stem  # Get filename without extension

            try:
                # Create data dict for this image
                data_dict = {"image": str(image_path)}

                # Create dataset for this single image
                dataset = Dataset([data_dict], transform=transforms)

                # Create DataLoader
                data_loader = DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0
                )

                # Process the image
                for batch in data_loader:
                    # Get image dimensions for output NIfTI
                    img = Image.open(image_path)
                    img_width, img_height = img.size

                    # Create a basic affine matrix for the NIfTI output
                    affine = np.eye(4)
                    header = nib.Nifti1Header()

                    # Forward pass
                    images = batch["image"].to(device)

                    # Make sure it's the right shape (B,C,H,W)
                    if images.ndim != 4:
                        print(f"Warning: Unexpected input shape {images.shape}, adjusting...")
                        if images.ndim == 3:  # Missing batch dimension
                            images = images.unsqueeze(0)
                        if images.ndim == 2:  # Missing batch and channel dimensions
                            images = images.unsqueeze(0).unsqueeze(0)

                    # Check that channel dimension is 1
                    if images.shape[1] != 1:
                        print(f"Warning: Expected 1 channel, got {images.shape[1]}. Using first channel.")
                        images = images[:, 0:1]

                    # For inference, we don't need domain_label
                    outputs = model(images)

                    # Apply sigmoid and threshold
                    probs = torch.sigmoid(outputs)
                    binary_mask = (probs > threshold).float()

                    # FIXED: Keep original channel order (no reversal)
                    # The top rib pair should be on the first channel (channel 0)
                    final_mask = binary_mask

                    # Convert to numpy
                    final_mask = final_mask.cpu().numpy().astype(np.uint8)

                    # Apply post-processing if enabled
                    if apply_postprocessing:
                        print(f"Applying post-processing to {filename}")
                        final_mask[0] = post_process_segmentation(
                            final_mask[0],
                            closing_kernel_size=closing_kernel_size,
                            size_threshold_ratio=size_threshold_ratio
                        )

                    # Save prediction (keeping all channels)
                    output_path = output_dir / f"{filename}_pred.nii.gz"

                    # Create NIfTI with basic affine and header
                    # We need to transpose to match NIfTI convention [W, H, C]
                    nib_img = nib.Nifti1Image(
                        final_mask[0].transpose(1, 2, 0),
                        affine=affine,
                        header=header
                    )
                    nib.save(nib_img, output_path)
                    print(f"Saved prediction to {output_path}")

                    # Get the numpy image for visualization
                    # Fix: Define image variable before using it in create_channel_heatmaps
                    numpy_image = images[0, 0].cpu().numpy()

                    # Create and save visualization using the post-processed mask
                    vis_path = vis_dir / f"{filename}_visualization.png"
                    create_colored_overlay(
                        numpy_image,
                        final_mask[0].astype(np.float32),  # Use final processed mask for visualization
                        vis_path,
                        use_probs=False  # Since it's already binary after post-processing
                    )
                    print(f"Saved visualization to {vis_path}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()


def main():
    print("Starting DANN inference with improved post-processing")
    run_inference(
        # Update this path to point to your DANN model
        model_path=str(Path(__file__).parent.parent / "models" / "best_model.pt"),
        data_dir=str(Path(__file__).parent.parent / "test_data"),
        output_dir=str(Path(__file__).parent.parent / "predictions"),
        threshold=0.5,
        img_extension='.png',
        # Post-processing parameters
        apply_postprocessing=True,
        closing_kernel_size=3,  # Size of disk kernel for morphological closing
        size_threshold_ratio=0.1  # Keep components larger than 10% of the largest component
    )


if __name__ == "__main__":
    main()