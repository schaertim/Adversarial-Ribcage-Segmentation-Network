from pathlib import Path
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    Lambdad,
    SpatialPadd,
    RandZoomd,
    CenterSpatialCropd,
    RandSpatialCropd,
    RandRotated,
    RandGaussianNoised,
    RandFlipd,
    RandAdjustContrastd,
    NormalizeIntensityd,
    RandShiftIntensityd
)
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from monai.transforms import MapTransform
import torch

class LoadPNGd:
    """Load PNG images with PIL with proper bit depth handling."""

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue
            img_path = d[key]
            try:
                img = Image.open(img_path)
                img_np = np.array(img)
                actual_max = np.max(img_np)
                if actual_max > 255:
                    max_val = 65535
                else:
                    max_val = 255
                img_np = img_np.astype(np.float32) / max_val
                d[key] = img_np
            except Exception as e:
                print(f"ERROR loading {key} from {img_path}: {e}")
                import traceback

                traceback.print_exc()
                raise
        return d

class SourceDomainDataset:
    def __init__(
            self,
            data_dir: str,
            is_train: bool = True,
            inference_mode: bool = False,
            augmentation_params: dict = None,
            patch_size: tuple = None,
            debug_augmentations: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.inference_mode = inference_mode
        self.augmentation_params = augmentation_params or {}
        self.patch_size = patch_size
        self.debug_augmentations = debug_augmentations
        self.debug_dir = os.path.join(os.path.dirname(os.path.dirname(str(data_dir))), "augmentation_debug")

        if self.debug_augmentations:
            print(f"Augmentation debugging enabled. Images will be saved to: {self.debug_dir}")
            os.makedirs(self.debug_dir, exist_ok=True)

        # Create data dicts
        self.data_dicts = self._split_train_val_dicts()
        print(f"Found {len(self.data_dicts)} images for {'training' if is_train else 'validation'}")

        # Set up transforms
        self.transforms = self._get_transforms()

        # Create dataset
        self.dataset = CacheDataset(
            data=self.data_dicts,
            transform=self.transforms,
            cache_rate=1.0 if not self.debug_augmentations else 0.0
        )

    def _get_all_data_dicts(self):
        """
        Get all data pairs from the directory without train/val split.
        Returns a list of dictionaries with all image-label pairs.
        """
        data_dir = self.data_dir
        image_dir = data_dir / "images" if (data_dir / "images").exists() else data_dir

        all_data_dicts = []
        for image_path in sorted(image_dir.glob("*.png")):
            file_stem = image_path.stem
            label_dir = data_dir / "labels" if (data_dir / "labels").exists() else data_dir
            label_path = label_dir / f"{file_stem}.nii.gz"

            if label_path.exists():
                all_data_dicts.append({
                    "image": str(image_path),
                    "label": str(label_path)
                })

        return all_data_dicts

    def _split_train_val_dicts(self):
        """
        Split the dataset into train and validation sets.
        """
        all_data_dicts = self._get_all_data_dicts()

        indices = list(range(len(all_data_dicts)))
        train_indices, val_indices = train_test_split(indices, test_size=0.2)

        if self.is_train:
            data_dicts = [all_data_dicts[i] for i in train_indices]
        else:
            data_dicts = [all_data_dicts[i] for i in val_indices]

        return data_dicts

    def _fix_image_shape(self, image):
        """Ensure grayscale image has the channel dimension at the front (C, H, W)."""
        if image.ndim == 3:  # Input is H, W, C
            if image.shape[2] > 1:  # Convert RGB/RGBA to grayscale
                image = np.mean(image, axis=2).astype(np.float32)
            else:  # Input is H, W, 1
                image = image[..., 0]  # Remove last dim -> H, W
        # Now image is H, W
        if image.ndim == 2:
            image = image[np.newaxis, ...]  # Add channel dim -> 1, H, W
        return image

    def _fix_label_shape(self, label):
        """Move channels to first dimension, convert to binary, and correct orientation."""
        # Check if label is already (C, H, W)
        if (
            label.ndim == 3
            and label.shape[0] < label.shape[1]
            and label.shape[0] < label.shape[2]
        ):
            return (label > 0).astype(np.float32)

        # Handle H, W, C (multi-channel NIfTI loaded by nibabel?)
        if len(label.shape) == 3 and label.shape[2] > 1:
            if label.shape[2] <= 24:  # Heuristic for multi-channel label
                num_channels = label.shape[2]
                # Output shape C, H_rotated, W_rotated -> C, W, H
                rotated_label = np.zeros(
                    (num_channels, label.shape[1], label.shape[0]),
                    dtype=np.float32,
                )
                for c in range(num_channels):
                    channel = label[:, :, c]
                    rotated = np.rot90(
                        channel, k=-1
                    )  # Rotate 90 deg clockwise -> W, H
                    rotated = np.fliplr(rotated)  # Flip horizontally
                    rotated_label[c] = rotated
                label = rotated_label  # Shape C, W, H
            else:  # Treat as H, W, C but only use first channel
                label = label[:, :, 0]  # -> H, W
                label = np.rot90(label, k=-1)  # -> W, H
                label = np.fliplr(label)  # -> W, H
                label = label[np.newaxis, ...]  # -> 1, W, H
        # Handle H, W, 1
        elif len(label.shape) == 3 and label.shape[2] == 1:
            label = label[..., 0]  # -> H, W
            label = np.rot90(label, k=-1)  # -> W, H
            label = np.fliplr(label)  # -> W, H
            label = label[np.newaxis, ...]  # -> 1, W, H
        # Handle H, W
        elif len(label.shape) == 2:
            label = np.rot90(label, k=-1)  # -> W, H
            label = np.fliplr(label)  # -> W, H
            label = label[np.newaxis, ...]  # -> 1, W, H
        else:
            print(f"Warning: Unexpected label shape {label.shape}, attempting to use as is.")

        # Ensure binary values and float type
        return (label > 0).astype(np.float32)


    def _get_transforms(self):
        """Get data transforms with only zoom augmentation."""
        keys = ["image", "label"]

        base_transforms = [
            LoadPNGd(keys=["image"]),
            LoadImaged(keys=["label"]),
            Lambdad(keys=["image"], func=self._fix_image_shape),
            Lambdad(keys=["label"], func=self._fix_label_shape),

            # Spatial operations
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=self.patch_size,
                mode=("constant", "constant"),
                constant_values=(0, 0)
            ),

            # Normalize intensity (Z-score)
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=True
            ),

            # Random crop during training or center crop during validation
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=self.patch_size,
                random_size=False
            ) if self.is_train else CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=self.patch_size
            ),
        ]
        if self.is_train and self.augmentation_params:
            # Enhanced augmentation with rotation and elastic deformation
            augmentations = [
                # Zoom augmentation with more aggressive range
                RandZoomd(
                    keys=["image", "label"],
                    min_zoom=self.augmentation_params.get('zoom_range', (0.8, 1.2))[0],  # More aggressive zoom
                    max_zoom=self.augmentation_params.get('zoom_range', (0.8, 1.2))[1],
                    prob=self.augmentation_params.get('p_zoom', 0.5),  # Get probability from config
                    mode=("bilinear", "nearest"),
                    align_corners=(True, None)
                ),

                # Random rotation (±15 degrees)
                RandRotated(
                    keys=["image", "label"],
                    range_x=np.radians(self.augmentation_params.get('rotate_range', (-15, 15))),
                    prob=self.augmentation_params.get('p_rotate', 0.5),
                    mode=("bilinear", "nearest"),
                    padding_mode=("zeros", "zeros"),
                    align_corners=(True, None)
                ),

                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=1,
                    prob=self.augmentation_params.get('p_flip', 0.5)
                ),

                # Random Gaussian noise
                RandGaussianNoised(
                    keys=["image"],
                    std=self.augmentation_params.get('gaussian_noise_std', 0.01),
                    prob=self.augmentation_params.get('p_gaussian_noise', 0.15)
                ),

                # Random contrast adjustment
                RandAdjustContrastd(
                    keys=["image"],
                    gamma=(0.25, 3.0),
                    prob=self.augmentation_params.get('p_adjust_contrast', 0.5)
                ),

                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.3,  # Shift intensity values by up to ±30%
                    prob=self.augmentation_params.get('p_shift_intensity', 0.5)
                )
            ]

            # Add augmentations to base transforms
            base_transforms.extend(augmentations)

        # Always add type conversion, whether training or validation
        base_transforms.append(EnsureTyped(keys=["image", "label"]))

        # Final visualization after all transformations
        if self.debug_augmentations:
            base_transforms.append(
                SaveDebugImageD(keys=["image", "label"], debug_dir=self.debug_dir,
                                prefix="11_final_result", max_samples=20)
            )

        # Now use base_transforms directly
        return Compose(base_transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class SaveDebugImageD(MapTransform):
    """Custom transform to save overlay visualization of image and segmentation."""

    def __init__(self, keys, debug_dir, prefix, max_samples=1):
        super().__init__(keys)
        self.debug_dir = debug_dir
        self.prefix = prefix
        self.sample_count = 0
        self.max_samples = max_samples

        # Create directory if it doesn't exist
        os.makedirs(self.debug_dir, exist_ok=True)

        # Verify we have both image and label keys
        self.has_both = len(keys) >= 2 and "image" in keys and "label" in keys

    def __call__(self, data):
        if self.sample_count >= self.max_samples:
            return data

        # Simplified case ID extraction
        case_id = ""
        if "image_meta_dict" in data and "filename_or_obj" in data["image_meta_dict"]:
            file_path = data["image_meta_dict"]["filename_or_obj"]
            if isinstance(file_path, str):
                case_id = os.path.basename(file_path).split('.')[0] + "_"

        # Get image data
        image = data.get("image", None)
        label = data.get("label", None)

        # Convert tensors to numpy if needed
        if image is not None and torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        if label is not None and torch.is_tensor(label):
            label = label.detach().cpu().numpy()

        # Handle channel-first format for image
        if image is not None and image.ndim == 3 and image.shape[0] <= 3:
            image = image[0]  # Use first channel

        # If both image and label are present, show both
        if image is not None and label is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Left: just the image
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title(f"{self.prefix} - Image Only")
            axes[0].axis('off')

            # Right: image with overlay
            axes[1].imshow(image, cmap='gray')

            if label.ndim == 3:
                n_channels = label.shape[0]
                cmap = plt.cm.plasma
                for i in range(n_channels - 1, -1, -1):
                    mask = label[i] > 0
                    if np.any(mask):
                        color = cmap(i / max(n_channels - 1, 1))
                        colored_mask = np.zeros((*image.shape, 4))
                        colored_mask[mask, 0] = color[0]
                        colored_mask[mask, 1] = color[1]
                        colored_mask[mask, 2] = color[2]
                        colored_mask[mask, 3] = 0.6
                        axes[1].imshow(colored_mask)
            else:
                print(f"Warning: Expected multi-channel label with shape (C,H,W), got {label.shape}. Skipping overlay.")

            axes[1].set_title(f"{self.prefix} - Image with Segmentation Overlay")
            axes[1].axis('off')

            filename = os.path.join(self.debug_dir, f"{self.prefix}_{case_id}overlay_sample_{self.sample_count}.png")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close(fig)

        # If only image is present, show just the image
        elif image is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image, cmap='gray')
            ax.set_title(f"{self.prefix} - Image Only")
            ax.axis('off')
            filename = os.path.join(self.debug_dir, f"{self.prefix}_{case_id}image_sample_{self.sample_count}.png")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close(fig)

        else:
            print(f"Warning: No image found for visualization. Skipping sample {self.sample_count}.")

        self.sample_count += 1
        return data


class TargetDomainDataset:
    """Dataset for target domain images (EOS X-rays) without segmentation labels"""

    def __init__(
            self,
            data_dir: str,
            is_train: bool = True,
            patch_size: tuple = None,
            augmentation_params: dict = None,
            debug_augmentations: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.patch_size = patch_size
        self.augmentation_params = augmentation_params or {}
        self.debug_augmentations = debug_augmentations
        self.debug_dir = os.path.join(os.path.dirname(os.path.dirname(str(data_dir))), "augmentation_debug")

        # Get all target domain image files
        self.image_files = self._get_image_files()
        print(f"Found {len(self.image_files)} target domain images for {'training' if is_train else 'validation'}")

        # Create data dictionaries for MONAI
        self.data_dicts = [{"image": str(img_path)} for img_path in self.image_files]

        # Set up transforms
        self.transforms = self._get_transforms()

        # Create dataset
        self.dataset = CacheDataset(
            data=self.data_dicts,
            transform=self.transforms,
            cache_rate=1.0 if not self.debug_augmentations else 0.0
        )

    def _get_image_files(self):
        """Get all target domain image files"""
        image_dir = self.data_dir / "images" if (self.data_dir / "images").exists() else self.data_dir
        return list(sorted(image_dir.glob("*.png")))

    def _fix_image_shape(self, image):
        """Ensure grayscale image has the channel dimension at the front (C, H, W)."""
        if image.ndim == 3:  # Input is H, W, C
            if image.shape[2] > 1:  # Convert RGB/RGBA to grayscale
                image = np.mean(image, axis=2).astype(np.float32)
            else:  # Input is H, W, 1
                image = image[..., 0]  # Remove last dim -> H, W
        # Now image is H, W
        if image.ndim == 2:
            image = image[np.newaxis, ...]  # Add channel dim -> 1, H, W
        return image

    def _get_transforms(self):
        """Set up transforms for target domain images"""

        base_transforms = [
            # Load images
            LoadPNGd(keys=["image"]),
            Lambdad(keys=["image"], func=self._fix_image_shape),
            # Spatial operations
            SpatialPadd(
                keys=["image"],
                spatial_size=self.patch_size,
                mode="constant",
                constant_values=0
            ),

            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=True
            ),

            # Random crop during training or center crop during validation
            RandSpatialCropd(
                keys=["image"],
                roi_size=self.patch_size,
                random_size=False
            ) if self.is_train else CenterSpatialCropd(
                keys=["image"],
                roi_size=self.patch_size
            ),

        ]

        # Add augmentations for training
        if self.is_train and self.augmentation_params:
            augmentations = [
                # Zoom augmentation
                RandZoomd(
                    keys=["image"],
                    min_zoom=self.augmentation_params.get('zoom_range', (0.8, 1.2))[0],
                    max_zoom=self.augmentation_params.get('zoom_range', (0.8, 1.2))[1],
                    prob=self.augmentation_params.get('p_zoom', 0.5),
                    mode="bilinear",
                    align_corners=True
                ),

                # Random rotation
                RandRotated(
                    keys=["image"],
                    range_x=np.radians(self.augmentation_params.get('rotate_range', (-10, 10))),
                    prob=self.augmentation_params.get('p_rotate', 0.5),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True
                ),

                RandFlipd(
                    keys=["image"],
                    spatial_axis=1,
                    prob=self.augmentation_params.get('p_flip', 0.5)
                ),

                # Random contrast adjustment
                RandAdjustContrastd(
                    keys=["image"],
                    gamma=(0.25, 3.0),
                    prob=self.augmentation_params.get('p_adjust_contrast', 0.5)
                ),

                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.3,  # Shift intensity values by up to ±30%
                    prob=self.augmentation_params.get('p_shift_intensity', 0.5)
                ),
            ]

            base_transforms.extend(augmentations)

        # Always add type conversion
        base_transforms.append(EnsureTyped(keys=["image"]))

        return Compose(base_transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]