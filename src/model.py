import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import ResidualUnit
from monai.networks.layers.factories import Conv
# --- Import traditional losses ---
from monai.losses import DiceLoss
from torch.nn import CrossEntropyLoss
from gradientReversalFunction import GradientReversal
import numpy as np


class DomainAdversarialModel(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 12,
            features_per_stage: list = [32, 64, 128, 256, 320, 320],
            n_stages: int = 6,
            kernel_size: int = 3,
            strides: list = [1, 2, 2, 2, 2, 2],
            n_conv_per_stage: int = 2,
            # --- Loss parameters ---
            use_combo_loss: bool = False,
            loss_weights: dict = None,
            channel_weights: list = None,
            dice_loss_weight: float = 0.6,  # Balanced between both implementations
            ce_loss_weight: float = 0.4,
            domain_loss_weight: float = 0.1,
            skip_domain_loss_weight: float = 0.05,
            alpha: float = 0.0,
            # --- Architecture control ---
            use_lightweight_skip_classifiers: bool = True,
            use_shared_grl: bool = True,  # Key improvement from first implementation
            skip_classifier_dropout: float = 0.1  # Light regularization
    ):
        super().__init__()

        # Main segmentation network with skip transformers
        self.network = SegmentationModel(
            in_channels=in_channels,
            out_channels=out_channels,
            features_per_stage=features_per_stage[:n_stages],
            kernel_size=kernel_size,
            strides=strides[:n_stages],
            n_conv_per_stage=n_conv_per_stage
        )

        # Get bottleneck and skip features dimensions
        bottleneck_features = features_per_stage[n_stages - 1]

        # FIXED: Skip features dimensions should match the decoder features, not encoder features
        # The segmentation skip features come from the decoder path, not encoder
        # These correspond to the reversed features used in decoder
        decoder_features = list(reversed(features_per_stage[:n_stages - 1]))

        print(f"Bottleneck features: {bottleneck_features}")
        print(f"Decoder skip features dimensions: {decoder_features}")

        # BEST FROM FIRST: Single shared GRL for unified adversarial training
        self.use_shared_grl = use_shared_grl
        if use_shared_grl:
            self.grl = GradientReversal(alpha=alpha)
            print("Using shared GRL for unified adversarial training")
        else:
            # Fallback to separate GRLs if needed
            self.grl = GradientReversal(alpha=alpha)
            self.skip_grls = nn.ModuleList([
                GradientReversal(alpha=alpha) for _ in decoder_features
            ])
            print("Using separate GRLs for each classifier")

        # BEST FROM SECOND: Enhanced bottleneck domain classifier with regularization
        self.domain_classifier = BottleneckDomainClassifier(
            bottleneck_features,
            dropout_rate=0.2
        )

        # FIXED: Create skip classifiers with correct decoder feature dimensions
        self.skip_classifiers = nn.ModuleList()
        self.use_lightweight_skip_classifiers = use_lightweight_skip_classifiers

        for features in decoder_features:
            if use_lightweight_skip_classifiers:
                # Lightweight but with slight regularization
                classifier = SkipDomainClassifier(
                    features,
                    dropout_rate=skip_classifier_dropout
                )
            else:
                # Full classifier for complex domains
                classifier = BottleneckDomainClassifier(
                    features,
                    dropout_rate=skip_classifier_dropout
                )
            self.skip_classifiers.append(classifier)

        print(
            f"Created {len(self.skip_classifiers)} {'lightweight' if use_lightweight_skip_classifiers else 'enhanced'} skip classifiers")
        print(f"Skip classifier input dimensions: {decoder_features}")

        # Store loss configuration
        self.use_combo_loss = use_combo_loss
        self.dice_loss_weight = dice_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.domain_loss_weight = domain_loss_weight
        self.skip_domain_loss_weight = skip_domain_loss_weight

        # --- Initialize custom ComboLoss if specified ---
        if self.use_combo_loss and loss_weights is not None:
            try:
                from losses import ComboLoss
                self.combo_loss = ComboLoss(
                    weights=loss_weights,
                    channel_weights=channel_weights or [1.0] * out_channels
                )
                print("Using ComboLoss with clDice and focal loss")
            except ImportError:
                print("Warning: ComboLoss not available, falling back to standard losses")
                self.use_combo_loss = False

        # --- Initialize loss functions ---
        self.dice_loss = DiceLoss(
            include_background=True,
            batch=True,
            to_onehot_y=False,
            reduction='mean'
        )

        self.ce_loss = CrossEntropyLoss(reduction='mean')
        self.domain_loss = nn.BCEWithLogitsLoss(reduction='mean')

        # For validation monitoring
        self.dice_loss_monitor = DiceLoss(
            include_background=True,
            batch=False,
            to_onehot_y=False,
            reduction='mean'
        )

        # BEST FROM SECOND: Adaptive loss weighting
        self.adaptive_weights = True
        self.bottleneck_weight = 0.7
        self.skip_weight = 0.3

    def forward(self, x, domain_label=None, return_features=False):
        """Enhanced forward pass with corrected domain adaptation"""

        # Prepare domain adaptation info for segmentation network
        domain_adaptation_info = None
        if domain_label is not None or self.training:
            domain_adaptation_info = {
                'grl_function': self.grl,
                'is_domain_training': (domain_label is not None and self.training)
            }

        # Get segmentation output and features
        # Now segmentation_skip_features are the ones actually used for segmentation
        seg_output, features, segmentation_skip_features = self.network(x, domain_adaptation_info)

        # Pure inference mode
        if domain_label is None and not self.training:
            return seg_output

        # Extract bottleneck features for domain adaptation
        bottleneck_features = features["bottleneck"]

        # Apply domain adaptation to bottleneck (unchanged)
        if self.use_shared_grl:
            # Use single GRL for all domain classifiers
            reversed_bottleneck = self.grl(bottleneck_features)
            domain_output = self.domain_classifier(reversed_bottleneck)

            # FIXED: Apply domain adaptation to the features actually used for segmentation
            skip_domain_outputs = []
            for seg_skip_feat, skip_clf in zip(segmentation_skip_features, self.skip_classifiers):
                reversed_skip = self.grl(seg_skip_feat)  # Same GRL instance, but correct features
                skip_domain_out = skip_clf(reversed_skip)
                skip_domain_outputs.append(skip_domain_out)
        else:
            # Separate GRLs if needed
            reversed_bottleneck = self.grl(bottleneck_features)
            domain_output = self.domain_classifier(reversed_bottleneck)

            skip_domain_outputs = []
            for seg_skip_feat, skip_grl, skip_clf in zip(
                    segmentation_skip_features, self.skip_grls, self.skip_classifiers
            ):
                reversed_skip = skip_grl(seg_skip_feat)  # Apply to segmentation features
                skip_domain_out = skip_clf(reversed_skip)
                skip_domain_outputs.append(skip_domain_out)

        if return_features:
            return seg_output, domain_output, skip_domain_outputs, features, segmentation_skip_features
        else:
            return seg_output, domain_output, skip_domain_outputs

    def update_alpha(self, alpha):
        """Update GRL alpha parameter(s)"""
        self.grl.alpha = alpha
        if not self.use_shared_grl:
            for skip_grl in self.skip_grls:
                skip_grl.alpha = alpha

    def update_loss_weights(self, epoch, max_epochs):
        """BEST FROM SECOND: Adaptive loss weighting during training"""
        if self.adaptive_weights:
            # Gradually shift weight from bottleneck to skip connections
            progress = epoch / max_epochs
            if progress < 0.3:
                self.bottleneck_weight = 0.8
                self.skip_weight = 0.2
            elif progress < 0.7:
                self.bottleneck_weight = 0.6
                self.skip_weight = 0.4
            else:
                self.bottleneck_weight = 0.5
                self.skip_weight = 0.5

    def training_step(self, batch, domain_batch=None):
        """Enhanced training step combining best practices"""
        images = batch["image"]
        labels = batch["label"]

        if domain_batch is None:
            # Source-only training
            seg_outputs_logits = self(images)

            # Calculate segmentation loss
            if self.use_combo_loss and hasattr(self, 'combo_loss'):
                seg_loss = self.combo_loss(seg_outputs_logits, labels)
            else:
                seg_outputs_sigmoid = torch.sigmoid(seg_outputs_logits)
                dice_loss_val = self.dice_loss(seg_outputs_sigmoid, labels)
                ce_loss_val = F.binary_cross_entropy_with_logits(seg_outputs_logits, labels)
                seg_loss = self.dice_loss_weight * dice_loss_val + self.ce_loss_weight * ce_loss_val

            return seg_loss, None, seg_outputs_logits

        # Domain adaptation training
        target_images = domain_batch["image"]
        source_size = images.size(0)
        target_size = target_images.size(0)

        # Create domain labels
        source_domain_labels = torch.zeros(source_size, 1, device=images.device)
        target_domain_labels = torch.ones(target_size, 1, device=target_images.device)

        # Forward passes
        seg_outputs_logits, source_domain_outputs, source_skip_outputs = self(
            images, domain_label=source_domain_labels
        )
        _, target_domain_outputs, target_skip_outputs = self(
            target_images, domain_label=target_domain_labels
        )

        # Calculate segmentation loss
        if self.use_combo_loss and hasattr(self, 'combo_loss'):
            seg_loss = self.combo_loss(seg_outputs_logits, labels)
        else:
            seg_outputs_sigmoid = torch.sigmoid(seg_outputs_logits)
            dice_loss_val = self.dice_loss(seg_outputs_sigmoid, labels)
            ce_loss_val = F.binary_cross_entropy_with_logits(seg_outputs_logits, labels)
            seg_loss = self.dice_loss_weight * dice_loss_val + self.ce_loss_weight * ce_loss_val

        # BEST FROM BOTH: Stable domain loss calculation
        # Bottleneck domain loss
        source_domain_stable = torch.clamp(source_domain_outputs, min=-15, max=15)
        target_domain_stable = torch.clamp(target_domain_outputs, min=-15, max=15)

        source_domain_loss = self.domain_loss(source_domain_stable, source_domain_labels)
        target_domain_loss = self.domain_loss(target_domain_stable, target_domain_labels)
        bottleneck_domain_loss = (source_domain_loss + target_domain_loss) / 2

        # Skip connection domain losses
        skip_domain_loss = 0
        skip_losses_detailed = []

        for src_skip_out, tgt_skip_out in zip(source_skip_outputs, target_skip_outputs):
            src_skip_stable = torch.clamp(src_skip_out, min=-15, max=15)
            tgt_skip_stable = torch.clamp(tgt_skip_out, min=-15, max=15)

            src_skip_loss = self.domain_loss(src_skip_stable, source_domain_labels)
            tgt_skip_loss = self.domain_loss(tgt_skip_stable, target_domain_labels)

            skip_loss = (src_skip_loss + tgt_skip_loss) / 2
            skip_domain_loss += skip_loss
            skip_losses_detailed.append(skip_loss.item())

        # Average skip domain loss
        if len(source_skip_outputs) > 0:
            skip_domain_loss = skip_domain_loss / len(source_skip_outputs)

        # BEST FROM FIRST: Unified loss combination with adaptive weighting
        if self.use_shared_grl:
            # Unified adversarial loss with adaptive weighting
            combined_domain_loss = (
                    self.bottleneck_weight * bottleneck_domain_loss +
                    self.skip_weight * skip_domain_loss
            )
        else:
            # Separate loss weighting
            combined_domain_loss = bottleneck_domain_loss + self.skip_domain_loss_weight * skip_domain_loss

        # Store detailed metrics for monitoring
        self._store_training_metrics(
            source_domain_stable, target_domain_stable,
            source_skip_outputs, target_skip_outputs,
            source_domain_labels, target_domain_labels,
            source_size, target_size,
            source_domain_loss, target_domain_loss,
            skip_losses_detailed
        )

        # Total loss
        total_loss = seg_loss + self.domain_loss_weight * combined_domain_loss

        return total_loss, combined_domain_loss, seg_outputs_logits

    def _store_training_metrics(self, source_domain_stable, target_domain_stable,
                                source_skip_outputs, target_skip_outputs,
                                source_domain_labels, target_domain_labels,
                                source_size, target_size,
                                source_domain_loss, target_domain_loss,
                                skip_losses_detailed):
        """Store detailed metrics for monitoring"""
        # Bottleneck metrics
        self.bottleneck_source_loss = source_domain_loss.item()
        self.bottleneck_target_loss = target_domain_loss.item()

        # Skip metrics
        self.skip_losses = skip_losses_detailed
        self.skip_source_loss = np.mean([l for l in skip_losses_detailed]) if skip_losses_detailed else 0
        self.skip_target_loss = self.skip_source_loss  # Approximation for monitoring

        # Calculate accuracies
        with torch.no_grad():
            # Bottleneck accuracy
            source_pred = (torch.sigmoid(source_domain_stable) < 0.5).float()
            target_pred = (torch.sigmoid(target_domain_stable) >= 0.5).float()

            source_acc = (source_pred == source_domain_labels).float().mean().item()
            target_acc = (target_pred == target_domain_labels).float().mean().item()

            self.source_domain_accuracy = source_acc
            self.target_domain_accuracy = target_acc
            self.domain_accuracy = (source_acc * source_size + target_acc * target_size) / (source_size + target_size)

            # Skip accuracies
            skip_source_accs = []
            skip_target_accs = []

            for src_skip_out, tgt_skip_out in zip(source_skip_outputs, target_skip_outputs):
                src_skip_pred = (torch.sigmoid(src_skip_out) < 0.5).float()
                tgt_skip_pred = (torch.sigmoid(tgt_skip_out) >= 0.5).float()

                src_acc = (src_skip_pred == source_domain_labels).float().mean().item()
                tgt_acc = (tgt_skip_pred == target_domain_labels).float().mean().item()

                skip_source_accs.append(src_acc)
                skip_target_accs.append(tgt_acc)

            self.skip_source_accuracy = np.mean(skip_source_accs) if skip_source_accs else 0
            self.skip_target_accuracy = np.mean(skip_target_accs) if skip_target_accs else 0
            self.skip_domain_accuracy = (
                    (self.skip_source_accuracy * source_size + self.skip_target_accuracy * target_size) /
                    (source_size + target_size)
            )


class BottleneckDomainClassifier(nn.Module):
    """Enhanced domain classifier with controlled regularization"""

    def __init__(self, in_channels, dropout_rate=0.2):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Lighter dropout for deeper layers
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SkipDomainClassifier(nn.Module):
    """Lightweight skip classifier optimized for high learning rates"""

    def __init__(self, in_channels, dropout_rate=0.1):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Lightweight but with slight regularization
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 128),  # Larger than original for better capacity
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SegmentationModel(nn.Module):
    """Enhanced segmentation model with MONAI ResidualUnit and skip transformers"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            features_per_stage: list,
            kernel_size: int = 3,
            strides: list = None,
            n_conv_per_stage: int = 2
    ):
        super().__init__()

        if strides is None:
            strides = [1] + [2] * (len(features_per_stage) - 1)

        assert len(features_per_stage) == len(strides)

        self.n_stages = len(features_per_stage)
        self.features_per_stage = features_per_stage

        # Create encoder path using MONAI ResidualUnit
        self.encoder = nn.ModuleList()
        current_channels = in_channels

        for stage, (features, stride) in enumerate(zip(features_per_stage, strides)):
            # Use dilation for deeper stages to capture long-range dependencies
            dilation = 1 if stage < 2 else 2

            block = self._create_residual_stage(
                in_channels=current_channels,
                out_channels=features,
                kernel_size=kernel_size,
                stride=stride,
                n_convs=n_conv_per_stage,
                dilation=dilation
            )
            self.encoder.append(block)
            current_channels = features

        # Create decoder path
        self.decoder = nn.ModuleList()
        self.upsampling = nn.ModuleList()

        # Reverse feature list for decoder (excluding bottleneck)
        reversed_features = list(reversed(features_per_stage[:-1]))
        current_channels = features_per_stage[-1]  # Start from bottleneck

        for features in reversed_features:
            # Upsampling
            self.upsampling.append(
                nn.ConvTranspose2d(
                    current_channels,
                    features,
                    kernel_size=2,
                    stride=2
                )
            )

            # Decoder block after concat
            block = self._create_residual_stage(
                in_channels=2 * features,  # Concatenated features
                out_channels=features,
                kernel_size=kernel_size,
                stride=1,
                n_convs=n_conv_per_stage,
                dilation=1
            )
            self.decoder.append(block)
            current_channels = features

        # Final 1x1 conv
        self.final_conv = Conv["conv", 2](
            in_channels=current_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        self.bottleneck_features = features_per_stage[-1]

        # Skip connection transformations for segmentation quality
        self.skip_transformers = nn.ModuleList()
        for feature_size in reversed_features:
            self.skip_transformers.append(SkipConnectionTransformer(feature_size))

    def _create_residual_stage(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            n_convs: int,
            dilation: int = 1
    ) -> nn.Sequential:
        """Create conv block using MONAI ResidualUnit"""
        layers = []
        current_channels = in_channels

        for i in range(n_convs):
            current_dilation = dilation if (i > 0 or stride == 1) else 1

            layers.append(ResidualUnit(
                spatial_dims=2,
                in_channels=current_channels,
                out_channels=out_channels,
                strides=stride if i == 0 else 1,
                kernel_size=kernel_size,
                subunits=1,
                norm="INSTANCE",
                act="LEAKYRELU",
                dilation=current_dilation
            ))
            current_channels = out_channels
            stride = 1

        return nn.Sequential(*layers)

    def forward(self, x, domain_adaptation_info=None):
        """
        Forward pass with optional domain adaptation applied to segmentation features

        Args:
            x: Input tensor
            domain_adaptation_info: Dict containing GRL function and domain training flag
                                  {'grl_function': grl, 'is_domain_training': bool}
        """
        features = {}
        skips = []

        # Encoder path
        for i, block in enumerate(self.encoder[:-1]):  # Exclude bottleneck
            x = block(x)
            skips.append(x)

        # Bottleneck
        x = self.encoder[-1](x)
        features["bottleneck"] = x

        # Store original skip features for backward compatibility
        original_skip_features = skips.copy()

        # Decoder path with skip transformations for segmentation
        skips = list(reversed(skips))

        # Store the features actually used for segmentation (for domain adaptation)
        segmentation_skip_features = []

        for i, (up, block, skip, skip_transformer) in enumerate(zip(
                self.upsampling, self.decoder, skips, self.skip_transformers)):
            x = up(x)

            # Handle dimension mismatch by center cropping
            if x.shape[2:] != skip.shape[2:]:
                min_h = min(x.size(2), skip.size(2))
                min_w = min(x.size(3), skip.size(3))

                if x.size(2) > min_h or x.size(3) > min_w:
                    x_h_diff = x.size(2) - min_h
                    x_w_diff = x.size(3) - min_w
                    x = x[:, :,
                        x_h_diff // 2:x_h_diff // 2 + min_h,
                        x_w_diff // 2:x_w_diff // 2 + min_w]

                if skip.size(2) > min_h or skip.size(3) > min_w:
                    s_h_diff = skip.size(2) - min_h
                    s_w_diff = skip.size(3) - min_w
                    skip = skip[:, :,
                           s_h_diff // 2:s_h_diff // 2 + min_h,
                           s_w_diff // 2:s_w_diff // 2 + min_w]

            # Apply transformation to skip connection for segmentation
            transformed_skip = skip_transformer(skip)

            # FIXED: Store the transformed features for domain adaptation
            # These are the features actually used for segmentation
            segmentation_skip_features.append(transformed_skip)

            # OPTIONAL: Apply domain adaptation to features used for segmentation
            if (domain_adaptation_info is not None and
                    domain_adaptation_info.get('is_domain_training', False) and
                    domain_adaptation_info.get('grl_function') is not None):

                # Apply GRL to the features actually used for segmentation
                grl = domain_adaptation_info['grl_function']
                domain_adapted_skip = grl(transformed_skip)
                x = torch.cat([x, domain_adapted_skip], dim=1)
            else:
                x = torch.cat([x, transformed_skip], dim=1)

            x = block(x)

        # Final output
        output = self.final_conv(x)

        # Return: segmentation output, bottleneck features, and CORRECTED skip features
        # Now returns the features actually used for segmentation
        return output, features, segmentation_skip_features

class SkipConnectionTransformer(nn.Module):
    """Enhanced skip connection transformer for better segmentation"""

    def __init__(self, channels):
        super().__init__()

        # 1x1 convolution for feature transformation
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.InstanceNorm2d(channels),
            nn.GELU()
        )

        # Channel attention mechanism
        self.channel_attention = ChannelAttention(channels)

    def forward(self, x):
        # Apply transformation
        x = self.transform(x)

        # Apply channel attention
        attention = self.channel_attention(x)
        x = attention * x

        return x


class ChannelAttention(nn.Module):
    """Channel attention module for feature enhancement"""

    def __init__(self, channels, reduction=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y