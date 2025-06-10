import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.data import list_data_collate
from pathlib import Path
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch.nn.functional as F

# Import from modified files
from dataset import SourceDomainDataset, TargetDomainDataset
from model import DomainAdversarialModel


def plot_tsne(source_features, target_features):
    """Calculate t-SNE embedding and return coordinates and labels"""
    if source_features.shape[0] == 0 or target_features.shape[0] == 0:
        print("Warning: Not enough features for t-SNE calculation.")
        return None, None

    X = np.concatenate([source_features, target_features], axis=0)
    y = np.array([0] * len(source_features) + [1] * len(target_features))

    n_samples = X.shape[0]
    perplexity_val = min(30.0, max(5.0, n_samples / 4.0))

    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, max_iter=300)
        X_embedded = tsne.fit_transform(X)
        return X_embedded, y
    except Exception as e:
        print(f"Error during t-SNE calculation: {e}")
        return None, None


def collect_bottleneck_features(model, source_loader, target_loader, device, num_batches=5):
    """Collect bottleneck features for t-SNE visualization"""
    model.eval()
    source_features = []
    target_features = []

    with torch.no_grad():
        # Source domain
        for i, batch in enumerate(source_loader):
            if i >= num_batches:
                break
            images = batch["image"].to(device)
            _, features, _ = model.network(images)
            bottleneck = features["bottleneck"]
            pooled = bottleneck.mean(dim=[2, 3])
            source_features.append(pooled.cpu().numpy())

        # Target domain
        if target_loader:
            for i, batch in enumerate(target_loader):
                if i >= num_batches:
                    break
                images = batch["image"].to(device)
                _, features, _ = model.network(images)
                bottleneck = features["bottleneck"]
                pooled = bottleneck.mean(dim=[2, 3])
                target_features.append(pooled.cpu().numpy())

    source_features = np.concatenate(source_features, axis=0) if source_features else np.array([]).reshape(0,
                                                                                                           model.network.bottleneck_features)
    target_features = np.concatenate(target_features, axis=0) if target_features else np.array([]).reshape(0,
                                                                                                           model.network.bottleneck_features)

    return source_features, target_features


class DANNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create directories
        if self.config.model_path:
            Path(self.config.model_path).parent.mkdir(exist_ok=True, parents=True)

        self.figures_dir = os.path.join(os.path.dirname(self.config.model_path), 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        print(f"Progress plot will be saved to: {self.figures_dir}/training_progress.png")

        # Setup data and model
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()

        # Initialize tracking
        self.best_dice = float('-inf')
        self.training_losses = []
        self.validation_losses = []
        self.training_dices = []
        self.validation_dices = []
        self.domain_losses = []
        self.epochs = []
        self.domain_accuracies = []
        self.source_accuracies = []
        self.target_accuracies = []
        self.skip_domain_accuracies = []
        self.skip_source_accuracies = []
        self.skip_target_accuracies = []
        self.latest_tsne_data = None

        print("Enhanced DANN Trainer initialized!")

    def setup_model(self):
        """Setup enhanced model with optimal configuration"""
        self.model = DomainAdversarialModel(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            features_per_stage=self.config.features_per_stage[:self.config.n_stages],
            n_stages=self.config.n_stages,
            strides=self.config.strides[:self.config.n_stages],
            n_conv_per_stage=self.config.n_conv_per_stage,
            # BEST FROM BOTH: Balanced loss weights
            dice_loss_weight=getattr(self.config, 'dice_loss_weight', 0.6),
            ce_loss_weight=getattr(self.config, 'ce_loss_weight', 0.4),
            domain_loss_weight=getattr(self.config, 'domain_loss_weight', 0.1),
            skip_domain_loss_weight=getattr(self.config, 'skip_domain_loss_weight', 0.05),
            # BEST FROM FIRST: Unified GRL and lightweight classifiers
            use_shared_grl=True,
            use_lightweight_skip_classifiers=True,
            alpha=0.0
        ).to(self.device)

        self.max_dann_alpha = getattr(self.config, 'max_dann_alpha', 0.1)

    def setup_optimizer(self):
        """BEST FROM FIRST: Optimal learning rates for different components"""
        if getattr(self.config, 'optimizer_type', 'AdamW').lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=getattr(self.config, 'learning_rate', 0.01),
                momentum=getattr(self.config, 'momentum', 0.99),
                nesterov=getattr(self.config, 'nesterov', True),
                weight_decay=getattr(self.config, 'weight_decay', 3e-5)
            )
            print("Using SGD optimizer")
        else:
            # CRITICAL: Optimized learning rates
            base_lr = getattr(self.config, 'learning_rate', 0.0002)
            domain_lr = base_lr * 0.5  # Lower for main domain classifier
            skip_lr = base_lr * 20  # MUCH higher for skip classifiers (key insight)

            print(f"Learning rates - Seg: {base_lr:.1E}, Domain: {domain_lr:.1E}, Skip: {skip_lr:.1E}")

            param_groups = [
                {'params': self.model.network.parameters(), 'lr': base_lr},
                {'params': self.model.domain_classifier.parameters(), 'lr': domain_lr},
                {'params': [p for classifier in self.model.skip_classifiers for p in classifier.parameters()],
                 'lr': skip_lr}
            ]

            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=getattr(self.config, 'weight_decay', 3e-5),
                betas=(0.9, 0.999)
            )

        # Learning rate scheduler
        lr_scheduler = getattr(self.config, 'lr_scheduler', 'poly').lower()
        if lr_scheduler == 'poly':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda epoch: max(
                    (1 - epoch / self.config.num_epochs) ** 0.9,
                    getattr(self.config, 'min_learning_rate', 1e-7) / base_lr
                )
            )
        else:
            self.scheduler = None

    def setup_data(self):
        """Setup datasets and dataloaders"""
        augmentation_params = {
            'p_zoom': getattr(self.config, 'p_zoom', 0.5),
            'zoom_range': getattr(self.config, 'zoom_range', (0.8, 1.2)),
            'p_rotate': getattr(self.config, 'p_rotate', 0.5),
            'rotate_range': getattr(self.config, 'rotate_range', (-15, 15)),
            'p_gaussian_noise': getattr(self.config, 'p_gaussian_noise', 0.15),
            'gaussian_noise_std': getattr(self.config, 'gaussian_noise_std', 0.01),
            'p_flip': getattr(self.config, 'p_flip', 0.5),
        }

        # Source domain
        self.source_train_ds = SourceDomainDataset(
            data_dir=self.config.data_dir,
            is_train=True,
            patch_size=self.config.patch_size,
            augmentation_params=augmentation_params,
            debug_augmentations=self.config.debug_augmentations_visualisation
        )

        self.source_val_ds = SourceDomainDataset(
            data_dir=self.config.data_dir,
            is_train=False,
            patch_size=self.config.patch_size
        )

        # Target domain
        target_data_dir = getattr(self.config, 'target_data_dir', None)
        if target_data_dir:
            self.target_ds = TargetDomainDataset(
                data_dir=target_data_dir,
                is_train=True,
                patch_size=self.config.patch_size,
                augmentation_params=augmentation_params
            )

            self.target_loader = DataLoader(
                self.target_ds,
                batch_size=min(getattr(self.config, 'train_batch_size', 5), len(self.target_ds)),
                shuffle=True,
                num_workers=min(getattr(self.config, 'num_workers', 4), 4),
                collate_fn=list_data_collate,
                pin_memory=True,
                drop_last=True
            )
            self.use_domain_adaptation = True
            print(f"Using domain adaptation with {len(self.target_ds)} target images")
        else:
            self.target_loader = None
            self.use_domain_adaptation = False
            print("No domain adaptation")

        # Source dataloaders
        self.source_train_loader = DataLoader(
            self.source_train_ds,
            batch_size=getattr(self.config, 'train_batch_size', 5),
            shuffle=True,
            num_workers=min(getattr(self.config, 'num_workers', 8), 8),
            collate_fn=list_data_collate,
            pin_memory=True,
            drop_last=True
        )

        self.source_val_loader = DataLoader(
            self.source_val_ds,
            batch_size=getattr(self.config, 'val_batch_size', 2),
            shuffle=False,
            num_workers=min(getattr(self.config, 'num_workers', 8), 8),
            collate_fn=list_data_collate,
            pin_memory=True
        )

    def update_alpha(self, epoch):
        """Update GRL alpha with sigmoid scheduling"""
        if epoch < 5:
            scaled_alpha = 0.0
        else:
            p = float(epoch - 5) / float(self.config.num_epochs - 5)
            sigmoid_like_value = (2.0 / (1.0 + np.exp(-5 * p)) - 1)
            scaled_alpha = self.max_dann_alpha * sigmoid_like_value
            scaled_alpha = max(0, min(scaled_alpha, self.max_dann_alpha))

        self.model.update_alpha(scaled_alpha)
        self.model.update_loss_weights(epoch, self.config.num_epochs)
        return scaled_alpha

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            # Update domain adaptation parameters
            if self.use_domain_adaptation:
                alpha = self.update_alpha(epoch)

            # Training and validation
            train_metrics = self.train_epoch(epoch)
            val_loss, val_dice = self.validate(epoch)

            # Store metrics
            self.epochs.append(epoch + 1)
            self.training_losses.append(train_metrics['loss'])
            self.validation_losses.append(val_loss)
            self.training_dices.append(train_metrics['dice'])
            self.validation_dices.append(val_dice)
            self.domain_losses.append(train_metrics.get('domain_loss', 0))
            self.domain_accuracies.append(train_metrics.get('domain_acc', 0))
            self.source_accuracies.append(train_metrics.get('src_acc', 0))
            self.target_accuracies.append(train_metrics.get('tgt_acc', 0))
            self.skip_domain_accuracies.append(train_metrics.get('skip_domain_acc', 0))
            self.skip_source_accuracies.append(train_metrics.get('skip_src_acc', 0))
            self.skip_target_accuracies.append(train_metrics.get('skip_tgt_acc', 0))

            # Print summary
            lr_seg = self.optimizer.param_groups[0]['lr']
            lr_dom = self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else lr_seg
            lr_skip = self.optimizer.param_groups[2]['lr'] if len(self.optimizer.param_groups) > 2 else lr_dom

            if self.use_domain_adaptation:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                      f"LR: {lr_seg:.1E}/{lr_dom:.1E}/{lr_skip:.1E} | α: {alpha:.4f} | "
                      f"Loss: {train_metrics['loss']:.4f} (T) {val_loss:.4f} (V) | "
                      f"Dice: {train_metrics['dice']:.4f} (T) {val_dice:.4f} (V) | "
                      f"Dom: {train_metrics['domain_loss']:.4f} | "
                      f"Acc: {train_metrics['domain_acc']:.3f} (S:{train_metrics['src_acc']:.3f}/T:{train_metrics['tgt_acc']:.3f})")
            else:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                      f"LR: {lr_seg:.1E} | "
                      f"Loss: {train_metrics['loss']:.4f} (T) {val_loss:.4f} (V) | "
                      f"Dice: {train_metrics['dice']:.4f} (T) {val_dice:.4f} (V)")

            # Update plots
            self.update_plots()

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Save best model
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                if self.config.model_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'loss': val_loss,
                        'dice': val_dice,
                    }, self.config.model_path)
                    print(f'---> Saved best model: Val Dice {self.best_dice:.4f}')

            # Generate t-SNE periodically
            if (epoch + 1) % 25 == 0 and self.use_domain_adaptation:
                self.generate_tsne(epoch + 1)

        print(f'Training finished. Best validation Dice: {self.best_dice:.4f}')

    def train_epoch(self, epoch):
        """Training epoch with comprehensive metrics"""
        self.model.train()
        metrics = {'loss': 0, 'dice': 0, 'domain_loss': 0, 'domain_acc': 0,
                   'src_acc': 0, 'tgt_acc': 0, 'skip_domain_acc': 0, 'skip_src_acc': 0, 'skip_tgt_acc': 0}
        num_batches = 0
        num_domain_batches = 0

        # Target batch iterator
        if self.use_domain_adaptation and self.target_loader:
            target_iter = iter(self.target_loader)

            def get_next_target_batch():
                nonlocal target_iter
                try:
                    return next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    return next(target_iter)
        else:
            def get_next_target_batch():
                return None

        progress_bar = tqdm(self.source_train_loader, desc=f'Epoch {epoch + 1} (Train)', leave=False)
        dice_monitor_fn = self.model.dice_loss_monitor

        for batch in progress_bar:
            source_images = batch["image"].to(self.device)
            source_labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()

            if self.use_domain_adaptation and self.target_loader:
                target_batch = get_next_target_batch()
                target_images = target_batch["image"].to(self.device)

                total_loss, domain_loss, seg_outputs_logits = self.model.training_step(
                    {"image": source_images, "label": source_labels},
                    {"image": target_images}
                )

                if domain_loss is not None:
                    metrics['domain_loss'] += domain_loss.item()
                    metrics['domain_acc'] += getattr(self.model, 'domain_accuracy', 0)
                    metrics['src_acc'] += getattr(self.model, 'source_domain_accuracy', 0)
                    metrics['tgt_acc'] += getattr(self.model, 'target_domain_accuracy', 0)
                    metrics['skip_domain_acc'] += getattr(self.model, 'skip_domain_accuracy', 0)
                    metrics['skip_src_acc'] += getattr(self.model, 'skip_source_accuracy', 0)
                    metrics['skip_tgt_acc'] += getattr(self.model, 'skip_target_accuracy', 0)
                    num_domain_batches += 1
            else:
                total_loss, _, seg_outputs_logits = self.model.training_step(
                    {"image": source_images, "label": source_labels}
                )

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            total_loss.backward()
            if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            # Calculate dice
            with torch.no_grad():
                outputs_sigmoid = torch.sigmoid(seg_outputs_logits)
                dice = 1.0 - dice_monitor_fn(outputs_sigmoid, source_labels).item()
                if np.isnan(dice) or np.isinf(dice):
                    dice = 0.0

            metrics['loss'] += total_loss.item()
            metrics['dice'] += dice
            num_batches += 1

            # Update progress bar
            postfix = {'loss': metrics['loss'] / num_batches, 'dice': metrics['dice'] / num_batches}
            if num_domain_batches > 0:
                postfix['dom_acc'] = metrics['domain_acc'] / num_domain_batches
            progress_bar.set_postfix(postfix)

        # Calculate averages
        final_metrics = {'loss': metrics['loss'] / max(num_batches, 1), 'dice': metrics['dice'] / max(num_batches, 1)}
        if num_domain_batches > 0:
            for key in ['domain_loss', 'domain_acc', 'src_acc', 'tgt_acc', 'skip_domain_acc', 'skip_src_acc',
                        'skip_tgt_acc']:
                final_metrics[key] = metrics[key] / num_domain_batches
        else:
            for key in ['domain_loss', 'domain_acc', 'src_acc', 'tgt_acc', 'skip_domain_acc', 'skip_src_acc',
                        'skip_tgt_acc']:
                final_metrics[key] = 0

        return final_metrics

    def validate(self, epoch=0):
        """Validation epoch"""
        self.model.eval()
        val_loss = 0
        val_dice = 0
        num_batches = 0

        progress_bar = tqdm(self.source_val_loader, desc=f'Epoch {epoch + 1} (Val)', leave=False)
        dice_monitor_fn = self.model.dice_loss_monitor

        with torch.no_grad():
            for batch in progress_bar:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                total_loss, _, outputs_logits = self.model.training_step(
                    {"image": images, "label": labels}
                )

                outputs_sigmoid = torch.sigmoid(outputs_logits)
                dice = 1.0 - dice_monitor_fn(outputs_sigmoid, labels).item()

                if np.isnan(dice) or np.isinf(dice):
                    dice = 0.0

                loss_value = total_loss.item()
                if not (np.isnan(loss_value) or np.isinf(loss_value)):
                    val_loss += loss_value
                    val_dice += dice
                    num_batches += 1

                if num_batches > 0:
                    progress_bar.set_postfix({
                        'loss': val_loss / num_batches,
                        'dice': val_dice / num_batches
                    })

        return (val_loss / max(num_batches, 1), val_dice / max(num_batches, 1)) if num_batches > 0 else (
        float('inf'), 0.0)

    def generate_tsne(self, epoch):
        """Generate t-SNE visualization"""
        print(f"Calculating t-SNE at epoch {epoch}...")
        source_features, target_features = collect_bottleneck_features(
            model=self.model,
            source_loader=self.source_val_loader,
            target_loader=self.target_loader,
            device=self.device,
            num_batches=5
        )

        X_embedded, y = plot_tsne(source_features, target_features)

        if X_embedded is not None and y is not None:
            self.latest_tsne_data = (X_embedded, y, epoch)
            print("t-SNE data updated for plotting.")

    def update_plots(self):
        """Update comprehensive training plots"""
        if len(self.training_losses) == 0:
            return

        if self.use_domain_adaptation:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
            ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Loss plot
        ax1.plot(self.epochs, self.training_losses, 'b-', label='Training')
        ax1.plot(self.epochs, self.validation_losses, 'r-', label='Validation')
        ax1.set_title(f'Loss (T:{self.training_losses[-1]:.4f}, V:{self.validation_losses[-1]:.4f})')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Dice plot
        ax2.plot(self.epochs, self.training_dices, 'b-', label='Training')
        ax2.plot(self.epochs, self.validation_dices, 'r-', label='Validation')
        ax2.set_title(
            f'Dice Score (T:{self.training_dices[-1]:.4f}, V:{self.validation_dices[-1]:.4f}, Best:{self.best_dice:.4f})')
        ax2.set_ylabel('Dice Score')
        ax2.legend()
        ax2.grid(True)

        if self.use_domain_adaptation:
            # Domain loss
            ax3.plot(self.epochs, self.domain_losses, 'g-', label='Domain Loss')
            ax3.set_title(f'Domain Loss ({self.domain_losses[-1]:.4f})')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True)

            # Learning rates
            current_lr_seg = self.optimizer.param_groups[0]['lr']
            current_lr_dom = self.optimizer.param_groups[1]['lr'] if len(
                self.optimizer.param_groups) > 1 else current_lr_seg
            current_lr_skip = self.optimizer.param_groups[2]['lr'] if len(
                self.optimizer.param_groups) > 2 else current_lr_dom

            lrs_seg = [current_lr_seg] * len(self.epochs)
            lrs_dom = [current_lr_dom] * len(self.epochs)
            lrs_skip = [current_lr_skip] * len(self.epochs)

            ax4.plot(self.epochs, lrs_seg, 'c-', label='Seg')
            ax4.plot(self.epochs, lrs_dom, 'm-', label='Domain')
            ax4.plot(self.epochs, lrs_skip, 'y-', label='Skip')
            ax4.set_title(f'Learning Rates')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True)

            # Domain accuracy
            ax5.plot(self.epochs, self.domain_accuracies, 'g-', label='Overall')
            ax5.plot(self.epochs, self.source_accuracies, 'b--', label='Source')
            ax5.plot(self.epochs, self.target_accuracies, 'r--', label='Target')
            ax5.set_title(f'Domain Classification Accuracy')
            ax5.set_ylabel('Accuracy')
            ax5.set_ylim(0, 1.05)
            ax5.legend()
            ax5.grid(True)

            # t-SNE plot
            ax6.set_facecolor('#f6f6f6')
            ax6.grid(True, linestyle=':', linewidth=0.5, color='gray')
            ax6.set_xticks([])
            ax6.set_yticks([])

            if self.latest_tsne_data is not None:
                X_embedded, y, tsne_epoch = self.latest_tsne_data
                ax6.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1],
                            label='Source', alpha=0.6, s=8, c='blue')
                ax6.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1],
                            label='Target', alpha=0.6, s=8, c='red')
                ax6.set_title(f't-SNE (Epoch {tsne_epoch})', fontsize=10)
                ax6.legend(fontsize=8, loc='upper right')
            else:
                ax6.text(0.5, 0.5, 't-SNE\n(pending)', ha='center', va='center',
                         fontsize=10, color='gray')

        # Footer with current status
        current_lr_seg = self.optimizer.param_groups[0]['lr']
        alpha_val = getattr(self.model.grl, 'alpha', 0) if hasattr(self.model, 'grl') else 0

        if self.use_domain_adaptation:
            plt.figtext(0.5, 0.01,
                        f"Epoch: {self.epochs[-1]}/{self.config.num_epochs} | "
                        f"LR: {current_lr_seg:.1E} | α: {alpha_val:.3f} | "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        ha="center", fontsize=9,
                        bbox={"facecolor": "orange", "alpha": 0.3, "pad": 5})
        else:
            plt.figtext(0.5, 0.01,
                        f"Epoch: {self.epochs[-1]}/{self.config.num_epochs} | "
                        f"LR: {current_lr_seg:.1E} | "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        ha="center", fontsize=9,
                        bbox={"facecolor": "orange", "alpha": 0.3, "pad": 5})

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        # Save plot
        plot_path = os.path.join(self.figures_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

