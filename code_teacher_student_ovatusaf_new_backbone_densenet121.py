import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import segmentation_models_pytorch as smp
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Set random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ===== DATASET CLASS =====
class KvasirDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, is_labeled=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_labeled = is_labeled

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_labeled:
            # Load mask
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            # Ensure mask has correct shape [H, W] -> [1, H, W]
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

            return image, mask.float()  # Changed to float for BCE loss
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']

            return image


# ===== DATA AUGMENTATION =====
def get_transforms():
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform


import torch
import torch.nn as nn
import torch.nn.functional as F

def sobel_edges(x):
    """Compute Sobel edge magnitude"""
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=torch.float32, device=x.device
    ).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=torch.float32, device=x.device
    ).unsqueeze(0).unsqueeze(0)

    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)

    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    return edge


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = target.float()

        pred_edge = sobel_edges(pred)
        target_edge = sobel_edges(target)

        return F.mse_loss(pred_edge, target_edge)

class LowContrastLoss(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = target.float()

        target_edge = sobel_edges(target)
        contrast_mask = (target_edge < self.threshold).float()

        return F.mse_loss(pred * contrast_mask, target * contrast_mask)



class BALCLoss(nn.Module):
    """
    Ablation modes:
    - 'bce'
    - 'bce+edge'
    - 'bce+contrast'
    - 'full' (BCE + Edge + Low-Contrast)
    """
    def __init__(self, mode='full', alpha=1.0, beta=0.5):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.beta = beta

        self.bce = nn.BCEWithLogitsLoss()
        self.edge_loss = BoundaryLoss()
        self.contrast_loss = LowContrastLoss()

    def forward(self, pred, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)

        loss = self.bce(pred, target)

        if self.mode in ['bce+edge', 'full']:
            loss = loss + self.alpha * self.edge_loss(pred, target)

        if self.mode in ['bce+contrast', 'full']:
            loss = loss + self.beta * self.contrast_loss(pred, target)

        return loss


from torch.nn.utils import parameters_to_vector


class ParameterMonitor:
    def __init__(self, student_model, teacher_model):
        self.student = student_model
        self.teacher = teacher_model

        # Storage for metrics
        self.history = {
            'epoch': [],
            'param_distance': [],
            'param_similarity': [],
            'ema_alpha': []
        }

    def compute_parameter_distance(self):
        """Compute L2 distance between student and teacher parameters"""
        distance = 0.0
        total_params = 0

        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            diff = (s_param.data - t_param.data).flatten()
            distance += torch.norm(diff, p=2).item() ** 2
            total_params += diff.numel()

        return np.sqrt(distance / total_params)

    def compute_parameter_similarity(self):
        """Compute cosine similarity between student and teacher parameters"""
        s_vec = parameters_to_vector(self.student.parameters())
        t_vec = parameters_to_vector(self.teacher.parameters())

        similarity = torch.nn.functional.cosine_similarity(s_vec.unsqueeze(0), t_vec.unsqueeze(0))
        return similarity.item()

    def update_history(self, epoch, alpha):
        """Update monitoring history"""
        param_dist = self.compute_parameter_distance()
        param_sim = self.compute_parameter_similarity()

        self.history['epoch'].append(epoch)
        self.history['param_distance'].append(param_dist)
        self.history['param_similarity'].append(param_sim)
        self.history['ema_alpha'].append(alpha)

    def plot_results(self, save_path=None):
        """Plot all monitoring results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Parameter Monitoring: Student vs Teacher', fontsize=16)

        epochs = self.history['epoch']

        # Parameter distance
        axes[0].plot(epochs, self.history['param_distance'], 'b-', linewidth=2)
        axes[0].set_title('Parameter Distance (L2)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Distance')
        axes[0].grid(True, alpha=0.3)

        # Parameter similarity
        axes[1].plot(epochs, self.history['param_similarity'], 'g-', linewidth=2)
        axes[1].set_title('Parameter Similarity (Cosine)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Similarity')
        axes[1].grid(True, alpha=0.3)

        # EMA Alpha
        axes[2].plot(epochs, self.history['ema_alpha'], 'r-', linewidth=2)
        axes[2].set_title('EMA Alpha Value')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Alpha')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def print_summary(self):
        """Print summary statistics"""
        if not self.history['epoch']:
            print("No data available")
            return

        print("="*50)
        print("PARAMETER MONITORING SUMMARY")
        print("="*50)
        print(f"Total Epochs: {len(self.history['epoch'])}")
        print(f"Final Distance: {self.history['param_distance'][-1]:.6f}")
        print(f"Final Similarity: {self.history['param_similarity'][-1]:.6f}")
        print(f"EMA Alpha: {self.history['ema_alpha'][-1]:.4f}")

        # Calculate trends
        if len(self.history['param_distance']) > 5:
            recent_distances = self.history['param_distance'][-5:]
            trend = np.polyfit(range(5), recent_distances, 1)[0]
            print(f"Distance Trend: {'↓ Decreasing' if trend < 0 else '↑ Increasing'} ({trend:.6f})")
        print("="*50)


# ===== MEAN TEACHER MODEL =====
class MeanTeacher:
    def __init__(self, student_model, teacher_model, alpha=0.99):
        self.student = student_model
        self.teacher = teacher_model
        self.alpha = alpha

        # Initialize teacher with student weights
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.copy_(student_param.data)

        # Initialize monitor
        self.monitor = ParameterMonitor(student_model, teacher_model)

    def update_teacher(self, epoch):
        """Update teacher model and monitoring"""
        # EMA update
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data = self.alpha * teacher_param.data + (1 - self.alpha) * student_param.data

        # Update monitoring
        self.monitor.update_history(epoch, self.alpha)

# ===== MODEL DEFINITION =====
encoder_names= "densenet121" #mobilenet_v2, efficientnet-b0, resnet34, efficientnet-b4+, mobilenet_v3, se_resnet50, resnet50
def create_model():
    model = smp.Unet(
        # encoder_name="densenet121",
        encoder_name=encoder_names,
        # mobilenet_v2, efficientnet-b0, resnet34, efficientnet-b4+, mobilenet_v3, se_resnet50, resnet50
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    return model

# ===== METRICS =====
def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # Ensure target has same shape as pred
    if target.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
        target = target.unsqueeze(1)

    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return dice.mean()

def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # Ensure target has same shape as pred
    if target.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
        target = target.unsqueeze(1)

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def train_semi_supervised(student_model, teacher_model, labeled_loader, unlabeled_loader,
                               val_loader, criterion, optimizer, scheduler, device, num_epochs=50):

    # Initialize Mean Teacher with monitoring
    mean_teacher = MeanTeacher(student_model, teacher_model, alpha=0.99)

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
    best_dice = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        # Training phase
        student_model.train()
        teacher_model.eval()

        running_loss = 0.0
        num_batches = 0

        # Create iterators
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        max_batches = max(len(labeled_loader), len(unlabeled_loader))

        for batch_idx in tqdm(range(max_batches), desc='Training'):
            optimizer.zero_grad()
            total_loss = 0.0

            # Supervised loss
            try:
                labeled_images, labeled_masks = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_images, labeled_masks = next(labeled_iter)

            labeled_images = labeled_images.to(device)
            labeled_masks = labeled_masks.to(device)

            student_pred = student_model(labeled_images)
            supervised_loss = criterion(student_pred, labeled_masks)
            total_loss += supervised_loss

            # Consistency loss
            try:
                unlabeled_images = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_images = next(unlabeled_iter)

            unlabeled_images = unlabeled_images.to(device)

            with torch.no_grad():
                teacher_pred = teacher_model(unlabeled_images)

            student_pred_unlabeled = student_model(unlabeled_images)
            consistency_loss = F.mse_loss(torch.sigmoid(student_pred_unlabeled),
                                        torch.sigmoid(teacher_pred))

            total_loss += 0.1 * consistency_loss

            total_loss.backward()
            optimizer.step()

            # Update teacher and monitoring
            mean_teacher.update_teacher(epoch)

            running_loss += total_loss.item()
            num_batches += 1

        # Validation phase
        student_model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_batches = 0

        with torch.no_grad():
            for val_images, val_masks in tqdm(val_loader, desc='Validation'):
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)

                val_pred = student_model(val_images)
                v_loss = criterion(val_pred, val_masks)
                v_dice = dice_coefficient(val_pred, val_masks.float())
                v_iou = iou_score(val_pred, val_masks.float())

                val_loss += v_loss.item()
                val_dice += v_dice.item()
                val_iou += v_iou.item()
                val_batches += 1

        # Calculate averages
        epoch_train_loss = running_loss / num_batches
        epoch_val_loss = val_loss / val_batches
        epoch_val_dice = val_dice / val_batches
        epoch_val_iou = val_iou / val_batches

        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_dice'].append(epoch_val_dice)
        history['val_iou'].append(epoch_val_iou)

        # Print results
        print(f'Train Loss: {epoch_train_loss:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Dice: {epoch_val_dice:.4f}, IoU: {epoch_val_iou:.4f}')

        # Print monitoring info every 5 epochs
        if epoch % 5 == 0:
            monitor = mean_teacher.monitor
            if monitor.history['param_distance']:
                dist = monitor.history['param_distance'][-1]
                sim = monitor.history['param_similarity'][-1]
                print(f'Param Distance: {dist:.6f}, Similarity: {sim:.6f}')

        # Save best model
        if epoch_val_dice > best_dice:
            best_dice = epoch_val_dice
            torch.save(student_model.state_dict(), '/mnt/nvme0/home/loanpt/CombineLoss/files_solid_code/best_model/best_model_ova.pth')
            print(f'✓ Best model saved! Dice: {best_dice:.4f}')

        scheduler.step()

        # Plot results every 10 epochs
        # if (epoch + 1) % 10 == 0:
        if epoch  == num_epochs-1:
            df = pd.DataFrame({
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'val_dice': history['val_dice'],
                'val_iou': history['val_iou']
            })

            df.to_csv('training_history_ovatu_densenet121.csv', index=False)
        #     mean_teacher.monitor.plot_results(save_path=f'monitoring_epoch_{epoch+1}.png')

    # Final results
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    mean_teacher.monitor.print_summary()
    # mean_teacher.monitor.plot_results(save_path='final_monitoring_results.png')

    return history, mean_teacher.monitor

# ===== PLOTTING FUNCTIONS =====
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training & Validation Loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Validation Dice Score
    axes[0, 1].plot(history['val_dice'], label='Validation Dice', color='green')
    axes[0, 1].set_title('Validation Dice Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Validation IoU
    axes[1, 0].plot(history['val_iou'], label='Validation IoU', color='orange')
    axes[1, 0].set_title('Validation IoU Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Combined metrics
    axes[1, 1].plot(history['val_dice'], label='Dice Score', color='green')
    axes[1, 1].plot(history['val_iou'], label='IoU Score', color='orange')
    axes[1, 1].set_title('Validation Metrics Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('training_history_resnet50.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===== MAIN TRAINING SCRIPT ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

    # Prepare data paths
base_path = "/mnt/nvme0/home/loanpt/CombineLoss/KetQuaChayLai/MERGED_V1_V3/"
images_path = os.path.join(base_path, "images")
masks_path = os.path.join(base_path, "mask")

image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.JPG', '.jpg', '.png'))])
# mask_files = [f.replace('.JPG', '.png') for f in image_files]
# mask_files = [f.replace('.jpg', '.png') for f in mask_files]
mask_files = [
    os.path.splitext(f)[0] + '.png'
    for f in image_files
]


image_paths = [os.path.join(images_path, f) for f in image_files]
mask_paths = [os.path.join(masks_path, f) for f in mask_files]

    # Split data
train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Split training data into labeled (30%) and unlabeled (70%)
labeled_images, unlabeled_images, labeled_masks, _ = train_test_split(
        train_images, train_masks, test_size=0.7, random_state=42
    )
# unlabeled_images_p = "/mnt/nvme0/home/loanpt/CombineLoss/KetQuaChayLai/MERGED_V1_V3/"
# unlabeled_images_image = sorted([f for f in os.listdir(unlabeled_images_p) if f.endswith(('.jpg', '.jpeg', '.png'))])
# unlabeled_images = [os.path.join(unlabeled_images_p, f) for f in unlabeled_images_image]

print(f"Labeled training samples: {len(labeled_images)}")
print(f"Unlabeled training samples: {len(unlabeled_images)}")
print(f"Validation samples: {len(val_images)}")

# Get transforms
train_transform, val_transform = get_transforms()

# Create datasets
labeled_dataset = KvasirDataset(labeled_images, labeled_masks, train_transform, is_labeled=True)
unlabeled_dataset = KvasirDataset(unlabeled_images, unlabeled_images, train_transform, is_labeled=False)  # No masks for unlabeled
val_dataset = KvasirDataset(val_images, val_masks, val_transform, is_labeled=True)

# Create data loaders
labeled_loader = DataLoader(labeled_dataset, batch_size=8, shuffle=True, num_workers=2)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

# Create models
student_model = create_model().to(device)
teacher_model = create_model().to(device)


print('Encoder Name: ', encoder_names)
# Loss function and optimizer
criterion = BALCLoss(mode='full')
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

print("Starting training...")

history, monitor = train_semi_supervised(
    student_model, teacher_model,
    labeled_loader, unlabeled_loader, val_loader,
    criterion, optimizer, scheduler, device,
    num_epochs=100
    )

# Plot results
# plot_training_history(history)

print("\nTraining completed!")
print("Best model saved as '/mnt/nvme0/home/loanpt/CombineLoss/files_solid_code/best_model/best_model_ova.pth'")
print("Training history plot saved as 'training_history_resnet50.png'")
def visualize_predictions(model_path, val_loader, device, num_samples=5):
    """Visualize model predictions"""
    model = create_model().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= num_samples:
                break

            images = images.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5

            # Take first image from batch
            image = images[0].cpu().numpy().transpose(1, 2, 0)
            mask = masks[0].cpu().numpy()
            pred = predictions[0].cpu().numpy()

            # Handle mask shape - remove channel dimension if exists
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)  # (1, H, W) -> (H, W)
            elif mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)  # (H, W, 1) -> (H, W)

            # Handle prediction shape - remove channel dimension if exists
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred.squeeze(0)  # (1, H, W) -> (H, W)
            elif pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)  # (H, W, 1) -> (H, W)

            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)

            axes[i, 0].imshow(image)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
    # plt.show()

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

visualize_predictions('/mnt/nvme0/home/loanpt/CombineLoss/files_solid_code/best_model/best_model_ova.pth', val_loader, device)
