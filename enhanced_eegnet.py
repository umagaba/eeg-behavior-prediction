"""
Enhanced EEGNet for Externalizing Score Regression (EEG2025 Task 2)
===================================================================

This script implements an Enhanced EEGNet architecture with demographic fusion,
domain adaptation, and multi-head self-attention for regression-based externalizing
behavior prediction from 128-channel EEG data.

Key Features:
- EEGNet-based regression architecture (regression output instead of classification)
- AdaBN (Adaptive Batch Normalization) for cross-subject domain adaptation
- Multi-head self-attention for capturing channel relationships
- Late fusion demographic integration (age, sex, handedness)
- RMSE/nRMSE loss functions for continuous target prediction
- Early stopping with validation monitoring
- Dropout (0.5) for regularization
- Data loading from multiple EEG tasks (CCD + passive tasks)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONCEPT DEFINITIONS
# ============================================================================
"""
KEY CONCEPTS USED IN THIS CODE:

1. **Regression vs Classification**: 
   While classification predicts discrete categories (softmax output),
   regression predicts continuous values (no activation on final layer).

2. **AdaBN (Adaptive Batch Normalization)**:
   Updates batch normalization statistics on target domain data without
   retraining weights, enabling domain adaptation across subjects.

3. **Multi-Head Self-Attention**:
   Mechanism allowing model to focus on different EEG channels/features
   simultaneously, learning channel importance weights dynamically.

4. **Late Fusion**:
   Concatenates EEG representations with demographic features in a separate
   fusion network, allowing independent feature learning before combination.

5. **RMSE Loss**:
   Root Mean Square Error: sqrt(mean((y_pred - y_true)^2)); emphasizes
   large errors more than MAE, suitable for continuous predictions.

6. **nRMSE (Normalized RMSE)**:
   RMSE divided by range of target values; scale-invariant metric for
   comparing performance across different prediction ranges.

7. **Early Stopping**:
   Terminates training when validation loss stops improving, preventing
   overfitting by finding optimal training epoch.

8. **Gradient Clipping**:
   Scales gradients if their norm exceeds threshold; prevents exploding
   gradients during backpropagation, stabilizing training.

9. **AdamW Optimizer**:
   Variant of Adam with weight decay (L2 regularization); combines adaptive
   learning rates with explicit regularization term.

10. **ReduceLROnPlateau**:
    Learning rate scheduler that reduces LR by factor (0.5x) if validation
    metric plateaus; helps escape local minima.
"""

# ============================================================================
# CONFIGURATION AND LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class RMSELoss(nn.Module):
    """
    Root Mean Square Error Loss for regression.
    Measures average magnitude of prediction errors.
    
    Formula: sqrt(mean((pred - target)^2))
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target) + 1e-8)

class NRMSELoss(nn.Module):
    """
    Normalized RMSE: RMSE divided by target range.
    Makes loss scale-invariant; comparable across different score ranges.
    
    Formula: sqrt(mean((pred - target)^2)) / (max(target) - min(target))
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        rmse = torch.sqrt(self.mse(pred, target) + 1e-8)
        target_range = target.max() - target.min() + 1e-8
        return rmse / target_range

# ============================================================================
# ENHANCED EEGNET ARCHITECTURE WITH DEMOGRAPHIC FUSION
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention: Allows model to jointly attend to information
    from different representation subspaces (different EEG features).
    Learns which channels/frequencies are most relevant for prediction.
    
    Args:
        embed_dim: Dimension of each attention head
        num_heads: Number of parallel attention mechanisms
        dropout: Dropout rate within attention
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            attn_output: (batch, seq_len, embed_dim)
        """
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class EnhancedEEGNetRegressor(nn.Module):
    """
    Enhanced EEGNet for continuous externalizing score regression.
    
    Architecture:
    1. Temporal Conv: Learns frequency-specific filters
    2. Spatial Conv: Depthwise convolution across 128 EEG channels
    3. Batch Norm: Normalizes activations
    4. Multi-Head Attention: Captures channel relationships
    5. Dropout: Regularization (0.5)
    6. Demographic Fusion: Late concatenation with age/sex/handedness
    7. Regression Head: Linear output (no activation)
    
    Key Changes from Standard EEGNet:
    - Output: 1 continuous value instead of class probabilities
    - Added BatchNorm after conv blocks
    - Integrated multi-head self-attention
    - Dropout increased to 0.5
    - Late fusion for demographic data
    """
    def __init__(
        self,
        n_channels: int = 128,
        n_times: int = 200,  # 2 seconds @ 100Hz
        n_demographic_features: int = 3,  # age, sex, handedness
        dropout: float = 0.5,
        F1: int = 16,  # temporal filters
        D: int = 2,    # depth multiplier for spatial conv
        num_heads: int = 8,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.dropout_rate = dropout
        
        # ===== EEG PROCESSING BRANCH =====
        
        # Temporal convolution: learns frequency filters (1-50 Hz range)
        self.temporal_conv = nn.Conv2d(
            1, F1, kernel_size=(1, 51), stride=(1, 1),
            padding=(0, 25), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Spatial depthwise convolution: learns spatial filters across 128 channels
        self.spatial_conv = nn.Conv2d(
            F1, F1 * D, kernel_size=(n_channels, 1), stride=(1, 1),
            groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        
        # Activation and pooling
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout1 = nn.Dropout(p=dropout)
        
        # Multi-head self-attention: captures channel/feature interactions
        self.attention = MultiHeadAttention(
            embed_dim=F1 * D,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Second pooling after attention
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout2 = nn.Dropout(p=dropout)
        
        # Calculate final EEG feature dimension
        # After temporal conv: 200 timepoints
        # After pool1 (stride 4): 50 timepoints
        # After attention: 50 timepoints
        # After pool2 (stride 2): 25 timepoints
        self.eeg_feature_dim = F1 * D * 25  # 32 * 25 = 800
        
        # ===== DEMOGRAPHIC FUSION BRANCH =====
        
        # Small MLP for demographic feature processing
        self.demographic_encoder = nn.Sequential(
            nn.Linear(n_demographic_features, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, 32)
        )
        
        # ===== FUSION AND REGRESSION HEAD =====
        
        # Combine EEG + demographic features
        fusion_input_dim = self.eeg_feature_dim + 32
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        # Regression output: continuous score (no softmax/sigmoid)
        self.regression_head = nn.Linear(32, 1)
    
    def forward(self, eeg: torch.Tensor, demographics: Optional[torch.Tensor] = None):
        """
        Forward pass for regression.
        
        Args:
            eeg: (batch_size, 1, n_channels, n_times)
            demographics: (batch_size, 3) - [age_normalized, sex, handedness]
        
        Returns:
            continuous externalizing scores: (batch_size, 1)
        """
        # ===== EEG BRANCH =====
        x = self.temporal_conv(eeg)
        x = self.bn1(x)
        x = self.elu(x)
        
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Reshape for attention: (batch, spatial_dim, seq_len)
        batch_size = x.shape[0]
        x = x.squeeze(2)  # Remove spatial dim (which is now 1)
        x = x.transpose(1, 2)  # (batch, seq_len, F1*D)
        
        # Apply multi-head attention
        x = self.attention(x)
        
        x = x.transpose(1, 2)  # Back to (batch, F1*D, seq_len)
        x = x.unsqueeze(2)     # (batch, F1*D, 1, seq_len)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten for dense layers
        eeg_features = x.view(batch_size, -1)
        
        # ===== DEMOGRAPHIC BRANCH (if provided) =====
        if demographics is not None:
            demo_features = self.demographic_encoder(demographics)
            combined_features = torch.cat([eeg_features, demo_features], dim=1)
        else:
            # If no demographics, use EEG features alone
            combined_features = eeg_features
        
        # ===== FUSION AND REGRESSION =====
        fused = self.fusion(combined_features)
        output = self.regression_head(fused)
        
        return output

# ============================================================================
# DOMAIN ADAPTATION WITH ADAPTIVE BATCH NORMALIZATION
# ============================================================================

def adapt_batch_norm(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Adaptive Batch Normalization (AdaBN): Updates batch norm statistics
    to target domain without retraining weights. Helps model generalize
    to new subjects with different EEG signal distributions.
    
    For each batch norm layer, computes running mean/variance on target domain
    data, then applies these statistics during inference.
    
    Args:
        model: The neural network model
        dataloader: DataLoader with validation/target domain data
        device: torch.device (cuda or cpu)
    """
    logger.info("Adapting BatchNorm statistics to validation domain...")
    model.eval()
    
    # Set BN to training mode (to compute statistics) but no gradient updates
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.train()
    
    with torch.no_grad():
        for eeg_batch, demo_batch, _ in tqdm(dataloader, desc="AdaBN", leave=False):
            eeg_batch = eeg_batch.to(device)
            if demo_batch is not None:
                demo_batch = demo_batch.to(device)
            
            _ = model(eeg_batch, demo_batch)
    
    # Return to eval mode
    model.eval()
    logger.info("BatchNorm adaptation complete.")

# ============================================================================
# DATASET HANDLING
# ============================================================================

class EEGRegressionDataset(Dataset):
    """
    PyTorch Dataset for EEG regression with demographic features.
    Loads EEG segments and corresponding demographic/target information.
    
    Args:
        eeg_data: (N, 128, 200) EEG segments or (N, n_samples)
        externalizing_scores: (N,) continuous target values
        demographics: (N, 3) demographic features [age, sex, handedness]
        normalize_eeg: Whether to normalize EEG data
        eeg_scaler: Pre-fitted StandardScaler (for test set)
    """
    def __init__(
        self,
        eeg_data: np.ndarray,
        externalizing_scores: np.ndarray,
        demographics: np.ndarray,
        normalize_eeg: bool = True,
        eeg_scaler: Optional[StandardScaler] = None
    ):
        self.eeg_data = eeg_data
        self.externalizing_scores = externalizing_scores.astype(np.float32)
        self.demographics = demographics.astype(np.float32)
        
        # Normalize EEG data channel-wise (crucial for cross-subject generalization)
        if normalize_eeg:
            if eeg_scaler is None:
                self.scaler = StandardScaler()
                # Reshape for scaling: (N*128, 200) or equivalent
                original_shape = eeg_data.shape
                eeg_reshaped = eeg_data.reshape(original_shape[0], -1)
                self.scaler.fit(eeg_reshaped)
            else:
                self.scaler = eeg_scaler
            
            eeg_reshaped = eeg_data.reshape(eeg_data.shape[0], -1)
            eeg_normalized = self.scaler.transform(eeg_reshaped)
            self.eeg_data = eeg_normalized.reshape(original_shape)
        else:
            self.scaler = None
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            eeg: (1, 128, 200) tensor
            demographics: (3,) tensor
            target: (1,) tensor with externalizing score
        """
        eeg = torch.FloatTensor(self.eeg_data[idx])
        
        # Add channel dimension if needed: (128, 200) -> (1, 128, 200)
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)
        
        target = torch.FloatTensor([self.externalizing_scores[idx]])
        demographics = torch.FloatTensor(self.demographics[idx])
        
        return eeg, demographics, target

# ============================================================================
# TRAINING LOOP WITH EARLY STOPPING
# ============================================================================

class EarlyStoppingCallback:
    """
    Early Stopping: Stops training when validation loss plateaus,
    preventing overfitting. Saves best model weights.
    
    Monitors validation metric and stops if no improvement for 'patience' epochs.
    
    Args:
        patience: Number of epochs without improvement before stopping
        verbose: Whether to print progress messages
    """
    def __init__(self, patience: int = 10, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module, save_path: str = "best_model.pt"):
        """
        Update early stopping state and save best model.
        
        Args:
            val_loss: Current validation loss
            model: Model to save
            save_path: Path to save best model
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            torch.save(model.state_dict(), save_path)
        elif val_loss < self.best_loss * 0.99:  # 1% improvement threshold
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), save_path)
            if self.verbose:
                logger.info(f"Validation loss improved to {val_loss:.6f}. Model saved.")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered.")

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0
) -> float:
    """
    Single training epoch with gradient clipping for stability.
    
    Gradient Clipping: Prevents exploding gradients by scaling gradients
    if their norm exceeds threshold.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: torch.device
        grad_clip: Maximum gradient norm
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for eeg_batch, demo_batch, target_batch in tqdm(train_loader, desc="Train", leave=False):
        eeg_batch = eeg_batch.to(device)
        demo_batch = demo_batch.to(device) if demo_batch is not None else None
        target_batch = target_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(eeg_batch, demo_batch)
        loss = criterion(predictions, target_batch)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Validation epoch: evaluate model without gradient computation.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: torch.device
        
    Returns:
        avg_loss: Average validation loss
        predictions: Array of predictions
        targets: Array of ground truth targets
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for eeg_batch, demo_batch, target_batch in tqdm(val_loader, desc="Val", leave=False):
            eeg_batch = eeg_batch.to(device)
            demo_batch = demo_batch.to(device) if demo_batch is not None else None
            target_batch = target_batch.to(device)
            
            predictions = model(eeg_batch, demo_batch)
            loss = criterion(predictions, target_batch)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target_batch.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    
    return avg_loss, predictions, targets

def calculate_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary with RMSE, nRMSE, MAE, Pearson correlation
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (targets.max() - targets.min() + 1e-8)
    mae = np.mean(np.abs(predictions - targets))
    
    # Pearson correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(predictions, targets)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'nrmse': nrmse,
        'mae': mae,
        'pearson_r': corr
    }

def train_regression_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 15,
    use_nrmse: bool = True
) -> Tuple[Dict, Dict]:
    """
    Complete training loop with early stopping and domain adaptation.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model: Neural network model
        device: torch.device
        n_epochs: Maximum number of epochs
        learning_rate: Learning rate for AdamW
        early_stopping_patience: Patience for early stopping
        use_nrmse: Use nRMSE loss (True) or RMSE loss (False)
        
    Returns:
        history: Dictionary with training history
        final_metrics: Dictionary with final metrics
    """
    # Loss function selection
    if use_nrmse:
        criterion = NRMSELoss()
        logger.info("Using nRMSE loss")
    else:
        criterion = RMSELoss()
        logger.info("Using RMSE loss")
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler: reduces LR if loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(patience=early_stopping_patience, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': []
    }
    
    best_model_path = "best_regression_model.pt"
    
    for epoch in range(n_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{n_epochs}")
        logger.info(f"{'='*60}")
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        logger.info(f"Train Loss: {train_loss:.6f}")
        
        # Validation
        val_loss, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        
        # Calculate additional metrics
        metrics = calculate_regression_metrics(val_preds, val_targets)
        history['val_rmse'].append(metrics['rmse'])
        history['val_mae'].append(metrics['mae'])
        
        logger.info(f"Val Loss: {val_loss:.6f}")
        logger.info(f"Val RMSE: {metrics['rmse']:.6f}")
        logger.info(f"Val nRMSE: {metrics['nrmse']:.6f}")
        logger.info(f"Val MAE: {metrics['mae']:.6f}")
        logger.info(f"Val Pearson r: {metrics['pearson_r']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss, model, best_model_path)
        if early_stopping.early_stop:
            logger.info(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    logger.info(f"\nLoaded best model from {best_model_path}")
    
    return history, metrics

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def create_dummy_data_for_demo():
    """
    Creates dummy data for demonstration.
    Replace with actual data loading from EEG2025 competition.
    
    Returns:
        eeg_data: (100, 128, 200) random EEG samples
        externalizing_scores: (100,) random scores [0, 100]
        demographics: (100, 3) random demographic features
    """
    logger.info("Creating demonstration dataset...")
    
    # Dummy EEG data: 100 samples, 128 channels, 200 timepoints (2 sec @ 100Hz)
    eeg_data = np.random.randn(100, 128, 200).astype(np.float32)
    
    # Dummy externalizing scores: continuous values [0, 100]
    externalizing_scores = np.random.uniform(0, 100, 100).astype(np.float32)
    
    # Dummy demographics: [age_normalized, sex, handedness]
    demographics = np.random.randn(100, 3).astype(np.float32)
    
    return eeg_data, externalizing_scores, demographics

if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("ENHANCED EEGNET FOR EXTERNALIZING SCORE REGRESSION")
    logger.info("="*80)
    
    # ===== DATA LOADING =====
    eeg_data, externalizing_scores, demographics = create_dummy_data_for_demo()
    
    # Split: 80% train, 20% val
    n_samples = len(eeg_data)
    n_train = int(0.8 * n_samples)
    
    train_eeg = eeg_data[:n_train]
    train_scores = externalizing_scores[:n_train]
    train_demo = demographics[:n_train]
    
    val_eeg = eeg_data[n_train:]
    val_scores = externalizing_scores[n_train:]
    val_demo = demographics[n_train:]
    
    # Create datasets and dataloaders
    train_dataset = EEGRegressionDataset(
        train_eeg, train_scores, train_demo, normalize_eeg=True
    )
    val_dataset = EEGRegressionDataset(
        val_eeg, val_scores, val_demo,
        normalize_eeg=True,
        eeg_scaler=train_dataset.scaler
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # ===== MODEL INITIALIZATION =====
    model = EnhancedEEGNetRegressor(
        n_channels=128,
        n_times=200,
        n_demographic_features=3,
        dropout=0.5,
        F1=16,
        D=2,
        num_heads=8
    ).to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # ===== TRAINING =====
    history, final_metrics = train_regression_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        n_epochs=100,
        learning_rate=0.001,
        early_stopping_patience=15,
        use_nrmse=True
    )
    
    # ===== DOMAIN ADAPTATION (AdaBN) =====
    adapt_batch_norm(model, val_loader, device)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Final Metrics:")
    logger.info(f"  RMSE: {final_metrics['rmse']:.6f}")
    logger.info(f"  nRMSE: {final_metrics['nrmse']:.6f}")
    logger.info(f"  MAE: {final_metrics['mae']:.6f}")
    logger.info(f"  Pearson r: {final_metrics['pearson_r']:.4f}")
