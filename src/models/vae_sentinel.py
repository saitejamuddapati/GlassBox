"""
Deep Variational Autoencoder (VAE) Anomaly Sentinel for GlassBox.

This module implements a semi-supervised Deep VAE for zero-day fraud and
out-of-distribution anomaly detection. The network is trained strictly on 100%
legitimate transactions (Class = 0) to learn the continuous manifold of normal
cardholder behavior.

When an unseen / novel fraud attempt occurs, its inability to reconstruct accurately
produces a high Reconstruction Loss (MSE), flagging the transaction for 2FA review
without requiring prior exposure to the attack vector.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class VAEArchitecture(nn.Module):
    """
    Symmetric Variational Autoencoder neural network.
    
    Architecture:
      Input (D) -> Enc_Hidden1 (24) -> Enc_Hidden2 (12) -> [mu (6), logvar (6)]
      Sample z ~ N(mu, exp(0.5 * logvar))
      z (6) -> Dec_Hidden1 (12) -> Dec_Hidden2 (24) -> Reconstructed Output (D)
    """

    def __init__(self, input_dim: int, latent_dim: int = 6, hidden_dim1: int = 24, hidden_dim2: int = 12):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.1),
        )
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)

        # Decoder
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim1, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class DeepVAESentinel:
    """
    Production wrapper for Deep VAE Anomaly Detection in GlassBox.
    
    Provides:
    - Preprocessing with StandardScaler fitted on legitimate transactions
    - PyTorch training with early stopping & mini-batch gradient descent
    - Vectorized sample anomaly scoring (MSE)
    - Feature-level reconstruction deltas (|x_i - x̂_i|) for explainability
    - Model persistence (.pt and .joblib metadata)
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        latent_dim: int = 6,
        beta_kl: float = 0.005,
        random_state: int = 42,
    ):
        self.feature_names = feature_names or []
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: Optional[VAEArchitecture] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_score: float = 0.0
        self.max_score: float = 1.0
        self.threshold: float = 0.5

    def _set_seed(self) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def fit(
        self,
        X_normal: Union[pd.DataFrame, np.ndarray],
        epochs: int = 20,
        batch_size: int = 512,
        lr: float = 2e-3,
        val_split: float = 0.1,
        verbose: bool = True,
    ) -> "DeepVAESentinel":
        """
        Train the VAE strictly on legitimate transactions (Class = 0).
        """
        self._set_seed()

        if isinstance(X_normal, pd.DataFrame):
            if not self.feature_names:
                self.feature_names = list(X_normal.columns)
            X_mat = X_normal[self.feature_names].values.astype(np.float32)
        else:
            X_mat = np.asarray(X_normal, dtype=np.float32)

        input_dim = X_mat.shape[1]

        # 1. Fit scaler on normal training data
        X_scaled = self.scaler.fit_transform(X_mat).astype(np.float32)

        # 2. Train / Val split
        n_samples = len(X_scaled)
        indices = np.random.permutation(n_samples)
        n_val = int(n_samples * val_split)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train_t = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
        X_val_t = torch.tensor(X_scaled[val_idx], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t), batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t), batch_size=batch_size, shuffle=False
        )

        # 3. Instantiate model
        self.model = VAEArchitecture(input_dim=input_dim, latent_dim=self.latent_dim).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        if verbose:
            print(f"Training Deep VAE Sentinel ({input_dim} -> 24 -> 12 -> {self.latent_dim}) on {len(train_idx):,} normal samples...")

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_mse_total = 0.0
            train_kl_total = 0.0
            n_batches = 0

            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()

                recon_x, mu, logvar = self.model(batch_x)

                # Reconstruction loss (MSE)
                mse_loss = nn.functional.mse_loss(recon_x, batch_x, reduction="mean")
                # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

                total_loss = mse_loss + self.beta_kl * kl_loss
                total_loss.backward()
                optimizer.step()

                train_mse_total += mse_loss.item()
                train_kl_total += kl_loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            self.model.eval()
            val_mse_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for (val_bx,) in val_loader:
                    val_bx = val_bx.to(self.device)
                    recon_v, _, _ = self.model(val_bx)
                    val_mse_total += nn.functional.mse_loss(recon_v, val_bx, reduction="mean").item()
                    val_batches += 1

            avg_val_mse = val_mse_total / max(1, val_batches)
            if avg_val_mse < best_val_loss:
                best_val_loss = avg_val_mse
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                avg_train_mse = train_mse_total / max(1, n_batches)
                avg_train_kl = train_kl_total / max(1, n_batches)
                print(
                    f"   Epoch {epoch:2d}/{epochs:2d} | "
                    f"Train MSE: {avg_train_mse:.5f} | "
                    f"KL Div: {avg_train_kl:.5f} | "
                    f"Val MSE: {avg_val_mse:.5f}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        # 4. Compute calibration score bounds on training data
        train_scores = self.score_samples(X_mat)
        self.min_score = float(train_scores.min())
        self.max_score = float(train_scores.max())
        # Set 99.5th percentile as anomaly threshold on normal distribution
        self.threshold = float(np.percentile(train_scores, 99.5))

        if verbose:
            print(f"[OK] Deep VAE training complete. Normal score range: [{self.min_score:.4f}, {self.max_score:.4f}], 99.5% Threshold: {self.threshold:.4f}")

        return self

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute reconstruction MSE anomaly scores for samples.
        Higher score = More anomalous (greater divergence from normal manifold).
        """
        if self.model is None:
            raise RuntimeError("DeepVAESentinel must be fitted before scoring.")

        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names].values.astype(np.float32)
        else:
            X_mat = np.asarray(X, dtype=np.float32)

        X_scaled = self.scaler.transform(X_mat).astype(np.float32)

        self.model.eval()
        scores = []
        with torch.no_grad():
            # Process in chunks of 2048 to prevent memory spikes
            chunk_size = 2048
            for i in range(0, len(X_scaled), chunk_size):
                chunk = torch.tensor(X_scaled[i : i + chunk_size], dtype=torch.float32, device=self.device)
                recon, _, _ = self.model(chunk)
                # Per-sample mean squared reconstruction error across features
                sample_mse = torch.mean((chunk - recon) ** 2, dim=1).cpu().numpy()
                scores.append(sample_mse)

        return np.concatenate(scores)

    def get_feature_deltas(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute per-feature squared reconstruction errors: (x_i - x̂_i)^2.
        Used to explain which specific feature failed to reconstruct.
        """
        if self.model is None:
            raise RuntimeError("DeepVAESentinel must be fitted before computing feature deltas.")

        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names].values.astype(np.float32)
        else:
            X_mat = np.asarray(X, dtype=np.float32)

        X_scaled = self.scaler.transform(X_mat).astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            inp = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            recon, _, _ = self.model(inp)
            deltas = ((inp - recon) ** 2).cpu().numpy()

        return deltas

    def save(self, save_dir: Union[Path, str]) -> None:
        """Serialize VAE PyTorch weights and metadata."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": self.model.state_dict() if self.model else None,
                "input_dim": len(self.feature_names),
                "latent_dim": self.latent_dim,
                "beta_kl": self.beta_kl,
                "feature_names": self.feature_names,
                "min_score": self.min_score,
                "max_score": self.max_score,
                "threshold": self.threshold,
            },
            save_path / "vae_sentinel.pt",
        )
        joblib.dump(self.scaler, save_path / "vae_scaler.joblib")
        print(f"Saved Deep VAE Sentinel to: {save_path / 'vae_sentinel.pt'}")

    @classmethod
    def load(cls, save_dir: Union[Path, str]) -> "DeepVAESentinel":
        """Load trained VAE Sentinel from disk."""
        save_path = Path(save_dir)
        checkpoint_path = save_path / "vae_sentinel.pt"
        scaler_path = save_path / "vae_scaler.joblib"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"VAE checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        sentinel = cls(
            feature_names=checkpoint["feature_names"],
            latent_dim=checkpoint["latent_dim"],
            beta_kl=checkpoint.get("beta_kl", 0.005),
        )
        sentinel.min_score = checkpoint.get("min_score", 0.0)
        sentinel.max_score = checkpoint.get("max_score", 1.0)
        sentinel.threshold = checkpoint.get("threshold", 0.5)

        if scaler_path.exists():
            sentinel.scaler = joblib.load(scaler_path)

        sentinel.model = VAEArchitecture(
            input_dim=checkpoint["input_dim"],
            latent_dim=checkpoint["latent_dim"],
        )
        if checkpoint["model_state"] is not None:
            sentinel.model.load_state_dict(checkpoint["model_state"])
        sentinel.model.eval()

        return sentinel
