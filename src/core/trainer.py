"""Training utilities matching vae-flow-dp logic."""
import torch
import torch.nn.functional as F
from torch.autograd import grad
from tqdm import tqdm
from typing import Optional, Callable, Any
import mlflow

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False


def train_vae(
    model,
    trainloader,
    optimizer,
    device: torch.device,
    epochs: int,
    sigma: float = 0.0,
    log_metrics: bool = True
) -> None:
    """Train the VAE model matching original logic.

    Args:
        model: The VAE model
        trainloader: DataLoader for training data
        optimizer: Optimizer for the model
        device: Device to run the model on (CPU or GPU)
        epochs: Number of epochs to train
        sigma: Noise multiplier for differential privacy (0.0 disables DP)
        log_metrics: Whether to log metrics to MLflow
    """
    dp_enabled = sigma > 0 and OPACUS_AVAILABLE
    
    if dp_enabled:
        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, trainloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=sigma,
            max_grad_norm=1.0,
        )
        delta = 1 / len(trainloader)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for x, y in tqdm(trainloader, desc=f"Epoch {epoch + 1}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward pass
            x_recon, z_mean, z_logvar = model(x, y)
            
            # Compute losses using original logic
            recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
            kl_loss = (
                -0.5
                * torch.sum(
                    1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=[1]
                ).mean()
            )
            
            # Beta scheduling like original
            beta = min(1.0, epoch / 10)
            loss = recon_loss + beta * kl_loss

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping (only when DP is disabled)
            if not dp_enabled:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        # Calculate averages
        avg_loss = total_loss / len(trainloader)
        avg_recon = total_recon_loss / len(trainloader)
        avg_kl = total_kl_loss / len(trainloader)
        
        if log_metrics:
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("recon_loss", avg_recon, step=epoch)
            mlflow.log_metric("kl_loss", avg_kl, step=epoch)
            
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}"
        )
    
    if dp_enabled:
        epsilon = privacy_engine.get_epsilon(delta)
        print(f"DP Accountant: {epsilon:.4f}")
        if log_metrics:
            mlflow.log_metric("epsilon", epsilon)


def compute_gradient_penalty(
    discriminator,
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    lambda_gp: float
) -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP."""
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    labels = labels.to(device)

    d_interpolates = discriminator(interpolates, labels)
    grad_outputs = torch.ones_like(d_interpolates, device=device)

    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # shape: (B, C, H, W)

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    gp = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return gp


def train_gan(
    generator,
    discriminator,
    trainloader,
    optimizer_G,
    optimizer_D,
    device: torch.device,
    epochs: int,
    latent_dim: int,
    scheduler=None,
    sigma: float = 0.0,
    lambda_gp: float = 1.0,
    log_metrics: bool = True
) -> None:
    """Train GAN model matching original logic.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model  
        trainloader: DataLoader for training data
        optimizer_G: Generator optimizer
        optimizer_D: Discriminator optimizer
        device: Device to run on
        epochs: Number of epochs
        latent_dim: Latent dimension
        scheduler: Optional GapAwareDStepScheduler
        sigma: Noise multiplier for DP (0.0 disables DP)
        lambda_gp: Gradient penalty coefficient
        log_metrics: Whether to log metrics to MLflow
    """
    adversarial_loss = torch.nn.BCELoss()
    generator.train()
    discriminator.train()

    # Enable DP only for the discriminator
    dp_enabled = sigma > 0 and OPACUS_AVAILABLE
    if dp_enabled:
        privacy_engine = PrivacyEngine(accountant="rdp")
        discriminator, optimizer_D, trainloader = privacy_engine.make_private(
            module=discriminator,
            optimizer=optimizer_D,
            data_loader=trainloader,
            noise_multiplier=sigma,
            max_grad_norm=1.0,
            poisson_sampling=True,
        )
        delta = 1.0 / len(trainloader.dataset)

    total_d_steps = 0

    for epoch in range(epochs):
        g_losses, d_losses = [], []
        mean_g = 0.0

        for real_imgs, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}"):
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            batch_size = real_imgs.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            z = torch.randn(batch_size, latent_dim, device=device)
            with torch.no_grad():
                fake_imgs = generator(z, labels)

            real_validity = discriminator(real_imgs, labels)
            fake_validity = discriminator(fake_imgs, labels)

            valid = torch.ones_like(real_validity)
            fake = torch.zeros_like(fake_validity)

            loss_real = adversarial_loss(real_validity, valid)
            loss_fake = adversarial_loss(fake_validity, fake)
            
            # Gradient penalty (only when DP is disabled)
            gp = (
                compute_gradient_penalty(
                    discriminator, real_imgs, fake_imgs, labels, device, lambda_gp
                )
                if not dp_enabled
                else 0.0
            )
            d_loss = (loss_real + loss_fake) / 2 + gp

            optimizer_D.zero_grad(set_to_none=True)
            d_loss.backward()
            optimizer_D.step()
            optimizer_D.zero_grad(set_to_none=True)

            d_losses.append(d_loss.item())

            if scheduler:
                scheduler.d_step(d_loss.item())
                total_d_steps += 1
                if not scheduler.is_g_step_time(total_d_steps):
                    continue

            # -----------------
            #  Train Generator
            # -----------------
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z, labels)
            gen_validity = discriminator(gen_imgs, labels)
            valid = torch.ones_like(gen_validity)

            g_loss = adversarial_loss(gen_validity, valid)

            optimizer_G.zero_grad(set_to_none=True)
            g_loss.backward()
            optimizer_G.step()

            g_losses.append(g_loss.item())

        # Logging
        mean_g = sum(g_losses) / len(g_losses) if g_losses else mean_g
        mean_d = sum(d_losses) / len(d_losses)
        
        if log_metrics:
            mlflow.log_metric("g_loss", mean_g, step=epoch)
            mlflow.log_metric("d_loss", mean_d, step=epoch)
            
        print(
            f"[Epoch {epoch + 1}/{epochs}] G_loss: {mean_g:.4f}, D_loss: {mean_d:.4f}"
        )

    # Final epsilon log
    if dp_enabled:
        epsilon = privacy_engine.get_epsilon(delta)
        print(f"Final ε (epsilon): {epsilon:.4f}")
        if log_metrics:
            mlflow.log_metric("epsilon", epsilon)


def train_diffusion(
    model,
    trainloader,
    optimizer,
    device: torch.device,
    epochs: int,
    sigma: float = 0.0,
    log_metrics: bool = True
) -> None:
    """Train the Diffusion model with TeaPearce's proven approach.

    Args:
        model: The DiffusionModel
        trainloader: DataLoader for training data
        optimizer: Optimizer for the model
        device: Device to run the model on (CPU or GPU)
        epochs: Number of epochs to train
        sigma: Noise multiplier for differential privacy (0.0 disables DP)
        log_metrics: Whether to log metrics to MLflow
    """
    dp_enabled = sigma > 0 and OPACUS_AVAILABLE
    
    if dp_enabled:
        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, trainloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=sigma,
            max_grad_norm=1.0,
        )
        delta = 1 / len(trainloader.dataset)
    
    # TeaPearce's learning rate decay schedule
    initial_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(epochs):
        # TeaPearce's linear learning rate decay
        if not dp_enabled:  # Only adjust LR when not using DP
            lr_decay = 1 - epoch / epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr * lr_decay
        
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_data in tqdm(trainloader, desc=f"Epoch {epoch + 1}"):
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                x, y = batch_data
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                # Handle label shape inconsistencies
                if y.dim() > 1:
                    y = y.squeeze()
            else:
                x = batch_data.to(device, non_blocking=True)
                y = None

            # Forward pass - compute diffusion loss
            if hasattr(model, '_module') and hasattr(model._module, 'compute_loss'):
                # Handle opacus wrapped model
                loss = model._module.compute_loss(x, y)
            else:
                # Standard model
                loss = model.compute_loss(x, y)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping (only when DP is disabled)
            if not dp_enabled:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Calculate averages
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if log_metrics:
            mlflow.log_metric("diffusion_loss", avg_loss, step=epoch)
            
        print(f"Epoch [{epoch + 1}/{epochs}], Diffusion Loss: {avg_loss:.4f}")
    
    if dp_enabled:
        epsilon = privacy_engine.get_epsilon(delta)
        print(f"DP Accountant: {epsilon:.4f}")
        if log_metrics:
            mlflow.log_metric("epsilon", epsilon) 