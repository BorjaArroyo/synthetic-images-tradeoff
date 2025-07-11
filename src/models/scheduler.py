"""Scheduler for GAN training matching vae-flow-dp logic."""
import math


class GapAwareDStepScheduler:
    """Gap-aware discriminator step scheduler for GAN training.
    
    This scheduler adjusts the number of discriminator steps based on the loss gap
    to maintain training balance between generator and discriminator.
    """
    
    def __init__(
        self,
        d_steps_rate_init: float = 1.0,
        grace: int = 5,
        thresh: float = 0.6,
        beta: float = 0.9,
        max_d_steps: int = 50,
        target_loss: float = math.log(4)
    ):
        """Initialize the scheduler.
        
        Args:
            d_steps_rate_init: Initial discriminator steps rate
            grace: Grace period before applying scheduling
            thresh: Threshold for loss difference
            beta: Exponential moving average factor
            max_d_steps: Maximum discriminator steps
            target_loss: Target loss value
        """
        self.d_steps_rate_init = d_steps_rate_init
        self.grace = grace
        self.thresh = thresh
        self.beta = beta
        self.max_d_steps = max_d_steps
        self.target_loss = target_loss
        
        # Internal state
        self.d_loss_ema = None
        self.d_steps_since_g = 0
        self.d_steps_rate = d_steps_rate_init
        self.total_d_steps = 0
        
    def d_step(self, d_loss: float) -> None:
        """Record a discriminator step and update internal state.
        
        Args:
            d_loss: Current discriminator loss
        """
        self.total_d_steps += 1
        self.d_steps_since_g += 1
        
        # Update exponential moving average of discriminator loss
        if self.d_loss_ema is None:
            self.d_loss_ema = d_loss
        else:
            self.d_loss_ema = self.beta * self.d_loss_ema + (1 - self.beta) * d_loss
    
    def is_g_step_time(self, total_d_steps: int) -> bool:
        """Determine if it's time to take a generator step.
        
        Args:
            total_d_steps: Total number of discriminator steps taken
            
        Returns:
            True if generator should take a step
        """
        if total_d_steps <= self.grace:
            return True
            
        if self.d_loss_ema is None:
            return True
            
        # Check if we've reached the maximum discriminator steps
        if self.d_steps_since_g >= self.max_d_steps:
            self._reset_for_g_step()
            return True
            
        # Check if discriminator loss is close to target
        loss_diff = abs(self.d_loss_ema - self.target_loss)
        if loss_diff < self.thresh:
            self._reset_for_g_step()
            return True
            
        return False
    
    def _reset_for_g_step(self) -> None:
        """Reset state when taking a generator step."""
        self.d_steps_since_g = 0 