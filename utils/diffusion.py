from tqdm import trange
import torch
import numpy as np
from utils.scheduler import Scheduler

class DiffusionSampler:
    """
        Diffusion sampler for reverse SDE or PF-ODE
    """

    def __init__(self, scheduler, solver='euler'):
        """
            Initializes the diffusion sampler with the given scheduler and solver.

            Parameters:
                scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
                solver (str): Solver method ('euler' or 'ddim').
        """
        super().__init__()
        self.scheduler = scheduler
        self.solver = solver
        if solver not in {'euler', 'ddim'}:
            raise NotImplementedError(f'Unknown diffusion solver: {solver}')

    def sample(self, model, x_start, SDE=False, verbose=False):
        """
            Samples from the diffusion process using the specified model.

            Parameters:
                model (DiffusionModel): Diffusion model supports 'score' and 'tweedie'
                x_start (torch.Tensor): Initial state.
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        if self.solver == 'euler':
            return self._euler(model, x_start, SDE, verbose)
        if self.solver == 'ddim':
            if SDE:
                raise ValueError("DDIM solver is deterministic; use SDE=False.")
            return self._ddim(model, x_start, verbose)
        raise NotImplementedError

    def score(self, model, x, sigma):
        """
            Computes the score function for the given model.

            Parameters:
                model (DiffusionModel): Diffusion model.
                x (torch.Tensor): Input tensor.
                sigma (float): Sigma value.

            Returns:
                torch.Tensor: The computed score.
        """
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (d - x) / sigma**2
    
    def _euler(self, model, x_start, SDE=False, verbose=False):
        """
            Euler's method for sampling from the diffusion process.
        """
        pbar = trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)

        x = x_start
        for step in pbar:
            sigma, factor, scaling_factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step], self.scheduler.scaling_factor[step]
            score = self.score(model, x / self.scheduler.scaling_steps[step], sigma) / self.scheduler.scaling_steps[step]
            if SDE:
                epsilon = torch.randn_like(x)
                x = x * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x * scaling_factor + factor * score * 0.5 
        return x

    def _ddim(self, model, x_start, verbose=False):
        """
            Deterministic DDIM-style update in scheduler coordinates.

            The sampler state follows x_t = s_t * (x_0 + sigma_t * eps).
            The model is expected to return an x_0 estimate when called as
            model(x_t / s_t, sigma_t), matching the preconditioned models used
            by the Euler sampler above.
        """
        pbar = trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)

        x = x_start
        for step in pbar:
            sigma = self.scheduler.sigma_steps[step]
            sigma_next = self.scheduler.sigma_steps[step + 1]
            scaling = self.scheduler.scaling_steps[step]
            scaling_next = self.scheduler.scaling_steps[step + 1]

            sigma_tensor = torch.as_tensor(sigma).to(x.device)
            x_scaled = x / scaling
            denoised = model(x_scaled, sigma_tensor)
            eps = (x_scaled - denoised) / sigma
            x = scaling_next * (denoised + sigma_next * eps)
        return x

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.scheduler.sigma_max
        return x_start
    
