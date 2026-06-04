import torch
from tqdm import tqdm
import torch.nn.functional as F
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler


class DMPlug(Algo):
    
    '''
    DMPlug algorithm implemented in EDM framework.
    '''
    
    def __init__(self, 
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale=1.0,
                 sde=False,
                 iteration=5000,
                 lr=0.1,
                 weight_decay=0.0,
                 loss_scaling='residual',
                 grad_check_interval=0):
        super(DMPlug, self).__init__(net, forward_op)
        self.net.eval().requires_grad_(False)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.guidance_scale = guidance_scale
        self.sde = sde
        self.iteration = iteration
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_scaling = loss_scaling
        self.grad_check_interval = grad_check_interval
        if self.loss_scaling not in {'residual', 'mse', 'none'}:
            raise ValueError("loss_scaling must be one of {'residual', 'mse', 'none'}")

    def _measurement_numel(self, observation):
        def numel(data):
            if torch.is_tensor(data):
                return data.numel()
            size = getattr(data, "size", None)
            return size() if callable(size) else size

        if torch.is_tensor(observation):
            return observation[0].numel() if observation.ndim > 0 else observation.numel()
        if isinstance(observation, (list, tuple)):
            total = 0
            for item in observation:
                data = getattr(item, "data", item)
                total += numel(data)
            return total
        data = getattr(observation, "data", None)
        if data is not None:
            return numel(data)
        raise TypeError("Cannot infer measurement size for loss_scaling='mse'.")
    
    def _compare_gradient_methods(self, x_initial, observation, loss_scale, gradient):
        """
        Sanity check: Compare current gradient method vs. direct loss.backward() on the same scale.

        Returns:
            Dict with comparison metrics
        """
        # Method 1: Current approach (already computed)
        grad_norm_method1 = torch.norm(x_initial.grad).item() if x_initial.grad is not None else 0.0
        loss_method1 = torch.sqrt(loss_scale).item()
        
        try:
            x_initial_copy = x_initial.detach().clone().requires_grad_(True)
            sampler = DiffusionSampler(self.scheduler)
            denoised_copy = sampler.sample(self.net, x_initial_copy, SDE=self.sde, verbose=False)
            measurement_pred = self.forward_op.forward(denoised_copy)

            if self.loss_scaling == 'residual':
                loss_mse = torch.sqrt(self.forward_op.loss_m(measurement_pred, observation).sum())
            elif self.loss_scaling == 'mse':
                loss_mse = F.mse_loss(measurement_pred, observation)
            else:
                loss_mse = self.forward_op.loss_m(measurement_pred, observation).sum()

            loss_mse.backward()
            x_initial_grad_method2 = x_initial_copy.grad
            if x_initial_grad_method2 is None:
                raise RuntimeError('Direct backward did not produce x_initial gradient.')

            grad_norm_method2 = torch.norm(x_initial_grad_method2).item()
            loss_method2 = loss_mse.item()

            # Compare the two x_initial gradients on the same scale.
            manual_grad = x_initial.grad
            if self.guidance_scale != 0:
                manual_grad = manual_grad / self.guidance_scale
            grad_diff = torch.norm(manual_grad - x_initial_grad_method2) / (torch.norm(manual_grad) + 1e-8)
            grad_diff = grad_diff.item()

        except Exception as e:
            return {
                'method': 'error',
                'error': f'{type(e).__name__}: {e}',
                'grad_norm_method1': grad_norm_method1,
                'loss_method1': loss_method1,
            }

        return {
            'grad_norm_method1': grad_norm_method1,
            'grad_norm_method2': grad_norm_method2,
            'loss_method1': loss_method1,
            'loss_method2': loss_method2,
            'grad_diff': grad_diff,
            'loss_diff': abs(loss_method1 - loss_method2),
        }
        
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            if not torch.is_tensor(observation):
                raise ValueError("DMPlug num_samples > 1 requires tensor observations.")
            observation = observation.repeat(num_samples, *([1] * (observation.ndim - 1)))
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_initial.requires_grad = True
        
        sampler = DiffusionSampler(self.scheduler)
        pbar = tqdm(range(self.iteration))
        
        optimizer = torch.optim.AdamW([x_initial], lr=self.lr, weight_decay=self.weight_decay)
        
        for iteration in pbar:
            optimizer.zero_grad()
            denoised = sampler.sample(self.net, x_initial, SDE=self.sde, verbose=False)

            gradient, loss_scale = self.forward_op.gradient(denoised, observation, return_loss=True)

            x_initial_grad = torch.autograd.grad(
                outputs=denoised,
                inputs=x_initial,
                grad_outputs=gradient,
            )[0]
            if self.loss_scaling == 'residual':
                x_initial_grad = x_initial_grad * 0.5 / torch.sqrt(loss_scale).clamp_min(1e-8)
            elif self.loss_scaling == 'mse':
                x_initial_grad = x_initial_grad / self._measurement_numel(observation)
            x_initial.grad = x_initial_grad * self.guidance_scale
            
            # Gradient sanity check before updating x_initial.
            desc = f'Iteration {iteration + 1}/{self.iteration}. Data fitting loss: {torch.sqrt(loss_scale)}, x_initial.grad norm: {torch.norm(x_initial.grad, 2).mean().item()}'
            # if self.grad_check_interval > 0 and (iteration + 1) % self.grad_check_interval == 0:
            #     comparison = self._compare_gradient_methods(x_initial, observation, loss_scale, gradient)
            #     if comparison.get('method') != 'error':
            #         desc += f" | Grad diff: {comparison['grad_diff']:.2e}, Loss diff: {comparison['loss_diff']:.2e}"
            #     else:
            #         desc += f" | Grad check error: {comparison['error']}"
            optimizer.step()
            pbar.set_description(desc)

        with torch.no_grad():
            denoised = sampler.sample(self.net, x_initial, SDE=self.sde, verbose=False)
        return denoised
