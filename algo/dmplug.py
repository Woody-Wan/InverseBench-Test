import copy
import json
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
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
                 solver='euler',
                 grad_check_interval=0,
                 trace_interval=0):
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
        self.solver = solver
        self.grad_check_interval = grad_check_interval
        self.trace_interval = trace_interval
        if self.loss_scaling not in {'residual', 'mse', 'none'}:
            raise ValueError("loss_scaling must be one of {'residual', 'mse', 'none'}")

    def _scheduler_stats(self):
        return {
            'num_steps': self.scheduler.num_steps,
            'sigma_max_state': self.scheduler.sigma_max,
            'first_sigma': float(self.scheduler.sigma_steps[0]),
            'second_sigma': float(self.scheduler.sigma_steps[1]) if len(self.scheduler.sigma_steps) > 1 else None,
            'last_sigma': float(self.scheduler.sigma_steps[-1]),
            'first_factor': float(self.scheduler.factor_steps[0]),
            'first_scaling_factor': float(self.scheduler.scaling_factor[0]),
            'solver': self.solver,
        }

    def _measurement_numel(self, observation):
        def numel(data):
            if torch.is_tensor(data):
                return data.numel()
            size = getattr(data, "size", None)
            return size() if callable(size) else size

        if torch.is_tensor(observation):
            return observation.numel()
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
        loss_method1 = self._objective_value(loss_scale, observation)
        
        try:
            x_initial_copy = x_initial.detach().clone().requires_grad_(True)
            sampler = DiffusionSampler(self.scheduler, solver=self.solver)
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

    def _objective_value(self, loss_scale, observation):
        if self.loss_scaling == 'residual':
            return torch.sqrt(loss_scale.detach()).mean().item()
        if self.loss_scaling == 'mse':
            return (loss_scale.detach() / self._measurement_numel(observation)).mean().item()
        return loss_scale.detach().mean().item()

    def _as_jsonable(self, value):
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        return value

    def _jpg_visual(self, recon):
        if recon.ndim == 3:
            recon = recon.unsqueeze(0)
        channels = recon.shape[1]
        if channels == 1:
            recon = recon.repeat(1, 3, 1, 1)
        elif channels == 2:
            recon = torch.linalg.norm(recon, dim=1, keepdim=True).repeat(1, 3, 1, 1)
        elif channels > 3:
            recon = recon[:, :3]
        return recon.clamp(0, 1)

    def _comparison_visual(self, tensor):
        return self._jpg_visual(tensor)

    def _save_loss_plot(self, trace_dir, history):
        loss_history = history.get('loss', [])
        if not loss_history:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        iterations = [entry['iteration'] for entry in loss_history]
        objectives = [entry['objective'] for entry in loss_history]

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.plot(iterations, objectives, linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective')
        ax.set_title('DMPlug optimization history')
        ax.grid(True, alpha=0.3)
        if all(value > 0 for value in objectives):
            ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(trace_dir, 'loss_history.jpg'))
        plt.close(fig)

    def _save_comparison_figure(self, trace_dir, iteration, recon_unnorm_cpu, target):
        if target is None:
            return
        target_cpu = self.forward_op.unnormalize(target.detach()).cpu()
        recon_visual = self._comparison_visual(recon_unnorm_cpu)
        target_visual = self._comparison_visual(target_cpu)
        if target_visual.shape[0] != recon_visual.shape[0]:
            target_visual = target_visual.repeat(recon_visual.shape[0], 1, 1, 1)

        diff_visual = (recon_visual - target_visual).abs()
        diff_max = diff_visual.flatten(start_dim=1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
        diff_visual = diff_visual / diff_max

        comparison = torch.stack([target_visual, recon_visual, diff_visual], dim=1).flatten(0, 1)
        save_image(
            comparison,
            os.path.join(trace_dir, f'comparison_iter_{iteration:06d}.jpg'),
            nrow=3,
        )

    def _tensor_stats(self, tensor):
        if not torch.is_tensor(tensor):
            return None
        tensor = tensor.detach()
        return {
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
        }

    def _save_trace(self, trace_dir, iteration, x_initial, denoised, observation, target, evaluator, history):
        if trace_dir is None:
            return
        os.makedirs(trace_dir, exist_ok=True)
        recon_cpu = denoised.detach().cpu()
        recon_unnorm_cpu = self.forward_op.unnormalize(denoised.detach()).cpu()
        trace = {
            'iteration': iteration,
            'recon': recon_cpu,
            'recon_unnormalized': recon_unnorm_cpu,
            'x_initial': x_initial.detach().cpu(),
            'stats': {
                'recon': self._tensor_stats(denoised),
                'recon_unnormalized': self._tensor_stats(recon_unnorm_cpu),
                'observation': self._tensor_stats(observation),
            },
        }
        torch.save(trace, os.path.join(trace_dir, f'recon_iter_{iteration:06d}.pt'))
        save_image(
            self._jpg_visual(recon_unnorm_cpu),
            os.path.join(trace_dir, f'recon_iter_{iteration:06d}.jpg'),
        )
        self._save_comparison_figure(trace_dir, iteration, recon_unnorm_cpu, target)

        if evaluator is not None and target is not None:
            metric_state = copy.deepcopy(evaluator.metric_state)
            try:
                with torch.no_grad():
                    target_cpu = self.forward_op.unnormalize(target.detach()).cpu()
                    obs = observation.detach().cpu() if torch.is_tensor(observation) else observation
                    metric_dict = evaluator(pred=recon_unnorm_cpu, target=target_cpu, observation=obs)
                history['eval'].append({
                    'iteration': iteration,
                    **{k: self._as_jsonable(v) for k, v in metric_dict.items()},
                })
            finally:
                evaluator.metric_state = metric_state

        torch.save(history, os.path.join(trace_dir, 'history.pt'))
        with open(os.path.join(trace_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        self._save_loss_plot(trace_dir, history)
        
    def inference(self, observation, num_samples=1, **kwargs):
        target = kwargs.get('target')
        evaluator = kwargs.get('evaluator')
        trace_dir = kwargs.get('trace_dir')
        device = self.forward_op.device
        if num_samples > 1:
            if not torch.is_tensor(observation):
                raise ValueError("DMPlug num_samples > 1 requires tensor observations.")
            observation = observation.repeat(num_samples, *([1] * (observation.ndim - 1)))
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_initial.requires_grad = True
        
        sampler = DiffusionSampler(self.scheduler, solver=self.solver)
        pbar = tqdm(range(self.iteration))
        
        optimizer = torch.optim.AdamW([x_initial], lr=self.lr, weight_decay=self.weight_decay)
        history = {'loss': [], 'eval': [], 'scheduler': self._scheduler_stats()}
        last_trace_iteration = 0
        
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
            display_loss = self._objective_value(loss_scale, observation)
            grad_norm = torch.norm(x_initial.grad.detach()).item()
            history['loss'].append({
                'iteration': iteration + 1,
                'objective': display_loss,
                'raw_loss': loss_scale.detach().mean().item(),
                'x_initial_grad_norm': grad_norm,
                'recon': self._tensor_stats(denoised),
                'observation': self._tensor_stats(observation),
            })
            desc = f'Iteration {iteration + 1}/{self.iteration}. Data fitting loss: {display_loss}, x_initial.grad norm: {grad_norm}'
            if self.grad_check_interval > 0 and (iteration + 1) % self.grad_check_interval == 0:
                comparison = self._compare_gradient_methods(x_initial, observation, loss_scale, gradient)
                history['loss'][-1]['grad_check'] = comparison
                if comparison.get('method') != 'error':
                    desc += f" | Grad diff: {comparison['grad_diff']:.2e}, Loss diff: {comparison['loss_diff']:.2e}"
                else:
                    desc += f" | Grad check error: {comparison['error']}"
            optimizer.step()
            if self.trace_interval > 0 and (iteration + 1) % self.trace_interval == 0:
                with torch.no_grad():
                    trace_denoised = sampler.sample(self.net, x_initial, SDE=self.sde, verbose=False)
                self._save_trace(trace_dir, iteration + 1, x_initial, trace_denoised, observation, target, evaluator, history)
                last_trace_iteration = iteration + 1
            pbar.set_description(desc)

        with torch.no_grad():
            denoised = sampler.sample(self.net, x_initial, SDE=self.sde, verbose=False)
        if trace_dir is not None and last_trace_iteration != self.iteration:
            self._save_trace(trace_dir, self.iteration, x_initial, denoised, observation, target, evaluator, history)
        return denoised
