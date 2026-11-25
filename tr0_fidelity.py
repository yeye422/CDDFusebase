"""Static identity-task assignment for fidelity-constrained complementary learning."""

import math

import torch

from tr0_complementary import ComplementaryViewGenerator


def task_assignments(config, fpcl):
    fraction = fpcl["identity_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("The identity fraction must lie in (0, 1)")
    first = config["reconstruction_epochs"] + 1
    seeds = {config["proxy_seed"] + epoch: config["train_count"]
             for epoch in range(first, first + config["proxy_epochs"])}
    seeds[config["proxy_validation_seed"]] = config["screen_count"]
    assignments = {}
    for seed, count in seeds.items():
        budget = fraction * count
        if not math.isclose(budget, round(budget), abs_tol=1e-9):
            raise ValueError("The fixed identity fraction must give an integer sample budget")
        rng = torch.Generator().manual_seed(fpcl["task_seed"] + seed)
        assignments[str(seed)] = dict(samples=count, identity_positions=sorted(
            torch.randperm(count, generator=rng)[:round(budget)].tolist()))
    return assignments


class FidelityViewGenerator:
    def __init__(self, view_settings, assignments):
        self.complementary = ComplementaryViewGenerator(**view_settings)
        self.tasks = {}
        for seed, plan in assignments.items():
            mask = torch.zeros(plan["samples"], dtype=torch.bool)
            mask[plan["identity_positions"]] = True
            self.tasks[int(seed)] = mask
        self.generator = None
        self.task_mask = None
        self.position = 0

    @torch.no_grad()
    def __call__(self, optical, generator):
        if generator is not self.generator:
            if self.task_mask is not None and self.position != len(self.task_mask):
                raise ValueError("The prior loader did not consume its fixed proxy task list")
            self.task_mask = self.tasks[generator.initial_seed()]
            self.position = 0
            self.generator = generator
        end = self.position + len(optical)
        if end > len(self.task_mask):
            raise ValueError("The loader exceeded its fixed proxy task list")
        identity = self.task_mask[self.position:end]
        self.position = end
        first, second, _, stats = self.complementary(optical, generator)
        mask = identity.to(optical.device)[:, None, None, None]
        stats = {f"sampled_view_{key}": value for key, value in stats.items()}
        stats["identity_fraction"] = identity.float().mean()
        return torch.where(mask, optical, first), torch.where(mask, optical, second), None, stats
