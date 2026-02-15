import torch
from abc import ABC, abstractmethod
from typing import Optional


class DecaySchedule(ABC):
    """Base class for decay schedules."""

    def __init__(self, initial_value: float, final_value: float, total_steps: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.current_step = 0

    @abstractmethod
    def get_value(self, step: Optional[int] = None) -> float:
        """Get the current value based on the step."""
        pass

    def step(self) -> float:
        """Increment step and return current value."""
        self.current_step += 1
        return self.get_value()


class LinearDecaySchedule(DecaySchedule):
    """Linear decay from initial_value to final_value over total_steps."""

    def get_value(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.current_step

        if step >= self.total_steps:
            return self.final_value

        progress = step / self.total_steps
        return self.initial_value + progress * (self.final_value - self.initial_value)


class ExponentialDecaySchedule(DecaySchedule):
    """Exponential decay from initial_value to final_value over total_steps."""

    def __init__(self, initial_value: float, final_value: float, total_steps: int,
                 decay_rate: Optional[float] = None):
        super().__init__(initial_value, final_value, total_steps)

        if decay_rate is None:
            # Calculate decay rate to reach final_value at total_steps
            self.decay_rate = (final_value / initial_value) ** (1.0 / total_steps)
        else:
            self.decay_rate = decay_rate

    def get_value(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.current_step

        if step >= self.total_steps:
            return self.final_value

        return self.initial_value * (self.decay_rate ** step)


class LearningRateScheduler:
    """Scheduler for optimizer learning rates with decay."""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 schedule: DecaySchedule,
                 param_group_idx: int = 0):
        self.optimizer = optimizer
        self.schedule = schedule
        self.param_group_idx = param_group_idx

    def step(self):
        """Update the learning rate based on the schedule."""
        new_lr = self.schedule.step()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def get_lr(self) -> list[float]:
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class EntropyScheduler:
    """Scheduler for entropy coefficient with decay."""

    def __init__(self, schedule: DecaySchedule):
        self.schedule = schedule

    def step(self) -> float:
        """Update and return the entropy coefficient."""
        return self.schedule.step()

    def get_value(self) -> float:
        """Get current entropy coefficient."""
        return self.schedule.get_value()
