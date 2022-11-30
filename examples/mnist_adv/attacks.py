import torch
from typing import Union, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import eagerpy as ep

from foolbox.types import Bounds
from foolbox.models.base import Model
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.distances import linf
from foolbox.attacks.base import FixedEpsilonAttack, T, raise_if_kwargs, get_criterion#, verify_input_bounds
from foolbox.attacks.gradient_descent_base import LinfBaseGradientDescent


def verify_input_bounds(input: ep.Tensor, model: Model) -> None:
    # verify that input to the attack lies within model's input bounds
    assert input.min().item() >= model.bounds.lower
    assert input.max().item() <= model.bounds.upper


class RandomLInfStep(FixedEpsilonAttack, ABC):
    distance = linf
    
    def __init__(self):
        super().__init__()
        self.rel_stepsize = 1.
        
    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        
        if hasattr(criterion, "target_classes"):
            raise ValueError("unsupported criterion")
        
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        verify_input_bounds(x0, model)

        stepsize = self.rel_stepsize * epsilon
        
        x = x0

        rand_pert = torch.randint_like(input=x.raw, low=0, high=2)*2-1
        
        x = x + ep.astensor(rand_pert)*stepsize
        x = self.project(x, x0, epsilon)
        x = ep.clip(x, *model.bounds)

        return restore_type(x)
    
    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.clip(x - x0, -epsilon, epsilon)
    
    
class LinfRandomSearch(FixedEpsilonAttack):
    distance = linf
    
    def __init__(self):
        super().__init__()

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        verify_input_bounds(x0, model)
        
        if isinstance(criterion_, Misclassification):
            classes = criterion_.labels
        elif hasattr(criterion_, "target_classes"):
            classes = criterion_.target_classes
        else:
            raise ValueError("unsupported criterion")
            
        x = self.get_random_start(x0, epsilon)
        x = self.project(x, x0, epsilon)
        x = ep.clip(x, *model.bounds)
        
        return restore_type(x)
    
    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.clip(x - x0, -epsilon, epsilon)
    
    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)
    
    
"""
    LinfFastGradientAttack that interprets the epsilons as being smaller or equal to zero.
    This is acieved by using the TargetedMisclassification as criterion and setting the target class
    to the original class. This flips the sign of the gradient descent step (gradient_step_sign)
    in gradient_descent_base.py.
    
    Note that the boolean success array needs to be inverted.
"""
class NegEpsLinfFastGradientAttack(LinfBaseGradientDescent):
    def __init__(self, *, random_start: bool = False):
        super().__init__(
            rel_stepsize=1.0,
            steps=1,
            random_start=random_start,
        )

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        if not hasattr(criterion, "target_classes"):
            raise ValueError("unsupported criterion")

        return super().run(
            model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs
        )