from functools import partial
from typing import Callable

import torch


def proximal_bregman_objective(
    x: torch.Tensor,
    y: torch.Tensor,
    h: torch.nn.Module,
    h_s: torch.nn.Module,
    cost: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    l2_penalty: float,
):
    """
    Compute the proximal Bregman objective for a given function h and its original function h_s

    Args:
        x: input tensor
        y: target tensor
        h: function to be optimized
        h_s: original function
        l2_penalty: L2 penalty

    Returns:
        proximal Bregman objective (torch.Tensor)
    """
    y_hat = h(x)
    y_hat_s = h_s(x)

    pbo = cost(y_hat, y)
    pbo -= cost(y_hat_s, y)
    pbo -= pbo_grad_term(y, y_hat, y_hat_s, cost)
    pbo -= l2_penalty / 2 * weight_l2_distance_squared(h, h_s)
    return pbo


def pbo_grad_term(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    y_hat_s: torch.Tensor,
    cost: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    """
    Compute the gradient term of the proximal Bregman objective

    Args:
        y: target tensor
        y_hat: output of the function to be optimized
        y_hat_s: output of the original function

    Returns:
        gradient term (torch.Tensor)
    """
    # compute $\nabla_{y_hat_s} L(y_hat_s, y)^T(y_hat-y_hat_s)$
    v = y_hat - y_hat_s  # (N, C)
    grad_fn = torch.func.grad(cost)
    grad = grad_fn(y_hat_s, y)  # (N, C)
    return (grad * v).sum()


def weight_l2_distance_squared(
    h: torch.nn.Module, h_s: torch.nn.Module
) -> torch.Tensor:
    """
    Compute the squared L2 distance between the weights of two functions

    Args:
        h: function to be optimized
        h_s: original function

    Returns:
        squared L2 distance (torch.Tensor)
    """
    return sum(
        (w - w_s).pow(2).sum()
        for w, w_s in zip(h.parameters(), h_s.parameters())
    )


if __name__ == "__main__":
    # test
    x = torch.rand(10, 5)
    y = torch.randint(2, (10,))
    h = torch.nn.Linear(5, 2)
    h_s = torch.nn.Linear(5, 2)
    cost = torch.nn.CrossEntropyLoss()
    l2_penalty = 0.1
    pbo = proximal_bregman_objective(x, y, h, h_s, cost, l2_penalty)
    pbo.backward()
    print(h.weight.grad)
