""" Influence function and EKFAC implementation

Adapted from https://github.com/nrimsky/InfluenceFunctions.
"""

import torch
from abc import ABC, abstractmethod
from typing import List, Callable
import einops


class InfluenceCalculable(ABC):
    @abstractmethod
    def get_a_l_minus_1(self):
        # Return the input to the linear layer
        pass

    @abstractmethod
    def get_d_s_l(self):
        # Return the gradient of the loss wrt the output of the linear layer
        pass

    @abstractmethod
    def get_dims(self):
        # Return the dimensions of the weights - (output_dim, input_dim)
        pass

    @abstractmethod
    def get_d_w_l(self):
        # Return the gradient of the loss wrt the weights
        pass


def get_ekfac_factors_and_pseudo_grads(
    model: torch.nn.Module,
    dataset: torch.utils.data.DataLoader,
    mlp_blocks: List[InfluenceCalculable],
    device: torch.device,
    loss_fn: Callable,
):
    kfac_input_covs = [
        torch.zeros((b.get_dims()[1] + 1, b.get_dims()[1] + 1)).to(device)
        for b in mlp_blocks
    ]
    kfac_grad_covs = [
        torch.zeros((b.get_dims()[0], b.get_dims()[0])).to(device)
        for b in mlp_blocks
    ]
    grads = [[] for _ in range(len(mlp_blocks))]
    tot = 0
    for data, target in dataset:
        model.zero_grad()
        data = data.to(device)
        target = target.to(device)
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)
        output = model(data)
        loss = loss_fn(output, target)
        for i, block in enumerate(mlp_blocks):
            a_l_minus_1 = block.get_a_l_minus_1()
            input_covs = torch.einsum(
                "...ti,...tj->tij", a_l_minus_1, a_l_minus_1
            )
            kfac_input_covs[i] += input_covs.mean(dim=0)
        loss.backward()
        for i, block in enumerate(mlp_blocks):
            d_s_l = block.get_d_s_l()
            grad_cov = torch.einsum("...ti,...tj->tij", d_s_l, d_s_l)
            kfac_grad_covs[i] += grad_cov.mean(dim=0)
            grads[i].append(block.get_d_w_l())
        tot += 1
    kfac_input_covs = [A / tot for A in kfac_input_covs]
    kfac_grad_covs = [S / tot for S in kfac_grad_covs]
    return kfac_input_covs, kfac_grad_covs, grads


def get_grads(
    model,
    dataset,
    mlp_blocks: List[InfluenceCalculable],
    device,
    loss_fn: Callable,
):
    grads = [[] for _ in range(len(mlp_blocks))]
    for data, target in dataset:
        model.zero_grad()
        data = data.to(device)
        target = target.to(device)
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        for i, block in enumerate(mlp_blocks):
            grads[i].append(block.get_d_w_l())
    return grads


def compute_lambda_ii(train_grads, q_a, q_s):
    """Compute Lambda_ii values for a block."""
    n_examples = len(train_grads)
    squared_projections_sum = 0.0
    for j in range(n_examples):
        dtheta = train_grads[j]
        result = (q_s @ dtheta @ q_a.T).view(-1)
        squared_projections_sum += result**2
    lambda_ii_avg = squared_projections_sum / n_examples
    return lambda_ii_avg


def get_ekfac_ihvp(
    kfac_input_covs, kfac_grad_covs, pseudo_grads, search_grads, damping=0.1
):
    """Compute EK-FAC inverse Hessian-vector products."""
    ihvp = []
    for i in range(len(search_grads)):
        V = search_grads[i]
        stacked = torch.stack(V)
        # Performing eigendecompositions on the input and gradient covariance matrices
        q_a, _, q_a_t = torch.svd(kfac_input_covs[i])
        q_s, _, q_s_t = torch.svd(kfac_grad_covs[i])
        lambda_ii = compute_lambda_ii(pseudo_grads[i], q_a, q_s)
        ekfacDiag_damped_inv = 1.0 / (
            lambda_ii + torch.mean(lambda_ii) * damping
        )
        ekfacDiag_damped_inv = ekfacDiag_damped_inv.reshape(
            (stacked.shape[-2], stacked.shape[-1])
        )
        intermediate_result = torch.einsum("bij,jk->bik", stacked, q_a_t)
        intermediate_result = torch.einsum(
            "ji,bik->bjk", q_s, intermediate_result
        )
        result = intermediate_result / ekfacDiag_damped_inv.unsqueeze(0)
        ihvp_component = torch.einsum("bij,jk->bik", result, q_a)
        ihvp_component = torch.einsum("ji,bik->bjk", q_s_t, ihvp_component)
        # flattening the result except for the batch dimension
        ihvp_component = einops.rearrange(ihvp_component, "b j k -> b (j k)")
        ihvp.append(ihvp_component)
    # Concatenating the results across blocks to get the final ihvp
    return torch.cat(ihvp, dim=-1)


def get_query_grad(
    model, query, mlp_blocks: List[InfluenceCalculable], device, loss_fn
):
    grads = get_grads(model, [query], mlp_blocks, device, loss_fn)
    return torch.cat([q[0].view(-1) for q in grads])


def get_influences(ihvp, query_grad):
    """
    Compute influences using precomputed iHVP and query_grad
    """
    return -1 * torch.einsum("ij,j->i", ihvp, query_grad)


def influence(
    model,
    mlp_blocks: List[InfluenceCalculable],
    queries,
    gradient_fitting_data,
    search_data,
    topk,
    device,
    loss_fn,
):
    kfac_input_covs, kfac_grad_covs, pseudo_grads = (
        get_ekfac_factors_and_pseudo_grads(
            model, gradient_fitting_data, mlp_blocks, device, loss_fn
        )
    )

    search_grads = get_grads(model, search_data, mlp_blocks, device, loss_fn)

    ihvp = get_ekfac_ihvp(
        kfac_input_covs, kfac_grad_covs, pseudo_grads, search_grads
    )

    all_top_training_samples = []
    all_top_influences = []

    for query in queries:
        query_grad = get_query_grad(model, query, mlp_blocks, device, loss_fn)
        top_influences = get_influences(ihvp, query_grad)
        top_influences, top_samples = torch.topk(top_influences, topk)
        all_top_training_samples.append(top_samples)
        all_top_influences.append(top_influences)

    return all_top_training_samples, all_top_influences
