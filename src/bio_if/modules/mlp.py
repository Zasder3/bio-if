""" Influence function and EKFAC implementation

Adapted from https://github.com/nrimsky/InfluenceFunctions.
"""

import torch

from bio_if.modules.influence import InfluenceCalculable


class MLPBlock(InfluenceCalculable, torch.nn.Module):
    def __init__(self, input_dim, output_dim, use_relu=True):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.input = None
        self.use_relu = use_relu
        self.d_s_l = None
        self.d_w_l = None

        # Save gradient of loss wrt output of linear layer (Ds_l, where s_l = self.linear(a_l_minus_1))
        def hook_fn(module, grad_input, grad_output):
            self.d_s_l = grad_output[0]

        self.linear.register_full_backward_hook(hook_fn)

    def forward(self, x):
        self.input = x
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        return x

    def get_a_l_minus_1(self):
        # Return the input to the linear layer as a homogenous vector
        return (
            torch.cat(
                [self.input, torch.ones((self.input.shape[0], 1))],
                dim=-1,
            )
            .clone()
            .detach()
        )

    def get_d_s_l(self):
        # Return the gradient of the loss wrt the output of the linear layer
        return self.d_s_l.clone().detach()

    def get_dims(self):
        # Return the dimensions of the weights - (output_dim, input_dim)
        return self.linear.weight.shape

    def get_d_w_l(self):
        # Return the gradient of the loss wrt the weights
        w_grad = self.linear.weight.grad
        b_grad = self.linear.bias.grad.unsqueeze(-1)
        full_grad = torch.cat([w_grad, b_grad], dim=-1)
        return full_grad.clone().detach()
