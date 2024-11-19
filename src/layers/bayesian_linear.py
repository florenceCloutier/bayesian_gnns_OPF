from typing import Optional

import torch
from torch.distributions import Normal
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from torch_geometric.nn.dense.linear import is_uninitialized_parameter


class BayesianLinear(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -3.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        if in_channels > 0:
            self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels))
            self.rho_weight = Parameter(torch.Tensor(out_channels, in_channels))

            self.register_buffer("eps_weight", torch.Tensor(out_channels, in_channels), persistent=False)
            self.register_buffer("prior_weight_mu", torch.Tensor(out_channels, in_channels), persistent=False)
            self.register_buffer(
                "prior_weight_sigma",
                torch.Tensor(out_channels, in_channels),
                persistent=False,
            )
        else:
            self.mu_weight = torch.nn.parameter.UninitializedParameter()
            self.rho_weight = torch.nn.parameter.UninitializedParameter()
            self.register_buffer("eps_weight", None, persistent=False)
            self.register_buffer("prior_weight_mu", None, persistent=False)
            self.register_buffer("prior_weight_sigma", None, persistent=False)
            self._hook = self.register_forward_pre_hook(self.initialize_parameters)

        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer("eps_bias", torch.Tensor(out_channels), persistent=False)
            self.register_buffer("prior_bias_mu", torch.Tensor(out_channels), persistent=False)
            self.register_buffer("prior_bias_sigma", torch.Tensor(out_channels), persistent=False)
        else:
            self.register_parameter("mu_bias", None)
            self.register_buffer("prior_bias_mu", None, persistent=False)
            self.register_buffer("prior_bias_sigma", None, persistent=False)
            self.register_parameter("mu_bias", None)
            self.register_parameter("rho_bias", None)
            self.register_buffer("eps_bias", None, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.in_channels <= 0:
            return
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        # TODO: look into Xavier initialization here
        self.mu_weight.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init, std=0.1)
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))

        weight = self.mu_weight + (sigma_weight * self.eps_weight.data.normal_())

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
        else:
            bias = None

        return F.linear(x, weight, bias)

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if is_uninitialized_parameter(self.mu_weight) and is_uninitialized_parameter(self.rho_weight):
            self.in_channels = input[0].size(-1)

            self.mu_weight.materialize((self.out_channels, self.in_channels))
            self.rho_weight.materialize((self.out_channels, self.in_channels))

            self.register_buffer("eps_weight", torch.Tensor(self.out_channels, self.in_channels), persistent=False)
            self.register_buffer(
                "prior_weight_mu", torch.Tensor(self.out_channels, self.in_channels), persistent=False
            )
            self.register_buffer(
                "prior_weight_sigma",
                torch.Tensor(self.out_channels, self.in_channels),
                persistent=False,
            )

            self.reset_parameters()
        self._hook.remove()
        delattr(self, "_hook")

    def kl_loss(self):
        """Calculate KL divergence between variational posterior and prior"""
        weight_prior = Normal(self.prior_weight_mu, self.prior_weight_sigma)
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        weight_posterior = Normal(self.mu_weight, sigma_weight)
        kl_weight = torch.distributions.kl_divergence(weight_posterior, weight_prior).sum()

        if self.mu_bias is not None:
            bias_prior = Normal(self.prior_bias_mu, self.prior_bias_sigma)
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias_posterior = Normal(self.mu_bias, sigma_bias)
            kl_bias = torch.distributions.kl_divergence(bias_posterior, bias_prior).sum()
        else:
            kl_bias = 0

        return kl_weight + kl_bias
