# -*- coding: utf-8 -*-
"""Main module."""
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.distributions import NegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder
from torch import logsumexp
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from .._utils import GumbelSigmoid

torch.backends.cudnn.benchmark = True


# VAE model
class sVAEModule(BaseModuleClass):
    """
    Variational auto-encoder model.

    This is a light modification implementation of the scVI model described in [Lopez18]_
    to accomodate for iVAE, betaVAE, and sVAE

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_layer_norm
        Whether to use layer norm in layers
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        latent_distribution: str = "normal",
        encode_covariates: bool = False,
        use_chem_prior: bool = True,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_chem_prior = use_chem_prior
        self.beta = 1
        self.sparse_mask_penalty = 1
        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softmax",
        )

        self.action_prior_mean = torch.nn.parameter.Parameter(
            torch.randn((n_labels, n_latent))
        )
        self.gumbel_action = GumbelSigmoid(num_action=n_labels, num_latent=n_latent)

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            z=z,
            library=library,
            batch_index=batch_index,
            y=y,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        return input_dict

    @auto_move_data
    def inference(self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        library = torch.log(x.sum(1)).unsqueeze(1)
        x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        ql = None

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )
        outputs = dict(z=z, qz=qz, ql=ql, library=library)
        return outputs

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        px_scale, _, px_rate, _ = self.decoder(
            "gene",
            decoder_input,
            library,
            batch_index,
            *categorical_input,
            #            y, IMPORTANT TO BE TAKEN AWAY, otherwise we have leakage of y into the decoder by other means than the shift
        )
        px_r = torch.exp(self.px_r)
        px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)

        # Priors
        pl = None
        # sample mask (size chemical times latent) for each datapoint
        mask = self.gumbel_action(y[:, 0].long())  # batch x latentdim
        # subsample actions we care about
        # extract chemical specific means
        mean_z = torch.index_select(
            self.action_prior_mean, 0, y[:, 0].long()
        )  # batch x latent dim

        # prune out entries according to mask
        if self.use_chem_prior:
            pz = Normal(mean_z * mask, torch.ones_like(z))
        else:
            pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return dict(
            px=px,
            pl=pl,
            pz=pz,
        )

    def freeze_params(self):
        # freeze
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.z_encoder.parameters():
            param.requires_grad = False
        self.px_r.requires_grad = False

        for _, mod in self.decoder.named_modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.momentum = 0

    def reinit_actsparse_and_freeze(self, loc):
        # here we must reinit for the embeddings that may have been shrank by sparsity
        with torch.no_grad():
            self.action_prior_mean[loc] = 0
            self.gumbel_action.log_alpha[loc] = 5
        self.gumbel_action.threshold()

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]

        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(
            dim=1
        )
        kl_divergence_l = 0.0

        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = (
            self.beta * kl_weight * kl_local_for_warmup + kl_local_no_warmup
        )

        loss = torch.mean(reconst_loss + weighted_kl_local)

        # add sparsity
        loss += self.sparse_mask_penalty * torch.mean(self.gumbel_action.get_proba())

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, reconst_loss, kl_local, kl_global)

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        sample_batch = tensors[REGISTRY_KEYS.X_KEY]
        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, generative_outputs, losses = self.forward(tensors)
            z = inference_outputs["z"]

            p_za = generative_outputs["pz"].log_prob(z).sum(dim=1)
            q_z_x = inference_outputs["qz"].log_prob(z).sum(dim=1)
            p_x_za = -losses.reconstruction_loss
            log_prob_sum = p_x_za + p_za - q_z_x
            to_sum[:, i] = log_prob_sum

        batch_log_lkl = logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        return batch_log_lkl
