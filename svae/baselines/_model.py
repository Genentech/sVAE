import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model.base import (
    ArchesMixin,
    BaseModelClass,
    RNASeqMixin,
    UnsupervisedTrainingMixin,
    VAEMixin,
)
from scvi.utils import setup_anndata_dsp

from svae.baselines._module import sVAEModule

logger = logging.getLogger(__name__)


class sVAE(
    RNASeqMixin, VAEMixin, ArchesMixin, UnsupervisedTrainingMixin, BaseModelClass
):
    """
    single-cell Variational Inference [Lopez18]_.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scvi.module.VAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.SCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()

    Notes
    -----
    See further usage examples in the following tutorials:

    1. :doc:`/tutorials/notebooks/api_overview`
    2. :doc:`/tutorials/notebooks/harmonization`
    3. :doc:`/tutorials/notebooks/scarches_scvi_tools`
    4. :doc:`/tutorials/notebooks/scvi_in_R`
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        latent_distribution: Literal["normal", "ln"] = "normal",
        **module_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch

        self.module = sVAEModule(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            latent_distribution=latent_distribution,
            **module_kwargs,
        )
        self._model_summary_string = (
            "sVAE Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_elbo(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        agg: bool = False,
    ) -> float:
        """
        Return the ELBO for the data.

        The ELBO is a lower bound on the log likelihood of the data used for optimization
        of VAEs. Note, this is not the negative ELBO, higher is better.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        # Iterate once over the data and compute the elbo
        elbo = []
        for tensors in scdl:
            _, _, scvi_loss = self.module(tensors)

            recon_loss = scvi_loss.reconstruction_loss
            kl_local = scvi_loss.kl_local
            elbo += [(recon_loss + kl_local).cpu().numpy()]

        # now aggregate by chemical
        elbo = np.concatenate(elbo)
        elbo_res = {}
        label_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)[
            "original_key"
        ]
        cat = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)[
            "categorical_mapping"
        ]
        for c in cat:
            if indices is not None:
                ind = np.where(adata.obs[label_key][indices].values == c)[0]
            else:
                ind = np.where(adata.obs[label_key].values == c)[0]
            if len(ind) > 10:
                elbo_res[c] = np.mean(elbo[ind])

        if agg:
            return pd.Series(elbo_res).values.mean()
        else:
            return elbo_res

    @torch.no_grad()
    def get_marginal_ll(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_mc_samples: int = 1000,
        batch_size: Optional[int] = None,
        agg: bool = False,
    ) -> float:
        """
        Return the marginal LL for the data, calculated by label

        The computation here is a biased estimator of the marginal log likelihood of the data.
        Note, this is not the negative log likelihood, higher is better.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_mc_samples
            Number of Monte Carlo samples to use for marginal LL estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        if hasattr(self.module, "marginal_ll"):
            log_lkl = []
            for tensors in scdl:
                log_lkl += [self.module.marginal_ll(tensors, n_mc_samples=n_mc_samples)]
        else:
            raise NotImplementedError(
                "marginal_ll is not implemented for current model. "
                "Please raise an issue on github if you need it."
            )

        # now aggregate by chemical
        log_lkl = np.concatenate(log_lkl)
        log_lkl_res = {}
        label_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)[
            "original_key"
        ]
        cat = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)[
            "categorical_mapping"
        ]
        for c in cat:
            if indices is not None:
                ind = np.where(adata.obs[label_key][indices].values == c)[0]
            else:
                ind = np.where(adata.obs[label_key].values == c)[0]
            if len(ind) > 10:
                log_lkl_res[c] = np.mean(log_lkl[ind])

        if agg:
            return pd.Series(log_lkl_res).values.mean()
        else:
            return log_lkl_res
