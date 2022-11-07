import argparse
import logging
import os

import numpy as np
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

logger = logging.getLogger("scvi")
settings = wandb.Settings(start_method="fork")

from svae import SpikeSlabVAE, metrics, sparse_shift, sVAE


def reinit_model(model):
    # create new model and copy module inside of it
    # this will create a new trainer etc..
    temp_module = model.module
    model_ = model.__class__(adata, n_latent=args.n_latent, n_layers=args.n_layers)
    model_.module = temp_module
    return model_


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sVAE benchmark experiment")

    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="simulation")
    parser.add_argument("--n_latent", type=int, default=15)
    parser.add_argument("--n_cells_per_chem", type=int, default=250)
    parser.add_argument("--n_chem", type=int, default=100)
    parser.add_argument("--n_genes", type=int, default=100)
    parser.add_argument("--n_epoch", type=int, default=300)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--sparse_penalty", type=float, default=0)
    parser.add_argument("--method", type=str, default="SpikeSlabVAE")
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()

    wandb_logger = WandbLogger(project="wandb-svae", log_model=True)

    # set up seeds ############################################################

    torch.backends.cudnn.benchmark = True

    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = f"svae_lat{args.n_latent}mth_{args.method}_sp{args.sparse_penalty}"

    # logger.info("Generating dataset")

    # load data ###############################################################
    adata = sparse_shift(
        n_latent=args.n_latent,
        n_cells_per_chem=args.n_cells_per_chem,
        n_chem=args.n_chem,
        n_genes=args.n_genes,
    )

    # split anndata in train / test
    if args.method != "SpikeSlabVAE":
        sVAE.setup_anndata(adata, labels_key="chem")
        model = sVAE(adata, n_latent=args.n_latent, n_layers=args.n_layers)

    if args.method == "SpikeSlabVAE":
        if args.sparse_penalty == 0:
            args.sparse_penalty = 1
        SpikeSlabVAE.setup_anndata(adata, labels_key="chem")
        model = SpikeSlabVAE(adata, n_latent=args.n_latent, n_layers=args.n_layers)

    # train or load model #####################################################

    test_range = range(80, 100)
    adata_train = adata[[x not in test_range for x in adata.obs["chem"]]].copy()
    adata_test = adata[[x in test_range for x in adata.obs["chem"]]].copy()

    chem_prior = (
        args.method == "sVAE" or args.method == "iVAE" or args.method == "SpikeSlabVAE"
    )
    # hack: focus on train only, but leave params for all chemicals
    model.adata = adata_train
    model.module.use_chem_prior = chem_prior
    model.module.sparse_mask_penalty = args.sparse_penalty
    model.train(
        max_epochs=args.n_epoch,
        check_val_every_n_epoch=1,
        early_stopping=True,
        plan_kwargs={
            "n_epochs_kl_warmup": 50,
        },
        logger=wandb_logger,
    )

    elbo_train = model.get_elbo(adata_train, agg=True)
    elbo_val = model.get_elbo(adata_train, indices=model.validation_indices, agg=True)
    wandb_logger.experiment.config.update(
        {
            "seed": args.seed,
            "method": args.method,
            "sparse_penalty": args.sparse_penalty,
            "n_latent": args.n_latent,
            "n_layers": args.n_layers,
        }
    )
    model.save(save_dir, overwrite=True)

    # obtain latents ##########################################################

    latents = model.get_latent_representation(adata_train)
    gt_latents = adata_train.obsm["groundtruth_latent"]

    # compute and report metrics ##############################################
    
    score_mat_pearson, score_pearson = metrics.mean_corr_coef_np(
        gt_latents, latents, method="pearson"
    )
    score_mat_spearman, score_spearman = metrics.mean_corr_coef_np(
        gt_latents, latents, method="spearman"
    )
    score_r2, _ = metrics.linear_regression_metric(gt_latents,
                                          latents)

    # evaluate sparsity pattern of graph on training data
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import f1_score, precision_score, recall_score

    if args.n_latent == adata_train.obsm["groundtruth_latent"].shape[1]:
        # match dimensions
        w = -np.corrcoef(latents, gt_latents, rowvar=False)[
            : args.n_latent, args.n_latent :
        ]

        if args.method in ["VAE", "iVAE"]:
            # evaluate with top 2 latent per chemical
            y1, y2 = np.abs(
                model.module.action_prior_mean.detach().cpu().numpy().T
            ).argsort(axis=0)[-2:, :]
            mat_A = np.zeros_like(
                model.module.action_prior_mean.detach().cpu().numpy().T
            )
            mat_A[y1, np.arange(args.n_chem)] = 1
            mat_A[y2, np.arange(args.n_chem)] = 1
        else:
            mat_A = (
                model.module.gumbel_action.get_proba().detach().cpu().numpy().T > 0.5
            ).astype(float)

        rate_hit = np.mean(mat_A)
        mat_B = (np.abs(adata.uns["prior_mean"].T) > 0).astype(float)
        y_true, y_pred = mat_B.flatten(), mat_A[linear_sum_assignment(w.T)[1]].flatten()
        score_p_graph, score_r_graph, score_f1_graph = (
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred),
        )

    # adjust model for sparsity binarization
    elbo_pre_pre_test = model.get_elbo(adata_test, agg=True)
    model = reinit_model(model)
    # 1. binarize the sparse mask -> this adds 1 to the unseen chemicals, and freeze mask
    model.module.reinit_actsparse_and_freeze(np.arange(80, 100))
    # 2. re-train the model on train data
    model.adata = adata_train
    model.module.use_chem_prior = chem_prior
    model.module.sparse_mask_penalty = args.sparse_penalty
    if args.method == "SpikeSlabVAE":
        # if ssMVI, make sure you disable to categorical KL, or you'll get nans
        model.module.use_global_kl = False
        model.module.warmup = False
    
    model.train(
        max_epochs=300,
        check_val_every_n_epoch=1,
        early_stopping=True,
        logger=False,
        plan_kwargs={
            "n_epochs_kl_warmup": 1,
        },
    )
    
    model.save(save_dir + "_binarized_train", overwrite=True)
    elbo_pre_test = model.get_elbo(adata_test, agg=True)
    iwelbo_pre_test = model.get_marginal_ll(adata_test, agg=True)

    # 3. fine-tune model on test data (freeze generative model and fit)
    model = reinit_model(model)
    model.module.freeze_params()
    model.adata = adata_test
    model.module.use_chem_prior = chem_prior
    model.module.sparse_mask_penalty = args.sparse_penalty
    if args.method == "SpikeSlabVAE":
        model.module.use_global_kl = False
        model.module.warmup = False
    model.train(
        max_epochs=300,
        train_size=1,
        early_stopping=False,
        logger=False,
        plan_kwargs={"n_epochs_kl_warmup": 1, "lr": 0.005},
    )
    model.save(save_dir + "_test", overwrite=True)

    elbo_test = model.get_elbo(adata_test, agg=True)
    iwelbo_test = model.get_marginal_ll(adata_test, agg=True)

    wandb.log(
        {
            f"mcc_spearman": score_spearman,
            f"mcc_pearson": score_pearson,
            f"r_2": score_r2,
            f"precision": score_p_graph,
            f"recall": score_r_graph,
            f"f1_graph": score_f1_graph,
            f"elbo_train": elbo_train,
            f"elbo_val": elbo_val,
            f"elbo_test": elbo_test,
            f"iwelbo_test": iwelbo_test,
            f"rate_hit": rate_hit,
            f"elbo_pre_test": elbo_pre_test,
            f"iwelbo_pre_test": iwelbo_pre_test,
            f"elbo_pre_pre_test": elbo_pre_pre_test,
        }
    )
    wandb.finish()
