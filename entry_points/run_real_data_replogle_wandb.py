import argparse
import logging

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

logger = logging.getLogger("scvi")
settings = wandb.Settings(start_method="fork")

from svae import SpikeSlabVAE, sVAE

EXOSOME = [
    "ZC3H3",
    "ZFC3H1",
    "CAMTA2",
    "DHX29",
    "DIS3",
    "EXOSC1",
    "EXOSC2",
    "EXOSC3",
    "EXOSC4",
    "EXOSC5",
    "EXOSC6",
    "EXOSC7",
    "EXOSC8",
    "EXOSC9",
    "MBNL1",
    "PABPN1",
    "PIBF1",
    "MTREX",
    "ST20-MTHFS",
    "THAP2",
]

SPLICEOSOME = [
    "ZMAT2",
    "CLNS1A",
    "DDX20",
    "DDX41",
    "DDX46",
    "ECD",
    "GEMIN4",
    "GEMIN5",
    "GEMIN6",
    "GEMIN8",
    "INTS3",
    "INTS4",
    "INTS9",
    "ICE1",
    "LSM2",
    "LSM3",
    "LSM5",
    "LSM5",
    "LSM6",
    "LSM7",
    "MMP17",
    "PHAX",
    "PRPF4",
    "PRPF6",
    "SART3",
    "SF3A2",
    "SMN2",
    "SNAPC1",
    "SNAPC3",
    "SNRPD3",
    "SNRPG",
    "TIPARP",
    "TTC27",
    "TXNL4A",
    "USPL1",
]

MEDIATOR_COMPLEX = [
    "ZDHHC7",
    "ADAM10",
    "EPS8L1",
    "FAM136A",
    "POGLUT3",
    "MED10",
    "MED11",
    "MED12",
    "MED14",
    "MED17",
    "MED18",
    "MED19",
    "MED1",
    "MED20",
    "MED21",
    "MED22",
    "MED28",
    "MED29",
    "MED30",
    "MED6",
    "MED7",
    "MED8",
    "MED9",
    "SUPT6H",
    "BRIX1",
    "TMX2",
]

NUCLEOTIDE_EXCISION_REPAIR = [
    "C1QBP",
    "CCNH",
    "ERCC2",
    "ERCC3",
    "GPN1",
    "GPN3",
    "GTF2E1",
    "GTF2E2",
    "GTF2H1",
    "GTF2H4",
    "MNAT1",
    "NUMA1",
    "PDRG1",
    "PFDN2",
    "POLR2B",
    "POLR2F",
    "POLR2G",
    "RPAP1",
    "RPAP2",
    "RPAP3",
    "TANGO6",
    "TMEM161B",
    "UXT",
]

S40_RIBOSOMAL_UNIT = [
    "ZCCHC9",
    "ZNF236",
    "C1orf131",
    "ZNF84",
    "ZNHIT6",
    "CCDC59",
    "AATF",
    "CPEB1",
    "DDX10",
    "DDX18",
    "DDX21",
    "DDX47",
    "DDX52",
    "DHX33",
    "DHX37",
    "DIMT1",
    "DKC1",
    "DNTTIP2",
    "ESF1",
    "FBL",
    "FBXL14",
    "FCF1",
    "GLB1",
    "HOXA3",
    "IMP4",
    "IMPA2",
    "KRI1",
    "KRR1",
    "LTV1",
    "MPHOSPH10",
    "MRM1",
    "NAF1",
    "NOB1",
    "NOC4L",
    "NOL6",
    "NOP10",
    "PDCD11",
    "ABT1",
    "PNO1",
    "POP1",
    "POP4",
    "POP5",
    "PSMG4",
    "PWP2",
    "RCL1",
    "RIOK1",
    "RIOK2",
    "RNF31",
    "RPP14",
    "RPP30",
    "RPP40",
    "RPS10-NUDT3",
    "RPS10",
    "RPS11",
    "RPS12",
    "RPS13",
    "RPS15A",
    "RPS18",
    "RPS19BP1",
    "RPS19",
    "RPS21",
    "RPS23",
    "RPS24",
    "RPS27A",
    "RPS27",
    "RPS28",
    "RPS29",
    "RPS2",
    "RPS3A",
    "RPS3",
    "RPS4X",
    "RPS5",
    "RPS6",
    "RPS7",
    "RPS9",
    "RPSA",
    "RRP12",
    "RRP7A",
    "RRP9",
    "SDR39U1",
    "SRFBP1",
    "TBL3",
    "TRMT112",
    "TSR1",
    "TSR2",
    "BYSL",
    "C12orf45",
    "USP36",
    "UTP11",
    "UTP20",
    "UTP23",
    "UTP6",
    "BUD23",
    "WDR36",
    "WDR3",
    "WDR46",
    "AAR2",
]

S39_RIBOSOMAL_UNIT = [
    "AARS2",
    "DHX30",
    "GFM1",
    "HMGB3",
    "MALSU1",
    "MRPL10",
    "MRPL11",
    "MRPL13",
    "MRPL14",
    "MRPL16",
    "MRPL17",
    "MRPL18",
    "MRPL19",
    "MRPL22",
    "MRPL23",
    "MRPL24",
    "MRPL27",
    "MRPL2",
    "MRPL33",
    "MRPL35",
    "MRPL36",
    "MRPL37",
    "MRPL38",
    "MRPL39",
    "MRPL3",
    "MRPL41",
    "MRPL42",
    "MRPL43",
    "MRPL44",
    "MRPL4",
    "MRPL50",
    "MRPL51",
    "MRPL53",
    "MRPL55",
    "MRPL9",
    "MRPS18A",
    "MRPS30",
    "NARS2",
    "PTCD1",
    "RPUSD4",
    "TARS2",
    "VARS2",
    "YARS2",
]

S60_RIBOSOMAL_UNIT = [
    "CARF",
    "CCDC86",
    "DDX24",
    "DDX51",
    "DDX56",
    "EIF6",
    "ABCF1",
    "GNL2",
    "LSG1",
    "MAK16",
    "MDN1",
    "MYBBP1A",
    "NIP7",
    "NLE1",
    "NOL8",
    "NOP16",
    "NVL",
    "PES1",
    "PPAN",
    "RBM28",
    "RPL10A",
    "RPL10",
    "RPL11",
    "RPL13",
    "RPL14",
    "RPL17",
    "RPL19",
    "RPL21",
    "RPL23A",
    "RPL23",
    "RPL24",
    "RPL26",
    "RPL27A",
    "RPL30",
    "RPL31",
    "RPL32",
    "RPL34",
    "RPL36",
    "RPL37A",
    "RPL37",
    "RPL38",
    "RPL4",
    "RPL5",
    "RPL6",
    "RPL7",
    "RPL8",
    "RPL9",
    "RRS1",
    "RSL1D1",
    "SDAD1",
    "BOP1",
    "TEX10",
    "WDR12",
]


MT_PROTEIN_TRANSLOCATION = [
    "AARS",
    "CHCHD4",
    "DNAJA3",
    "DNAJC19",
    "EIF2B1",
    "EIF2B2",
    "EIF2B3",
    "EIF2B4",
    "EIF2B5",
    "FARSA",
    "FARSB",
    "GFER",
    "GRPEL1",
    "HARS",
    "HSPA9",
    "HSPD1",
    "HSPE1",
    "IARS2",
    "LARS",
    "LETM1",
    "NARS",
    "OXA1L",
    "PGS1",
    "PHB2",
    "PHB",
    "PMPCA",
    "PMPCB",
    "ATP5F1A",
    "ATP5F1B",
    "ATP5PD",
    "QARS",
    "RARS",
    "SAMM50",
    "PRELID3B",
    "TARS",
    "TIMM23B",
    "TIMM44",
    "TOMM22",
    "TTC1",
    "VARS",
]


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
    parser.add_argument("--split", type=str, default="SPLICEOSOME")
    parser.add_argument("--n_latent", type=int, default=15)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--sparse_penalty", type=float, default=0)
    parser.add_argument("--method", type=str, default="SpikeSlabVAE")
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()

    wandb_logger = WandbLogger(project="wandb-replogle-svae", log_model=True)

    # set up seeds ############################################################

    torch.backends.cudnn.benchmark = True

    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = f"simulations/scvi_replogle_sm_beta{args.beta}_latent{args.n_latent}_meth_{args.method}_sparse{args.sparse_penalty}_split{args.split}"

    # load data ###############################################################
    adata = sc.read_h5ad("replogle.h5ad")
    adata.obs["chem"] = adata.obs["gene"]

    pathway_samples = []
    pathway_list = [
        EXOSOME,
        SPLICEOSOME,
        MEDIATOR_COMPLEX,
        NUCLEOTIDE_EXCISION_REPAIR,
        S40_RIBOSOMAL_UNIT,
        S39_RIBOSOMAL_UNIT,
        S60_RIBOSOMAL_UNIT,
        MT_PROTEIN_TRANSLOCATION,
    ]
    pathway_names = [
        "EXOSOME",
        "SPLICEOSOME",
        "MEDIATOR_COMPLEX",
        "NUCLEOTIDE_EXCISION_REPAIR",
        "S40_RIBOSOMAL_UNIT",
        "S39_RIBOSOMAL_UNIT",
        "S60_RIBOSOMAL_UNIT",
        "MT_PROTEIN_TRANSLOCATION",
    ]

    for perturb in adata.obs["gene"]:
        matched = False
        for pathway_id, pathway_ in enumerate(pathway_list):
            if perturb in pathway_:
                pathway_samples += [pathway_names[pathway_id]]
                matched = True
        if not matched:
            pathway_samples += ["OTHER"]

    if args.split == "effect":
        # get some form of energy distance using MMD
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

        res_energy = {}
        for guide in adata.obs["guide_ids"].unique():
            index_control = np.where(adata.obs["guide_ids"] == "")[0][:1000]
            index_condition = np.where(adata.obs["guide_ids"] == guide)[0][:1000]
            res_energy[guide] = mmd_linear(
                adata.obsm["X_pca"][index_control], adata.obsm["X_pca"][index_condition]
            )

        series = pd.Series(res_energy)
        hold_out_guides = series.sort_values(ascending=False)[:30].index

    elif args.split in pathway_names:
        hold_out_guides = pathway_list[
            np.where(np.array(pathway_names) == args.split)[0][0]
        ]
        # convert format

    adata_train = adata[[x not in hold_out_guides for x in adata.obs["chem"]]].copy()
    adata_test = adata[[x in hold_out_guides for x in adata.obs["chem"]]].copy()

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
    # train the VAE

    guide_list = (
        adata.obs[["_scvi_labels", "chem"]].groupby("_scvi_labels").first().values[:, 0]
    )
    test_range = [
        np.where(x == guide_list)[0][0] for x in adata_test.obs["chem"].unique()
    ]

    chem_prior = (
        args.method == "sVAE" or args.method == "iVAE" or args.method == "SpikeSlabVAE"
    )
    # hack: focus on train only, but leave params for all chemicals
    model.adata = adata_train
    model.module.use_chem_prior = chem_prior
    model.module.beta = args.beta
    model.module.sparse_mask_penalty = args.sparse_penalty
    model.train(
        max_epochs=args.n_epoch,
        check_val_every_n_epoch=1,
        early_stopping=True,
        plan_kwargs={
            "n_epochs_kl_warmup": 10,
        },
        logger=wandb_logger,
    )

    elbo_train = model.get_elbo(adata_train, agg=True)
    elbo_val = model.get_elbo(adata_train, indices=model.validation_indices, agg=True)
    wandb_logger.experiment.config.update(
        {
            "seed": args.seed,
            "method": args.method,
            "split": args.split,
            "beta": args.beta,
            "sparse_penalty": args.sparse_penalty,
            "n_latent": args.n_latent,
            "n_layers": args.n_layers,
        }
    )
    model.save(save_dir, overwrite=True)

    # obtain latents ##########################################################

    latents = model.get_latent_representation(adata_train)

    # adjust model for sparsity binarization
    elbo_pre_pre_test = model.get_elbo(adata_test, agg=True)
    model = reinit_model(model)
    # 1. binarize the sparse mask -> this will add 1 to the unseen chemicals, and freeze all the mask
    model.module.reinit_actsparse_and_freeze(test_range)
    rate_hit = np.mean(
        model.module.gumbel_action.get_proba().detach().cpu().numpy() > 0.5
    )
    # 2. re-train the model on train data
    model.adata = adata_train
    model.module.use_chem_prior = chem_prior
    model.module.sparse_mask_penalty = args.sparse_penalty
    if args.method == "SpikeSlabVAE":
        # if SpikeSlabVAE, make sure you disable to categorical KL, or you'll get nans
        model.module.use_global_kl = False
        model.module.warmup = False
    model.train(
        max_epochs=args.n_epoch,
        check_val_every_n_epoch=1,
        early_stopping=True,
        logger=False,
        plan_kwargs={
            "n_epochs_kl_warmup": 1,
        },
    )
    model.save(save_dir + "_binarized_train", overwrite=True)
    elbo_pre_test = model.get_elbo(adata_test, agg=True)
    iwelbo_pre_test = model.get_marginal_ll(adata_test, n_mc_samples=1000, agg=True)

    # 3. fine-tune model on test data (freeze generative model and fit)
    model = reinit_model(model)
    model.module.freeze_params()
    model.adata = adata_test
    model.module.use_chem_prior = chem_prior
    model.module.sparse_mask_penalty = args.sparse_penalty
    if args.method == "SpikeSlabVAE":
        # if SpikeSlabVAE, make sure you disable to categorical KL, or you'll get nans
        model.module.use_global_kl = False
        model.module.warmup = False
    if args.method == "SpikeSlabVAE" or args.method == "sVAE" or args.method == "iVAE":
        model.train(
            max_epochs=args.n_epoch,
            train_size=1,
            early_stopping=False,
            logger=False,
            plan_kwargs={"n_epochs_kl_warmup": 1, "lr": 0.005},
        )
        model.save(save_dir + "_test", overwrite=True)

    elbo_test = model.get_elbo(adata_test, agg=True)
    iwelbo_test = model.get_marginal_ll(adata_test, n_mc_samples=1000, agg=True)

    wandb.log(
        {
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
