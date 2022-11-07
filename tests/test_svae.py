from svae import SpikeSlabVAE, sparse_shift, sVAE


def test_simulation():
    adata = sparse_shift()


def test_spikeslab_scvi():
    adata = sparse_shift(n_cells_per_chem=10, n_chem=10)
    SpikeSlabVAE.setup_anndata(adata, labels_key="chem")
    model = SpikeSlabVAE(adata, n_latent=10)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    assert len(model.history["elbo_train"]) == 1
    assert len(model.history["elbo_validation"]) == 1

    # test likelihood per chemical
    model.get_elbo()
    model.get_marginal_ll()


def test_sparse_scvi():
    adata = sparse_shift(n_cells_per_chem=10, n_chem=10)
    sVAE.setup_anndata(adata, labels_key="chem")
    model = sVAE(adata, n_latent=10)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    assert len(model.history["elbo_train"]) == 1
    assert len(model.history["elbo_validation"]) == 1

    # test likelihood per chemical
    model.get_elbo()
    model.get_marginal_ll()
