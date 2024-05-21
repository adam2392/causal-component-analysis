from pathlib import Path
import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
import torch
from sklearn.metrics import r2_score
import joblib

from config import DGP
from data_generator import MultiEnvDataModule, make_multi_env_dgp
from model.cauca_model import LinearCauCAModel, NaiveNonlinearModel, NonlinearCauCAModel
from model.utils import mean_correlation_coefficient

PYTORCH_MPS_HIGH_WATERMARK_RATIO = 0.0


import ast

def parse_nested_list(string):
    try:
        # Use ast.literal_eval to safely evaluate the string as a Python expression
        parsed_list = ast.literal_eval(string)
        # Check if the parsed expression is indeed a list
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list format: '{string}'")


def run_exp(
    sim_name,
    training_seed,
    adjacency_matrix,
    intervention_targets,
    noise_shift_type="mean-std",
    num_samples=200_000,
    batch_size=4096,
    max_epochs=200,
    overwrite=False,
):
    results_dir = Path("./results/")
    results_dir.mkdir(exist_ok=True, parents=True)

    fname = (
        results_dir
        / f"{sim_name}-seed={training_seed}-nsamples={num_samples}-results.npz"
    )
    nonparametric_base_distr = True  # if True, we use soft, if false we use hard

    if not overwrite and fname.exists():
        return

    pl.seed_everything(training_seed, workers=True)

    latent_dim = len(adjacency_matrix)
    # accelerator = "mps"
    # devices = 1
    accelerator = "cuda"
    devices = 1
    # accelerator = "cpu"
    n_jobs = 1
    print("Running with n_jobs:", n_jobs)

    # Define the data generating model
    multi_env_dgp = make_multi_env_dgp(
        latent_dim=latent_dim,
        observation_dim=latent_dim,
        adjacency_matrix=adjacency_matrix,
        intervention_targets_per_env=intervention_targets,
        noise_shift_type=noise_shift_type,
        mixing="nonlinear",
        scm="linear",
        n_nonlinearities=1,
        scm_coeffs_low=-3,
        scm_coeffs_high=3,
        coeffs_min_abs_value=0.5,
        edge_prob=None,
        snr=1.0,
    )

    # now we can wrap this in a pytorch lightning datamodule
    data_module = MultiEnvDataModule(
        multi_env_dgp=multi_env_dgp,
        num_samples_per_env=num_samples,
        batch_size=batch_size,
        num_workers=n_jobs,
        intervention_targets_per_env=intervention_targets,
    )
    data_module.setup()

    k_flows = 1  # number of flows to use in nonlinear ICA model
    k_flows_cbn = 3  # number of flows in nonlinear latent CBN model
    lr_scheduler = None
    lr_min = 0.0
    lr = 1e-4

    # Define the model
    net_hidden_dim = 128
    net_hidden_dim_cbn = 128
    net_hidden_layers = 3
    net_hidden_layers_cbn = 3
    fix_mechanisms = False
    fix_all_intervention_targets = True

    model = NonlinearCauCAModel(
        latent_dim=latent_dim,
        adjacency_matrix=data_module.medgp.adjacency_matrix,
        k_flows=k_flows,
        lr=lr,
        intervention_targets_per_env=intervention_targets,
        lr_scheduler=lr_scheduler,
        lr_min=lr_min,
        adjacency_misspecified=False,
        net_hidden_dim=net_hidden_dim,
        net_hidden_layers=net_hidden_layers,
        fix_mechanisms=fix_mechanisms,
        fix_all_intervention_targets=fix_all_intervention_targets,
        nonparametric_base_distr=nonparametric_base_distr,
        K_cbn=k_flows_cbn,
        net_hidden_dim_cbn=net_hidden_dim_cbn,
        net_hidden_layers_cbn=net_hidden_layers_cbn,
    )
    checkpoint_root_dir = f"{sim_name}-samples={num_samples}-seed={training_seed}"
    checkpoint_dir = Path(checkpoint_root_dir) / "default"
    logger = None
    wandb = False
    check_val_every_n_epoch = 1
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=checkpoint_dir,
    #     save_last=True,
    #     every_n_epochs=check_val_every_n_epoch,
    # )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=3,
        monitor="train_loss",
        every_n_epochs=check_val_every_n_epoch,
    )

    # Train the model
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        devices=devices,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
    )
    trainer.fit(
        model,
        datamodule=data_module,
    )

    print(f"Checkpoint dir: {checkpoint_dir}")
    trainer.test(datamodule=data_module)

    # save the output
    x, v, u, e, int_target, log_prob_gt = data_module.test_dataset[:]
    print(x.shape)
    print(e.shape)

    # Step 1: Obtain learned representations, which are "predictions
    vhat = model.forward(x)

    corr_arr_v_vhat = np.zeros((latent_dim, latent_dim))
    for idx in range(latent_dim):
        for jdx in range(latent_dim):
            corr_arr_v_vhat[jdx, idx] = mean_correlation_coefficient(
                vhat[:, (idx,)], v[:, (jdx,)]
            )

    print("Saving file to: ", fname)

    np.savez_compressed(
        fname,
        x=x.detach().numpy(),
        v=v.detach().numpy(),
        u=u.detach().numpy(),
        e=e.detach().numpy(),
        int_target=int_target.detach().numpy(),
        log_prob_gt=log_prob_gt.detach().numpy(),
        vhat=vhat.detach().numpy(),
        corr_arr_v_vhat=corr_arr_v_vhat,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment for Causal Component Analysis (CauCA)."
    )
    parser.add_argument("--sim-name", type=str, help="Simulation name")
    parser.add_argument("--training-seed", type=int, help="training seed")
    parser.add_argument("--sim_graphs", type=parse_nested_list, help="simulation graphs")
    parser.add_argument("--intervention_targets", type=parse_nested_list, help="intervention targets")
    parser.add_argument("--noise_shift_type", type=str, help="noise shift type")
    parser.add_argument("--num_samples", type=int, help="number of samples")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--max_epochs", type=int, help="maximum epochs")
    args = parser.parse_args()

    run_exp(
        args.sim_name,
        args.training_seed,
        adjacency_matrix=args.sim_graphs,
        intervention_targets=args.intervention_targets,
        noise_shift_type=args.noise_shift_type,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )
