from pathlib import Path

import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"

# NOTE: This file previously contained tests for:
# - test_experiments: Running all experiment configs
# - test_hydra_sweep: Parameter sweeps with multiple values
# - test_optuna_sweep: Optuna hyperparameter optimization
# - test_optuna_sweep_ddp_sim_wandb: Optuna with DDP simulation
#
# These tests were removed because the PR replaces Hydra's @hydra.main decorator
# with a lightweight OmegaConf-based config loader. The new loader supports:
# - Loading and merging config files from the configs/ directory
# - CLI overrides via OmegaConf.from_cli()
#
# The new loader does NOT support:
# - Hydra's multirun/sweep functionality (--multirun flag)
# - Hydra's Optuna plugin for hyperparameter optimization
#
# If these features are needed, they should be implemented separately or
# the project should continue using Hydra's @hydra.main decorator.


@RunIf(sh=True)
@pytest.mark.slow
def test_cli_overrides_fast_dev_run(tmp_path: Path) -> None:
    """Test CLI overrides with fast dev run enabled.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "trainer.fast_dev_run=true",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "data.num_workers=0",
        "data.pin_memory=false",
        "extras.print_config=false",
        "extras.enforce_tags=false",
        f"paths.log_dir={tmp_path}",
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_cli_overrides_max_epochs(tmp_path: Path) -> None:
    """Test CLI overrides for short training runs.

    :param tmp_path: The temporary logging path.
    """
    command = [
        startfile,
        "trainer.max_epochs=1",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "data.num_workers=0",
        "data.pin_memory=false",
        "extras.print_config=false",
        "extras.enforce_tags=false",
        f"paths.log_dir={tmp_path}",
    ]
    run_sh_command(command)
