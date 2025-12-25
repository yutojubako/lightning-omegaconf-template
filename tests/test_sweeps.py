from pathlib import Path

import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"


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
