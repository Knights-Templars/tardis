import logging
from pathlib import Path

from tardis.io.atom_data.atom_web_download import (
    get_atomic_repo_config,
)
from tardis.io.configuration.config_internal import get_data_dir

logger = logging.getLogger(__name__)


def resolve_atom_data_fname(fname):
    """
    Check where if atom data HDF file is available on disk, can be downloaded or does not exist

    Parameters
    ----------
    fname : Path
        name or path of atom data HDF file

    Returns
    -------
        : Path
        resolved fpath
    """
    fname = Path(fname)
    if fname.exists():
        return fname

    fname = Path(fname.stem).with_suffix(".h5")
    fpath = Path(get_data_dir()) / fname
    if fpath.exists():
        logger.info(
            f"\n\tAtom Data {fname} not found in local path.\n\tExists in TARDIS Data repo {fpath}"
        )
        return fpath

    atom_data_name = fname.stem
    atom_repo_config = get_atomic_repo_config()
    if atom_data_name in atom_repo_config:
        raise OSError(
            f"Atom Data {fname} not found in path or in TARDIS data repo - it is available as download:\n"
            f"from tardis.io.atom_data import download_atom_data\n"
            f"download_atom_data('{atom_data_name}')"
        )

    raise OSError(
        f"Atom Data {fname} is not found in current path or in TARDIS data repo. {atom_data_name} "
        "is also not a standard known TARDIS atom dataset."
    )
