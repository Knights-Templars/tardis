import os
import pytest
import numpy as np
import numpy.testing as npt
from pathlib import Path
import radioactivedecay as rd

import tardis
from tardis.model import SimulationState
from tardis.io.configuration import config_reader
from tardis.io.util import yaml_load_file, YAMLLoader
from tardis.energy_input.gamma_ray_transport import (
    calculate_shell_masses,
    create_isotope_dicts,
    get_all_isotopes,
    create_inventories_dict,
    calculate_total_decays,
)
import astropy.units as u
import astropy.constants as c

# DATA_PATH = os.path.join(
#    tardis.__path__[0], "io", "configuration", "tests", "data"
# )
NI56_NUCLIDE = rd.Nuclide("Ni56")


@pytest.fixture(scope="module")
def gamma_ray_config(example_configuration_dir: Path):
    """
    Parameters
    ----------
    example_configuration_dir: Path to the configuration directory.

    Returns
    -------
    Tardis configuration
    """
    yml_path = (
        example_configuration_dir
        / "tardis_configv1_density_exponential_nebular.yml"
    )

    return config_reader.Configuration.from_yaml(yml_path)


@pytest.fixture(scope="module")
def simulation_setup(gamma_ray_config):
    """
    Parameters
    ----------
    gamma_ray_config:

    Returns
    -------

    """
    gamma_ray_config.model.structure.velocity.start = 1.0 * u.km / u.s
    gamma_ray_config.model.structure.density.rho_0 = 5.0e2 * u.g / (u.cm**3)
    gamma_ray_config.supernova.time_explosion = 1.0 * (u.d)
    model = SimulationState.from_config(gamma_ray_config)
    return model


def test_calculate_shell_masses(simulation_setup):
    """
    Function to test calculation of shell masses.
    Parameters
    ----------
    simulation_setup: A simulation setup which returns a model.
    """
    model = simulation_setup
    volume = model.volume.to("cm^3")
    density = model.density.to("g/cm^3")
    desired = (volume * density).to(u.g).value

    shell_masses = calculate_shell_masses(model).value
    npt.assert_almost_equal(shell_masses, desired)


@pytest.mark.parametrize("nuclide_name", ["Ni-56"])
def test_activity(simulation_setup, nuclide_name):
    """
    Function to test the decay of 56Ni in radioactivedecay with an analytical solution.
    Parameters
    ----------
    simulation_setup: A simulation setup which returns a model.
    nuclide_name: Name of the nuclide.
    """
    nuclide = rd.Nuclide(nuclide_name)
    model = simulation_setup
    t_half = nuclide.half_life() * u.s
    decay_constant = np.log(2) / t_half
    time_delta = 1.0 * u.s
    shell_masses = calculate_shell_masses(model)
    raw_isotope_abundance = model.raw_isotope_abundance
    raw_isotope_abundance_mass = raw_isotope_abundance.apply(
        lambda x: x * shell_masses, axis=1
    )
    mass = raw_isotope_abundance_mass.loc[nuclide.Z, nuclide.A][0]
    iso_dict = create_isotope_dicts(raw_isotope_abundance, shell_masses)
    inv_dict = create_inventories_dict(iso_dict)

    total_decays = calculate_total_decays(inv_dict, time_delta)
    actual = total_decays[0][nuclide.Z, nuclide.A][nuclide_name]

    isotopic_mass = nuclide.atomic_mass * (u.g)
    number_of_moles = mass * (u.g) / isotopic_mass
    number_of_atoms = (number_of_moles * c.N_A).value
    N1 = number_of_atoms * np.exp(-decay_constant * time_delta)
    expected = number_of_atoms - N1.value

    npt.assert_allclose(actual, expected, rtol=1e-7)


@pytest.mark.xfail(reason="To be implemented")
def test_activity_chain(simulation_setup):
    model = simulation_setup
    t_half_Ni = NI_INV.half_lives("s")["Ni-56"]
    t_half_Co = 77.236 * u.d.to(u.s)
    decay_constant_Ni = np.log(2) / t_half_Ni
    decay_constant_Co = np.log(2) / t_half_Co
    time_delta = 80.0 * u.d.to(u.s)
    shell_masses = calculate_shell_masses(model)
    raw_isotope_abundance = model.raw_isotope_abundance
    raw_isotope_abundance_mass = raw_isotope_abundance.apply(
        lambda x: x * shell_masses, axis=1
    )
    ni_mass = raw_isotope_abundance_mass.loc[28, 56][0]
    iso_dict = create_isotope_dicts(raw_isotope_abundance, shell_masses)
    inv_dict = create_inventories_dict(iso_dict)
    total_decays = calculate_total_decays(inv_dict, time_delta)
    actual_Ni = total_decays[0][28, 56]["Ni-56"]
    actual_Co = total_decays[0][28, 56]["Co-56"]

    isotopic_mass_Ni = NI_INV._get_atomic_mass("Ni-56") * (u.g)
    number_of_moles = ni_mass * (u.g) / isotopic_mass_Ni
    number_of_atoms = number_of_moles * c.N_A
    N1 = number_of_atoms.value * np.exp(-decay_constant_Ni * time_delta)
    N2 = (
        number_of_atoms.value
        * decay_constant_Ni
        / (decay_constant_Co - decay_constant_Ni)
        * (
            np.exp(-decay_constant_Ni * time_delta)
            - np.exp(-decay_constant_Co * time_delta)
        )
    )
    expected_Ni = number_of_atoms.value * (
        1 - np.exp(-decay_constant_Ni * time_delta)
    )
    expected_Co = number_of_atoms.value - N1 - N2

    npt.assert_almost_equal(actual_Ni, expected_Ni)
    npt.assert_almost_equal(actual_Co, expected_Co)
