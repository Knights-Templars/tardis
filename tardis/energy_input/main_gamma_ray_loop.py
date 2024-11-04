import logging

import astropy.units as u
import numpy as np
import pandas as pd

from tardis.energy_input.gamma_packet_loop import gamma_packet_loop, new_gamma_packet_transport, propagate_packet_over_time
from tardis.energy_input.gamma_ray_packet_source import GammaRayPacketSource
from tardis.energy_input.gamma_ray_transport import (
    calculate_ejecta_velocity_volume,
    get_taus,
    iron_group_fraction_per_shell,
)
from tardis.energy_input.GXPacket import GXPacket
from tardis.energy_input.util import get_index
from tardis.energy_input.util import make_isotope_string_tardis_like



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_effective_time_array(time_start, time_end, time_space, time_steps):
    """
    Function to get the effective time array for the gamma-ray loop.

    Parameters
    ----------
    time_start : float
        start time in days.
    time_end : float
        end time in days.
    time_space : str
        linear or log.
    time_steps : int
        number of time steps.

    Returns
    -------
    times : np.ndarray
        array of times in secs.
    effective_time_array : np.ndarray
        effective time array in secs.
    """

    assert time_start < time_end, "time_start must be smaller than time_end!"
    if time_space == "log":
        times = np.geomspace(time_start, time_end, time_steps + 1)
        effective_time_array = np.sqrt(times[:-1] * times[1:])
    else:
        times = np.linspace(time_start, time_end, time_steps + 1)
        effective_time_array = 0.5 * (times[:-1] + times[1:])

    return times, effective_time_array


def run_gamma_ray_loop(
    model,
    isotope_decay_df,
    cumulative_decays_df,
    num_decays,
    times,
    effective_time_array,
    time_steps,
    seed,
    positronium_fraction,
    spectrum_bins,
    grey_opacity,
    photoabsorption_opacity="tardis",
    pair_creation_opacity="tardis",
):
    """
    Main loop to determine the gamma-ray propagation through the ejecta.

    Parameters
    ----------
    model : tardis.model.Radial1DModel
            Tardis model object.
    plasma : tardis.plasma.standard_plasmas.BasePlasma
            Tardis plasma object.
    isotope_decay_df : pd.DataFrame
            DataFrame containing the cumulative decay data.
    cumulative_decays_df : pd.DataFrame
            DataFrame containing the time evolving mass fractions.
    num_decays : int
            Number of packets to decay.
    times : np.ndarray
            Array of times in days.
    effective_time_array : np.ndarray
            Effective time array in days.
    seed : int
            Seed for the random number generator.
    positronium_fraction : float
            Fraction of positronium.
    spectrum_bins : int
            Number of spectrum bins.
    grey_opacity : float
            Grey opacity.
    photoabsorption_opacity : str
            Photoabsorption opacity.
    pair_creation_opacity : str
            Pair creation opacity.


    Returns
    -------

    escape_energy : pd.DataFrame
            DataFrame containing the energy escaping the ejecta.
    packets_df_escaped : pd.DataFrame
            DataFrame containing the packets info that escaped the ejecta.


    """
    np.random.seed(seed)
    times = times * u.d.to(u.s)
    effective_time_array = effective_time_array * u.d.to(u.s)
    inner_velocities = model.v_inner.to("cm/s").value
    outer_velocities = model.v_outer.to("cm/s").value
    ejecta_volume = model.volume.to("cm^3").value
    shell_masses = model.volume * model.density
    number_of_shells = len(shell_masses)
    # TODO: decaying upto times[0]. raw_isotope_abundance is possibly not the best name
    raw_isotope_abundance = model.composition.raw_isotope_abundance.sort_values(
        by=["atomic_number", "mass_number"], ascending=False
    )

    dt_array = np.diff(times)

    ejecta_velocity_volume = calculate_ejecta_velocity_volume(model)

    inv_volume_time = (
        1.0 / ejecta_velocity_volume[:, np.newaxis]
    ) / effective_time_array**3.0

    for (
        atom_number
    ) in model.composition.isotopic_number_density.index.get_level_values(0):
        values = model.composition.isotopic_number_density.loc[
            atom_number
        ].values
        if values.shape[1] > 1:
            model.elemental_number_density.loc[atom_number] = np.sum(
                values, axis=0
            )
        else:
            model.elemental_number_density.loc[atom_number] = values
    

    # Electron number density
    electron_number_density = model.elemental_number_density.mul(
        model.elemental_number_density.index,
        axis=0,
    ).sum()
    print ("The electron number density is ", electron_number_density)
    electron_number = np.array(electron_number_density * ejecta_volume)
    #print ("The electron number is ", electron_number)

    # Evolve electron number and mass density with time
    electron_number_density_time = (
        electron_number[:, np.newaxis] * inv_volume_time
    )
    #print ("The electron number density time is ", electron_number_density_time)    
    mass_density_time = shell_masses[:, np.newaxis] * inv_volume_time
    #print ("The mass density time is ", mass_density_time)

    taus, parents = get_taus(raw_isotope_abundance)
    # Need to get the strings for the isotopes without the dashes
    taus = make_isotope_string_tardis_like(taus)

    gamma_df = cumulative_decays_df[cumulative_decays_df["radiation"] == "g"]
    gamma_df_decay_energy_keV = gamma_df["decay_energy_keV"]
    #print (gamma_df_decay_energy_keV.groupby("shell_number").sum())
    #gamma_df_decay_energy_keV.to_csv("gamma_decay_energy_keV.csv", sep=" ")

    total_energy_gamma = gamma_df["decay_energy_erg"].sum()

    energy_per_packet = total_energy_gamma / num_decays

    logger.info(f"Total energy in gamma-rays is {total_energy_gamma}")
    logger.info(f"Energy per packet is {energy_per_packet}")

    packet_source = GammaRayPacketSource(
        energy_per_packet,
        isotope_decay_df,
        positronium_fraction,
        inner_velocities,
        outer_velocities,
        inv_volume_time,
        times,
        effective_time_array,
        taus,
        parents,
    )

    logger.info("Creating packets")
    packet_collection, isotope_positron_fraction = packet_source.create_packets(
        cumulative_decays_df, num_decays, seed
    )

    total_energy = np.zeros((number_of_shells, len(times) - 1))

   

    #positron_energy_fraction = packet_source.calculate_positron_energy_fraction(cumulative_decays_df)

    #Bin the positron fraction by time and shell
    
    # for i in range(len(shell_masses)):
    #    for j in range(len(times)):
    #        positron_energy_deposition[i, j] += positron_energy_fraction.loc[j, i] * energy_per_packet


    logger.info("Creating packet list")
    packets = []
    # This for loop is expensive. Need to rewrite GX packet to handle arrays
    packets = [
        GXPacket(
            packet_collection.location[:, i],
            packet_collection.direction[:, i],
            packet_collection.energy_rf[i],
            packet_collection.energy_cmf[i],
            packet_collection.nu_rf[i],
            packet_collection.nu_cmf[i],
            packet_collection.status[i],
            packet_collection.shell[i],
            packet_collection.time_start[i],
            packet_collection.time_index[i],
        )
        for i in range(num_decays)
    ]
    time_current = []
    packet_opacity_array = np.zeros((num_decays, 10))
    for i, p in enumerate(packets):
        total_energy[p.shell, p.time_index] \
        += isotope_positron_fraction[i] * energy_per_packet
        time_current.append(p.time_start)

        packet_opacity_array[i] = np.array([
        i, 
        p.energy_rf,
        p.energy_cmf,
        p.get_location_r(),
        p.time_start,
        int(p.status),
        0,
        0,
        0,
        0])
        #print (isotope_positron_fraction[i])
    
    #return packets, time_current
    

    # shell_number = []
    # time_current = []

    # # Bin the frequency of the packets in shell and time

    # packets_nu_cmf_array = np.zeros((number_of_shells, len(times)))
    # packets_nu_rf_array = np.zeros((number_of_shells, len(times)))
    # packets_energy_cmf_array = np.zeros((number_of_shells, len(times)))
    # packets_energy_rf_array = np.zeros((number_of_shells, len(times)))
    # #packets_positron_energy_array = np.zeros((number_of_shells, len(times) -1))

    # positron_energy = pd.DataFrame(
    #      data=positron_energy_deposition, columns=times
    # )


    # for p in packets:
    #     shell_number.append(p.shell)
    #     time_current.append(p.time_start)
    #     packets_nu_cmf_array[p.shell, p.time_index] += p.nu_cmf
    #     packets_nu_rf_array[p.shell, p.time_index] += p.nu_rf
    #     packets_energy_cmf_array[p.shell, p.time_index] += p.energy_cmf
    #     packets_energy_rf_array[p.shell, p.time_index] += p.energy_rf

    # return (
    #     packets,
    #     packets_nu_cmf_array,
    #     packets_nu_rf_array,
    #     packets_energy_cmf_array,
    #     packets_energy_rf_array,
    #     positron_energy,
    # )

    # # return packets

    energy_bins = np.logspace(2, 3.8, spectrum_bins)
    energy_out = np.zeros((len(energy_bins - 1), len(times) - 1))
    energy_deposited = np.zeros((number_of_shells, len(times) - 1))
    packets_info_array = np.zeros((int(num_decays), 8))
    iron_group_fraction = iron_group_fraction_per_shell(model)
    total_gamma_packet_opacity = np.zeros((number_of_shells, len(times) - 1))   
    #print ("The iron group fraction is ", iron_group_fraction)

    logger.info("Entering the main gamma-ray loop")

    total_cmf_energy = 0
    total_rf_energy = 0

    for p in packets:
        total_cmf_energy += p.energy_cmf
        total_rf_energy += p.energy_rf

    logger.info(f"Total CMF energy is {total_cmf_energy}")
    logger.info(f"Total RF energy is {total_rf_energy}")
    

    (
        energy_out,
        packets_array,
        energy_deposited_gamma,
        total_energy,
        total_opacity,
        total_gamma_packet_opacity
    ) = gamma_packet_loop(
        packets,
        grey_opacity,
        photoabsorption_opacity,
        pair_creation_opacity,
        electron_number_density_time,
        mass_density_time,
        iron_group_fraction.to_numpy(),
        inner_velocities,
        outer_velocities,
        dt_array,
        times,
        effective_time_array,
        energy_bins,
        energy_out,
        total_energy,
        energy_deposited,
        total_gamma_packet_opacity,
        packet_opacity_array,
        packets_info_array,
    )

    packets_df_escaped = pd.DataFrame(
        data=packets_array,
        columns=[
            "packet_index",
            "status",
            "nu_cmf",
            "nu_rf",
            "energy_cmf",
            "luminosity",
            "energy_rf",
            "shell_number",
        ],
    )

    opacity_df = pd.DataFrame(
        data=total_gamma_packet_opacity,
        columns=["packet_index",
                 "nu_rf",
                 "energy_cmf",
                 "location_r",
                 "time_start",
                 "status",
                 "compton_opacity",
                 "photoabsorption_opacity",
                 "pair_creation_opacity",
                 "total_opacity"],

    )

    escape_energy = pd.DataFrame(
        data=energy_out, columns=times[:-1], index=energy_bins
    )

    # deposited energy by gamma-rays in ergs
    deposited_energy = pd.DataFrame(
        data=energy_deposited_gamma, columns=times[:-1]
    )
    # total_energy in ergs
    total_energy = pd.DataFrame(
        data=total_energy, columns=times[:-1]
    )
    total_opacity = pd.DataFrame(data=total_opacity, columns=times[:-1])

    print ("Total positron energy is ", total_energy.sum().sum())    

    #total_deposited_energy = (positron_energy + deposited_energy) / dt_array
    total_deposited_energy = total_energy / dt_array

    return (
        escape_energy,
        packets_df_escaped,
        deposited_energy,
        total_deposited_energy,
        total_opacity,
        opacity_df,
    )


def get_packet_properties(number_of_shells, times, time_steps, packets):

    """
    Function to get the properties of the packets.

    Parameters
    ----------
    packets : list
        List of packets.

    Returns
    -------
    packets_nu_cmf_array : np.ndarray
        Array of packets in cmf.
    packets_nu_rf_array : np.ndarray
        Array of packets in rf.
    packets_energy_cmf_array : np.ndarray
        Array of packets energy in cmf.
    packets_energy_rf_array : np.ndarray
        Array of packets energy in rf.
    packets_positron_energy_array : np.ndarray
        Array of packets positron energy.
    """

    # collect the properties of the packets
    shell_number = []
    time_current = []

    # Bin the frequency of the packets in shell and time

    packets_nu_cmf_array = np.zeros((number_of_shells, len(times)))
    packets_nu_rf_array = np.zeros((number_of_shells, len(times)))
    packets_energy_cmf_array = np.zeros((number_of_shells, len(times)))
    packets_energy_rf_array = np.zeros((number_of_shells, len(times)))
    packets_positron_energy_array = np.zeros((number_of_shells, len(times)))

    for p in packets:
        time_index = get_index(p.time_current, times)
        shell_number.append(p.shell)
        time_current.append(p.time_current)
        packets_nu_cmf_array[p.shell, time_index] += p.nu_cmf
        packets_nu_rf_array[p.shell, time_index] += p.nu_rf
        packets_energy_cmf_array[p.shell, time_index] += p.energy_cmf
        packets_energy_rf_array[p.shell, time_index] += p.energy_rf
        packets_positron_energy_array[p.shell, time_index] += p.positron_energy

    return (
        packets_nu_cmf_array,
        packets_nu_rf_array,
        packets_energy_cmf_array,
        packets_energy_rf_array,
        packets_positron_energy_array,
    )


def run_gamma_ray_loop_new_transport(
    model,
    isotope_decay_df,
    cumulative_decays_df,
    num_decays,
    times,
    effective_time_array,
    seed,
    positronium_fraction,
    spectrum_bins,
    grey_opacity,
):
    """
    Main loop to determine the gamma-ray propagation through the ejecta.

    Parameters
    ----------
    model : tardis.model.Radial1DModel
            Tardis model object.
    plasma : tardis.plasma.standard_plasmas.BasePlasma
            Tardis plasma object.
    isotope_decay_df : pd.DataFrame
            DataFrame containing the cumulative decay data.
    cumulative_decays_df : pd.DataFrame
            DataFrame containing the time evolving mass fractions.
    num_decays : int
            Number of packets to decay.
    times : np.ndarray
            Array of times in days.
    effective_time_array : np.ndarray
            Effective time array in days.
    seed : int
            Seed for the random number generator.
    positronium_fraction : float
            Fraction of positronium.
    spectrum_bins : int
            Number of spectrum bins.
    grey_opacity : float
            Grey opacity.
    photoabsorption_opacity : str
            Photoabsorption opacity.
    pair_creation_opacity : str
            Pair creation opacity.


    Returns
    -------

    escape_energy : pd.DataFrame
            DataFrame containing the energy escaping the ejecta.
    packets_df_escaped : pd.DataFrame
            DataFrame containing the packets info that escaped the ejecta.


    """
    np.random.seed(seed)
    times = times * u.d.to(u.s)
    effective_time_array = effective_time_array * u.d.to(u.s)
    inner_velocities = model.v_inner.to("cm/s").value
    outer_velocities = model.v_outer.to("cm/s").value
    ejecta_volume = model.volume.to("cm^3").value
    shell_masses = model.volume * model.density
    number_of_shells = len(shell_masses)
    # TODO: decaying upto times[0]. raw_isotope_abundance is possibly not the best name
    raw_isotope_abundance = model.composition.raw_isotope_abundance.sort_values(
        by=["atomic_number", "mass_number"], ascending=False
    )

    dt_array = np.diff(times)

    ejecta_velocity_volume = calculate_ejecta_velocity_volume(model)

    inv_volume_time = (
        1.0 / ejecta_velocity_volume[:, np.newaxis]
    ) / effective_time_array**3.0

    for (
        atom_number
    ) in model.composition.isotopic_number_density.index.get_level_values(0):
        values = model.composition.isotopic_number_density.loc[
            atom_number
        ].values
        if values.shape[1] > 1:
            model.elemental_number_density.loc[atom_number] = np.sum(
                values, axis=0
            )
        else:
            model.composition.elemental_number_density.loc[atom_number] = values
    

    # Electron number density
    electron_number_density = model.composition.elemental_number_density.mul(
        model.composition.elemental_number_density.index,
        axis=0,
    ).sum()
    print ("The electron number density is ", electron_number_density)
    electron_number = np.array(electron_number_density * ejecta_volume)
    #print ("The electron number is ", electron_number)

    # Evolve electron number and mass density with time
    electron_number_density_time = (
        electron_number[:, np.newaxis] * inv_volume_time
    )
    #print ("The electron number density time is ", electron_number_density_time)    
    mass_density_time = shell_masses[:, np.newaxis] * inv_volume_time
    #print ("The mass density time is ", mass_density_time)

    taus, parents = get_taus(raw_isotope_abundance)
    # Need to get the strings for the isotopes without the dashes
    taus = make_isotope_string_tardis_like(taus)

    gamma_df = cumulative_decays_df[cumulative_decays_df["radiation"] == "g"]
    gamma_df_decay_energy_keV = gamma_df["decay_energy_keV"]
    #print (gamma_df_decay_energy_keV.groupby("shell_number").sum())
    #gamma_df_decay_energy_keV.to_csv("gamma_decay_energy_keV.csv", sep=" ")

    total_energy_gamma = gamma_df["decay_energy_erg"].sum()

    energy_per_packet = total_energy_gamma / num_decays

    logger.info(f"Total energy in gamma-rays is {total_energy_gamma}")
    logger.info(f"Energy per packet is {energy_per_packet}")

    packet_source = GammaRayPacketSource(
        energy_per_packet,
        isotope_decay_df,
        positronium_fraction,
        inner_velocities,
        outer_velocities,
        inv_volume_time,
        times,
        effective_time_array,
        taus,
        parents,
    )

    logger.info("Creating packets")
    packet_collection, isotope_positron_fraction = packet_source.create_packets(
        cumulative_decays_df, num_decays, seed
    )

    total_energy = np.zeros((number_of_shells, len(times) - 1))

   

    #positron_energy_fraction = packet_source.calculate_positron_energy_fraction(cumulative_decays_df)

    #Bin the positron fraction by time and shell
    
    # for i in range(len(shell_masses)):
    #    for j in range(len(times)):
    #        positron_energy_deposition[i, j] += positron_energy_fraction.loc[j, i] * energy_per_packet


    logger.info("Creating packet list")
    packets = []
    # This for loop is expensive. Need to rewrite GX packet to handle arrays
    packets = [
        GXPacket(
            packet_collection.location[:, i],
            packet_collection.direction[:, i],
            packet_collection.energy_rf[i],
            packet_collection.energy_cmf[i],
            packet_collection.nu_rf[i],
            packet_collection.nu_cmf[i],
            packet_collection.status[i],
            packet_collection.shell[i],
            packet_collection.time_start[i],
            packet_collection.time_index[i],
        )
        for i in range(num_decays)
    ]
    time_current = []
    for i, p in enumerate(packets):
        total_energy[p.shell, p.time_index] \
        += isotope_positron_fraction[i] * energy_per_packet
        time_current.append(p.time_start)
        #print (isotope_positron_fraction[i])
    packet_times, counts = np.unique(time_current, return_counts=True)

    #return packets, time_current
    

    # shell_number = []
    # time_current = []

    # # Bin the frequency of the packets in shell and time

    # packets_nu_cmf_array = np.zeros((number_of_shells, len(times)))
    # packets_nu_rf_array = np.zeros((number_of_shells, len(times)))
    # packets_energy_cmf_array = np.zeros((number_of_shells, len(times)))
    # packets_energy_rf_array = np.zeros((number_of_shells, len(times)))
    # #packets_positron_energy_array = np.zeros((number_of_shells, len(times) -1))

    # positron_energy = pd.DataFrame(
    #      data=positron_energy_deposition, columns=times
    # )


    # for p in packets:
    #     shell_number.append(p.shell)
    #     time_current.append(p.time_start)
    #     packets_nu_cmf_array[p.shell, p.time_index] += p.nu_cmf
    #     packets_nu_rf_array[p.shell, p.time_index] += p.nu_rf
    #     packets_energy_cmf_array[p.shell, p.time_index] += p.energy_cmf
    #     packets_energy_rf_array[p.shell, p.time_index] += p.energy_rf

    # return (
    #     packets,
    #     packets_nu_cmf_array,
    #     packets_nu_rf_array,
    #     packets_energy_cmf_array,
    #     packets_energy_rf_array,
    #     positron_energy,
    # )

    # # return packets

    energy_bins = np.logspace(2, 3.8, spectrum_bins)
    energy_out = np.zeros((len(energy_bins - 1), len(times) - 1))
    energy_deposited = np.zeros((number_of_shells, len(times) - 1))
    packets_info_array = np.zeros((int(num_decays), 8))
    iron_group_fraction = iron_group_fraction_per_shell(model)
    #print ("The iron group fraction is ", iron_group_fraction)

    logger.info("Entering the main gamma-ray loop")

    total_cmf_energy = 0
    total_rf_energy = 0

    for p in packets:
        total_cmf_energy += p.energy_cmf
        total_rf_energy += p.energy_rf

    logger.info(f"Total CMF energy is {total_cmf_energy}")
    logger.info(f"Total RF energy is {total_rf_energy}")
    

    (
        energy_deposited_gamma, 
        total_energy_deposited
    ) = new_gamma_packet_transport(
        packets,
        packet_times,
        counts,
        electron_number_density_time,
        mass_density_time,
        iron_group_fraction.to_numpy(),
        inner_velocities,
        outer_velocities,
        energy_deposited,
        total_energy,
        times, 
        effective_time_array,
    )

    escape_energy = pd.DataFrame(
        data=energy_out, columns=times[:-1], index=energy_bins
    )

    # deposited energy by gamma-rays in ergs
    deposited_energy = pd.DataFrame(
        data=energy_deposited_gamma, columns=times[:-1]
    )
    # total_energy in ergs
    total_energy = pd.DataFrame(
        data=total_energy_deposited, columns=times[:-1]
    )

    print ("Total positron energy is ", total_energy.sum().sum())    

    #total_deposited_energy = (positron_energy + deposited_energy) / dt_array
    total_deposited_energy = total_energy / dt_array

    return (
        deposited_energy,
        total_deposited_energy,
    )


def run_gamma_ray_loop_over_time(model,
    isotope_decay_df,
    cumulative_decays_df,
    num_decays,
    times,
    effective_time_array,
    grey_opacity,
    seed=29,
    positronium_fraction=0.0):

    np.random.seed(seed)
    times = times * u.d.to(u.s)
    effective_time_array = effective_time_array * u.d.to(u.s)
    inner_velocities = model.v_inner.to("cm/s").value
    outer_velocities = model.v_outer.to("cm/s").value
    ejecta_volume = model.volume.to("cm^3").value
    shell_masses = model.volume * model.density
    number_of_shells = len(shell_masses)
    # TODO: decaying upto times[0]. raw_isotope_abundance is possibly not the best name
    raw_isotope_abundance = model.composition.raw_isotope_abundance.sort_values(
        by=["atomic_number", "mass_number"], ascending=False
    )

    ejecta_velocity_volume = calculate_ejecta_velocity_volume(model)

    inv_volume_time = (
        1.0 / ejecta_velocity_volume[:, np.newaxis]
    ) / effective_time_array**3.0

    for (
        atom_number
    ) in model.composition.isotopic_number_density.index.get_level_values(0):
        values = model.composition.isotopic_number_density.loc[
            atom_number
        ].values
        if values.shape[1] > 1:
            model.elemental_number_density.loc[atom_number] = np.sum(
                values, axis=0
            )
        else:
            model.composition.elemental_number_density.loc[atom_number] = values
    

    # Electron number density
    electron_number_density = model.composition.elemental_number_density.mul(
        model.composition.elemental_number_density.index,
        axis=0,
    ).sum()
    print ("The electron number density is ", electron_number_density)
    electron_number = np.array(electron_number_density * ejecta_volume)
    #print ("The electron number is ", electron_number)

    # Evolve electron number and mass density with time
    electron_number_density_time = (
        electron_number[:, np.newaxis] * inv_volume_time
    )
    #print ("The electron number density time is ", electron_number_density_time)    
    mass_density_time = shell_masses[:, np.newaxis] * inv_volume_time
    #print ("The mass density time is ", mass_density_time)

    taus, parents = get_taus(raw_isotope_abundance)
    # Need to get the strings for the isotopes without the dashes
    taus = make_isotope_string_tardis_like(taus)

    gamma_df = cumulative_decays_df[cumulative_decays_df["radiation"] == "g"]
    gamma_df_decay_energy_keV = gamma_df["decay_energy_keV"]
    #print (gamma_df_decay_energy_keV.groupby("shell_number").sum())
    #gamma_df_decay_energy_keV.to_csv("gamma_decay_energy_keV.csv", sep=" ")

    total_energy_gamma = gamma_df["decay_energy_erg"].sum()

    energy_per_packet = total_energy_gamma / num_decays
    iron_group_fraction = iron_group_fraction_per_shell(model)

    #logger.info(f"Total energy in gamma-rays is {total_energy_gamma}")
    #logger.info(f"Energy per packet is {energy_per_packet}")

    packet_source = GammaRayPacketSource(
        energy_per_packet,
        isotope_decay_df,
        positronium_fraction,
        inner_velocities,
        outer_velocities,
        inv_volume_time,
        times,
        effective_time_array,
        taus,
        parents,
    )

    #logger.info("Creating packets")
    packet_collection, isotope_positron_fraction = packet_source.create_packets(
        cumulative_decays_df, num_decays, seed
    )

    packets = []
    # This for loop is expensive. Need to rewrite GX packet to handle arrays
    packets = [
        GXPacket(
            packet_collection.location[:, i],
            packet_collection.direction[:, i],
            packet_collection.energy_rf[i],
            packet_collection.energy_cmf[i],
            packet_collection.nu_rf[i],
            packet_collection.nu_cmf[i],
            packet_collection.status[i],
            packet_collection.shell[i],
            packet_collection.time_start[i],
            packet_collection.time_index[i],
        )
        for i in range(num_decays)
    ]

    # Choose packets which are within the a particular time step

    #time_start = times[0]
    #time_end = times[1]

    # Find which packet lies in which time step
    print ("Propagating all particles over a single time step")

    n_alive = num_decays

    for t in range(len(times) - 1):

        time_start = times[t]
        time_end = times[t + 1]
        print ("start time:", time_start)
        print ("end time:", time_end)    

        packets_active, n_active, n_escaped, n_end, n_deposited = (
        propagate_packet_over_time(time_start,
                               time_end,
                               n_alive,
                               packets,
                               grey_opacity,
                               electron_number_density_time,
                               mass_density_time,
                               iron_group_fraction,
                               inner_velocities,
                               outer_velocities,
                               times)
        )

        n_alive -= n_escaped + n_end + n_deposited
        print (f"Number of packets alive at time {time_end} is {n_alive}")

    





    
