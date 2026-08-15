import numpy as np
from numba import get_thread_id, njit, prange

from tardis.energy_input.transport.gamma_ray_grid import (
    distance_trace,
    move_packet,
)
from tardis.energy_input.transport.gamma_ray_interactions import (
    compton_scatter,
    get_compton_fraction_artis,
    pair_creation_packet,
    scatter_type,
)
from tardis.energy_input.transport.GXPacket import GXPacket, GXPacketStatus
from tardis.energy_input.util import (
    C_CGS,
    H_CGS_KEV,
    doppler_factor_3d,
    get_index,
)
from tardis.opacities.opacities import (
    SIGMA_T,
    compton_opacity_calculation,
    kappa_calculation,
    pair_creation_opacity_artis,
    pair_creation_opacity_calculation,
    photoabsorption_opacity_calculation,
)
from tardis.transport.montecarlo import njit_dict, njit_dict_no_parallel


@njit(**njit_dict_no_parallel)
def gamma_packet_loop_single_packet(
    packet_index,
    location,
    direction,
    energy_rf,
    energy_cmf,
    nu_rf,
    nu_cmf,
    status,
    shell,
    time_start,
    time_index,
    grey_opacity,
    photoabsorption_opacity_type,
    pair_creation_opacity_type,
    electron_number_density_time,
    mass_density_time,
    iron_group_fraction_per_shell,
    inner_velocities,
    outer_velocities,
    dt_array,
    times,
    effective_time_array,
    energy_bins,
    energy_out_thread,
    energy_out_cosi_thread,
    energy_deposited_gamma_thread,
    total_energy_thread,
    packets_info_array,
):
    """Propagate one packet, writing estimators to its worker's buffers."""
    packet = GXPacket(
        location[:, packet_index],
        direction[:, packet_index],
        energy_rf[packet_index],
        energy_cmf[packet_index],
        nu_rf[packet_index],
        nu_cmf[packet_index],
        status[packet_index],
        shell[packet_index],
        time_start[packet_index],
        time_index[packet_index],
    )
    packet_time_index = packet.time_index
    worker = get_thread_id()
    luminosity = 0.0

    if packet_time_index < 0:
        raise ValueError("Packet time index less than 0!")

    while packet.status == GXPacketStatus.IN_PROCESS:
        dt = dt_array[packet_time_index]
        comoving_energy = H_CGS_KEV * packet.nu_cmf

        if grey_opacity < 0:
            doppler_factor = doppler_factor_3d(
                packet.direction, packet.location, times[packet_time_index]
            )
            kappa = kappa_calculation(comoving_energy)
            if kappa < 1e-2:
                compton_opacity = SIGMA_T * electron_number_density_time[
                    packet.shell, packet_time_index
                ]
            else:
                compton_opacity = compton_opacity_calculation(
                    comoving_energy,
                    electron_number_density_time[
                        packet.shell, packet_time_index
                    ],
                )

            if photoabsorption_opacity_type == "kasen":
                photoabsorption_opacity = 0.0
            elif photoabsorption_opacity_type == "tardis":
                photoabsorption_opacity = photoabsorption_opacity_calculation(
                    comoving_energy,
                    mass_density_time[packet.shell, packet_time_index],
                    iron_group_fraction_per_shell[packet.shell],
                )
            else:
                raise ValueError("Invalid photoabsorption opacity type!")

            if pair_creation_opacity_type == "artis":
                pair_creation_opacity = pair_creation_opacity_artis(
                    comoving_energy,
                    mass_density_time[packet.shell, packet_time_index],
                    iron_group_fraction_per_shell[packet.shell],
                )
            elif pair_creation_opacity_type == "tardis":
                pair_creation_opacity = pair_creation_opacity_calculation(
                    comoving_energy,
                    mass_density_time[packet.shell, packet_time_index],
                    iron_group_fraction_per_shell[packet.shell],
                )
            else:
                raise ValueError("Invalid pair creation opacity type!")
        else:
            compton_opacity = 0.0
            pair_creation_opacity = 0.0
            photoabsorption_opacity = grey_opacity * mass_density_time[
                packet.shell, packet_time_index
            ]
            doppler_factor = 1.0

        total_opacity = (
            compton_opacity + photoabsorption_opacity + pair_creation_opacity
        ) * doppler_factor
        packet.tau = -np.log(np.random.random())
        (
            distance_interaction,
            distance_boundary,
            distance_time,
            shell_change,
        ) = distance_trace(
            packet,
            inner_velocities,
            outer_velocities,
            total_opacity,
            effective_time_array[packet_time_index],
            times[packet_time_index + 1],
        )
        distance = min(distance_interaction, distance_boundary, distance_time)
        packet.time_start += distance / C_CGS
        packet = move_packet(packet, distance)

        if distance == distance_time:
            packet_time_index += 1
            if packet_time_index > len(effective_time_array) - 1:
                packet.status = GXPacketStatus.END
            else:
                packet.shell = get_index(
                    packet.get_location_r(),
                    inner_velocities * times[packet_time_index],
                )
        elif distance == distance_interaction:
            packet.status = scatter_type(
                compton_opacity, photoabsorption_opacity, total_opacity
            )
            packet, ejecta_energy_gained = process_packet_path(packet)
            energy_deposited_gamma_thread[
                worker, packet.shell, packet_time_index
            ] += ejecta_energy_gained
            total_energy_thread[worker, packet.shell, packet_time_index] += (
                ejecta_energy_gained
            )
            if packet.status == GXPacketStatus.PHOTOABSORPTION:
                break
            packet.status = GXPacketStatus.IN_PROCESS
        else:
            packet.shell += shell_change
            if packet.shell > len(mass_density_time[:, 0]) - 1:
                rest_energy = packet.nu_rf * H_CGS_KEV
                bin_index = get_index(rest_energy, energy_bins)
                bin_width = energy_bins[bin_index + 1] - energy_bins[bin_index]
                freq_bin_width = bin_width / H_CGS_KEV
                energy_out_thread[worker, bin_index, packet_time_index] += (
                    packet.energy_rf / dt / freq_bin_width
                )
                energy_out_cosi_thread[
                    worker, bin_index, packet_time_index
                ] += 1 / dt / bin_width
                luminosity = packet.energy_rf / dt
                packet.status = GXPacketStatus.ESCAPED
            elif packet.shell < 0:
                packet.energy_rf = 0.0
                packet.energy_cmf = 0.0
                packet.status = GXPacketStatus.END

    packets_info_array[packet_index] = np.array(
        [packet_index, packet.status, packet.nu_cmf, packet.nu_rf,
         packet.energy_cmf, luminosity, packet.energy_rf, packet.shell]
    )


@njit(**njit_dict)
def gamma_packet_loop(
    location, direction, energy_rf, energy_cmf, nu_rf, nu_cmf, status, shell,
    time_start, time_index, grey_opacity, photoabsorption_opacity_type,
    pair_creation_opacity_type, electron_number_density_time, mass_density_time,
    iron_group_fraction_per_shell, inner_velocities, outer_velocities, dt_array,
    times, effective_time_array, energy_bins, energy_out_thread,
    energy_out_cosi_thread, energy_deposited_gamma_thread, total_energy_thread,
    packets_info_array,
):
    """Run the independent single-packet kernel in parallel.

    Estimators are indexed by worker to avoid races.  The caller reduces those
    buffers after this loop completes.
    """
    for packet_index in prange(energy_rf.size):
        gamma_packet_loop_single_packet(
            packet_index, location, direction, energy_rf, energy_cmf, nu_rf,
            nu_cmf, status, shell, time_start, time_index, grey_opacity,
            photoabsorption_opacity_type, pair_creation_opacity_type,
            electron_number_density_time, mass_density_time,
            iron_group_fraction_per_shell, inner_velocities, outer_velocities,
            dt_array,
            times,
            effective_time_array,
            energy_bins,
            energy_out_thread,
            energy_out_cosi_thread, energy_deposited_gamma_thread,
            total_energy_thread, packets_info_array,
        )


@njit(**njit_dict_no_parallel)
def process_packet_path(packet):
    """Move the packet through interactions

    Parameters
    ----------
    packet : GXPacket
        Packet for processing

    Returns
    -------
    GXPacket
        Packet after processing
    float
        Energy injected into the ejecta
    """
    if packet.status == GXPacketStatus.COMPTON_SCATTER:
        comoving_freq_energy = packet.nu_cmf * H_CGS_KEV

        compton_angle, compton_fraction = get_compton_fraction_artis(
            comoving_freq_energy
        )

        # Packet is no longer a gamma-ray, destroy it
        if np.random.random() < 1 / compton_fraction:
            packet.nu_cmf = packet.nu_cmf / compton_fraction

            packet.direction = compton_scatter(packet, compton_angle)

            # Calculate rest frame frequency after scaling by the fraction that remains
            doppler_factor = doppler_factor_3d(
                packet.direction,
                packet.location,
                packet.time_start,
            )

            packet.nu_rf = packet.nu_cmf / doppler_factor
            packet.energy_rf = packet.energy_cmf / doppler_factor

            ejecta_energy_gained = 0.0
        else:
            packet.status = GXPacketStatus.PHOTOABSORPTION

    if packet.status == GXPacketStatus.PAIR_CREATION:
        packet = pair_creation_packet(packet)
        ejecta_energy_gained = 0.0

    if packet.status == GXPacketStatus.PHOTOABSORPTION:
        # Ejecta gains comoving energy
        ejecta_energy_gained = packet.energy_cmf

    return packet, ejecta_energy_gained
