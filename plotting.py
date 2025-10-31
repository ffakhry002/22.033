"""
Visualization script for PWR Fusion Breeder Reactor simulation results.
Creates comprehensive plots of flux distributions and tritium breeding performance.
Modified to include surface current tallies visualization with OUTWARD ONLY current.
"""
import os
import sys
import openmc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import local modules
from tallies import calc_norm_factor
from inputs import inputs, get_derived_dimensions


def plot_surface_current_spectrum_comparison(sp, plot_dir):
    print("\n" + "="*70)
    print("Creating Surface Current Energy Spectrum Comparison Plots")
    print("="*70)

    # Calculate normalization factor
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)
    derived = get_derived_dimensions()

    # Define surfaces to plot
    surfaces = ['core', 'outer_tank', 'rpv_inner', 'rpv_outer', 'lithium']
    surface_titles = {
        'core': 'Core Boundary',
        'outer_tank': 'Outer Tank Boundary',
        'rpv_inner': 'RPV Inner Boundary',
        'rpv_outer': 'RPV Outer Boundary',
        'lithium': 'Lithium Blanket Boundary'
    }
    surface_radii = {
        'core': inputs['r_core'],
        'outer_tank': derived['r_outer_tank'],
        'rpv_inner': derived['r_rpv_1'],
        'rpv_outer': derived['r_rpv_2'],
        'lithium': derived['r_lithium']
    }

    # Add moderator region surfaces if enabled
    if inputs['enable_moderator_region']:
        surfaces.insert(-1, 'moderator')  # Insert before lithium
        surfaces.insert(-1, 'wall_divider')  # Insert before lithium
        surface_titles['moderator'] = 'Moderator Boundary'
        surface_titles['wall_divider'] = 'Wall Divider Boundary'
        surface_radii['moderator'] = derived['r_moderator']
        surface_radii['wall_divider'] = derived['r_wall_divider']
    height = derived['z_fuel_top'] - derived['z_fuel_bottom']
    surface_areas = {}
    for surf_name, radius in surface_radii.items():
        surface_areas[surf_name] = 2 * np.pi * radius * height  # cm²
    energy_bins_eV = {
        'Thermal': (0.0, 0.625),  # eV
        'Epithermal': (0.625, 1e5),  # eV
        'Fast': (1e5, 2e7)  # eV
    }
    energy_bins_MeV = {
        name: (e_min/1e6, e_max/1e6) for name, (e_min, e_max) in energy_bins_eV.items()
    }

    # Create figure with 5 rows x 3 columns (one row per surface)
    fig, axes = plt.subplots(len(surfaces), 3, figsize=(20, 4*len(surfaces)))

    for row_idx, surface in enumerate(surfaces):
        surface_area = surface_areas[surface]
        tally_name = f'surface_{surface}_current_log1001'
        try:
            tally = sp.get_tally(name=tally_name)
        except Exception as e:
            print(f"Warning: Could not find tally {tally_name}, skipping...")
            continue
        energy_filter = tally.find_filter(openmc.EnergyFilter)
        energy_bins_array = energy_filter.bins
        energy_mids_eV = np.sqrt(energy_bins_array[:, 0] * energy_bins_array[:, 1])
        energy_mids_MeV = energy_mids_eV / 1e6
        current_outward = tally.mean.flatten() * norm_factor
        current_std = tally.std_dev.flatten() * norm_factor

        current_density = current_outward / surface_area
        current_density_std = current_std / surface_area

        current_density = np.maximum(current_density, 1e-30)
        three_group_currents = {}
        for group in ['thermal', 'epithermal', 'fast']:
            tally_name_3g = f'surface_{surface}_current_{group}'
            try:
                tally_3g = sp.get_tally(name=tally_name_3g)
                # Get absolute current and convert to density
                abs_current = tally_3g.mean[0, 0, 0] * norm_factor
                three_group_currents[group.capitalize()] = abs_current / surface_area
            except:
                three_group_currents[group.capitalize()] = 0.0

        # Calculate normalized fractions
        total_current = sum(three_group_currents.values())
        fractions = {name: curr/total_current if total_current > 0 else 0
                    for name, curr in three_group_currents.items()}

        ax_left = axes[row_idx, 0] if len(surfaces) > 1 else axes[0]

        ax_left.loglog(energy_mids_MeV, current_density, 'b-', linewidth=2, label='log1001 Spectrum (Outward)')

        ax_left.set_xlabel('Energy [MeV]', fontsize=11)
        ax_left.set_ylabel('Outward Current Density [n/cm²/s]', fontsize=11)
        ax_left.set_title(f'{surface_titles[surface]} (r={surface_radii[surface]:.1f} cm) - Log-Log',
                         fontsize=12, fontweight='bold')
        ax_left.legend(loc='best', fontsize=8)
        ax_left.grid(True, which='both', alpha=0.3)
        ax_left.set_xlim(1e-12, 20)  # 1e-6 eV to 20 MeV in MeV units

        # Set y-axis limits to handle positive values only
        valid_current = current_density[current_density > 1e-30]
        if len(valid_current) > 0:
            y_min = np.min(valid_current) * 0.1
            y_max = np.max(valid_current) * 10
            ax_left.set_ylim(y_min, y_max)

        ax_middle = axes[row_idx, 1] if len(surfaces) > 1 else axes[1]

        ax_middle.semilogx(energy_mids_MeV, current_density, 'b-', linewidth=2, label='log1001 Spectrum (Outward)')

        ax_middle.set_xlabel('Energy [MeV]', fontsize=11)
        ax_middle.set_ylabel('Outward Current Density [n/cm²/s]', fontsize=11)
        ax_middle.set_title(f'{surface_titles[surface]} (r={surface_radii[surface]:.1f} cm) - Linear-Log',
                           fontsize=12, fontweight='bold')
        ax_middle.legend(loc='best', fontsize=8)
        ax_middle.grid(True, which='both', alpha=0.3)
        ax_middle.set_xlim(1e-12, 20)  # 1e-6 eV to 20 MeV in MeV units

        ax_right = axes[row_idx, 2] if len(surfaces) > 1 else axes[2]

        colors_3g = {'Thermal': 'green', 'Epithermal': 'orange', 'Fast': 'red'}
        group_order = ['Thermal', 'Epithermal', 'Fast']
        current_levels = []

        for group_name in group_order:
            e_min, e_max = energy_bins_MeV[group_name]
            if group_name in three_group_currents and three_group_currents[group_name] > 0:
                current_val = three_group_currents[group_name]
                frac = fractions[group_name]
                current_levels.append((group_name, current_val, e_min, e_max))

                # Horizontal line
                ax_right.hlines(current_val, e_min, e_max,
                              colors=colors_3g[group_name], linewidth=4,
                              label=f'{group_name}: {frac:.3f}', alpha=0.8)

        for i in range(len(current_levels) - 1):
            curr_group, curr_current, _, curr_e_max = current_levels[i]
            next_group, next_current, next_e_min, _ = current_levels[i + 1]

            ax_right.vlines(curr_e_max, curr_current, next_current,
                          colors=colors_3g[curr_group], linewidth=2.5,
                          linestyles='-', alpha=0.8)

        ax_right.set_xlabel('Energy [MeV]', fontsize=11)
        ax_right.set_ylabel('Outward Current Density [n/cm²/s]', fontsize=11)
        ax_right.set_title(f'{surface_titles[surface]} (r={surface_radii[surface]:.1f} cm) - Three-Group',
                          fontsize=12, fontweight='bold')
        ax_right.set_xscale('log')
        ax_right.set_yscale('log')

        ax_right.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)

        ax_right.grid(True, which='both', alpha=0.3)
        ax_right.set_xlim(1e-12, 20)  # 1e-6 eV to 20 MeV in MeV units

    plt.tight_layout()
    plt.savefig(plot_dir / 'surface_current_energy_spectrum_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_tritium_production(sp):
    """Print tritium production rates in various units."""
    print("\n" + "="*70)
    print("Tritium Production Summary")
    print("="*70)

    # Calculate normalization
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)

    # Get tritium breeding tallies (all per source neutron)
    try:
        # Get tritium breeding ratio tally
        tbr_tally = sp.get_tally(name='tritium_breeding_ratio')
        tritium_per_source = tbr_tally.mean[0, 0, 0]  # T atoms/source neutron
        tritium_per_source_std = tbr_tally.std_dev[0, 0, 0]

        # Get Li-6 and Li-7 contributions (per source)
        try:
            li6_tally = sp.get_tally(name='tritium_breeding_li6')
            li6_per_source = li6_tally.mean[0, 0, 0]
            li6_per_source_std = li6_tally.std_dev[0, 0, 0]
        except:
            li6_per_source = 0.0
            li6_per_source_std = 0.0

        try:
            li7_tally = sp.get_tally(name='tritium_breeding_li7')
            li7_per_source = li7_tally.mean[0, 0, 0]
            li7_per_source_std = li7_tally.std_dev[0, 0, 0]
        except:
            li7_per_source = 0.0
            li7_per_source_std = 0.0

        # =====================================================================
        # TRITIUM BREEDING RATIO (TBR) - Tritium per Source Neutron
        # =====================================================================
        print(f"\nTritium Breeding Ratio (TBR) [T atoms/source neutron]:")
        print(f"  Total: {tritium_per_source:.6f} ± {tritium_per_source_std:.6f}")
        if li6_per_source > 0:
            print(f"  Li-6:  {li6_per_source:.6f} ± {li6_per_source_std:.6f}")
        if li7_per_source > 0:
            print(f"  Li-7:  {li7_per_source:.6f} ± {li7_per_source_std:.6f}")

        # =====================================================================
        # ABSOLUTE TRITIUM PRODUCTION RATE
        # =====================================================================
        # T_rate = (T/source) × (sources/s) = T/s
        t_production = tritium_per_source * norm_factor  # T atoms/s
        t_production_std = tritium_per_source_std * norm_factor

        # Convert to grams (tritium-3 has atomic mass of 3.0 g/mol)
        atoms_to_grams = 3.0 / 6.022e23  # g/atom

        # Calculate rates
        g_per_s = t_production * atoms_to_grams
        g_per_day = g_per_s * 86400
        g_per_week = g_per_day * 7
        g_per_year = g_per_day * 365.25

        # Calculate uncertainties
        g_per_s_std = t_production_std * atoms_to_grams
        g_per_day_std = g_per_s_std * 86400
        g_per_week_std = g_per_day_std * 7
        g_per_year_std = g_per_day_std * 365.25

        print(f"\nTritium Production Rate:")
        print(f"  {g_per_s:.6e} ± {g_per_s_std:.6e} g/s")
        print(f"  {g_per_day:.6e} ± {g_per_day_std:.6e} g/day")
        print(f"  {g_per_week:.6e} ± {g_per_week_std:.6e} g/week")
        print(f"  {g_per_year:.6e} ± {g_per_year_std:.6e} g/year")

        # =====================================================================
        # TRITIUM PER FISSION (TPF) and Related Metrics
        # =====================================================================
        try:
            fission_tally = sp.get_tally(name='fission')
            nu_fission_tally = sp.get_tally(name='nu-fission')

            fissions_per_source = fission_tally.mean[0, 0, 0]  # fissions/source
            fissions_per_source_std = fission_tally.std_dev[0, 0, 0]
            nu_fissions_per_source = nu_fission_tally.mean[0, 0, 0]  # neutrons/source

            # Calculate nu (neutrons per fission)
            nu = nu_fissions_per_source / fissions_per_source  # neutrons/fission

            # Calculate TPF (tritium per fission)
            # TPF = (T/source) / (fissions/source) = T/fission
            TPF = tritium_per_source / fissions_per_source  # T atoms/fission
            TPF_std = tritium_per_source_std / fissions_per_source  # Simplified error propagation

            # Calculate component TPFs
            TPF_li6 = li6_per_source / fissions_per_source if li6_per_source > 0 else 0.0
            TPF_li6_std = li6_per_source_std / fissions_per_source if li6_per_source > 0 else 0.0
            TPF_li7 = li7_per_source / fissions_per_source if li7_per_source > 0 else 0.0
            TPF_li7_std = li7_per_source_std / fissions_per_source if li7_per_source > 0 else 0.0

            print(f"\nTritium Per Fission (TPF) [T atoms/fission]:")
            print(f"  Total: {TPF:.6f} ± {TPF_std:.6f}")
            if li6_per_source > 0:
                print(f"  Li-6:  {TPF_li6:.6f} ± {TPF_li6_std:.6f}")
            if li7_per_source > 0:
                print(f"  Li-7:  {TPF_li7:.6f} ± {TPF_li7_std:.6f}")

            print(f"\nNeutron multiplication:")
            print(f"  ν (neutrons/fission): {nu:.4f}")

            # Calculate and print fission rate for reference
            fission_rate = fissions_per_source * norm_factor  # fissions/s
            fission_rate_std = fissions_per_source_std * norm_factor
            print(f"\nFission Rate:")
            print(f"  {fission_rate:.6e} ± {fission_rate_std:.6e} fissions/s")

        except Exception as e:
            print(f"\n  (TPF calculation unavailable: {e})")

    except Exception as e:
        print(f"\nWarning: Could not calculate tritium production: {e}")

    print("="*70)


def plot_keff_entropy(sp, plot_dir):
    print("\n" + "="*70)
    print("Creating k-effective and Entropy Plots")
    print("="*70)

    # Get entropy data and batch numbers
    entropy = sp.entropy
    n_batches = len(entropy)
    batches = np.arange(1, n_batches + 1)

    # Get k-effective data
    k_data = sp.k_generation

    # Get number of inactive batches from inputs
    n_inactive = inputs['inactive']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])

    # Plot k-effective vs batch number
    ax1.plot(batches, k_data, 'b-', linewidth=1, label='k-effective')
    ax1.axvline(x=n_inactive, color='red', linestyle='--',
                label=f'Active Batches Start (n={n_inactive})')
    ax1.set_ylabel('k', fontsize=11)
    ax1.set_title('k-effective Convergence', fontsize=12, fontweight='bold')
    ax1.grid(True, which='major', linestyle='-', alpha=0.2)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.1)
    ax1.legend(fontsize=10)

    # Plot entropy vs batch number
    ax2.plot(batches, entropy, 'b-', linewidth=1, label='Shannon Entropy')
    ax2.axvline(x=n_inactive, color='red', linestyle='--',
                label=f'Active Batches Start (n={n_inactive})')
    ax2.set_xlabel('Batch', fontsize=11)
    ax2.set_ylabel('Shannon Entropy', fontsize=11)
    ax2.set_title('Shannon Entropy Convergence', fontsize=12, fontweight='bold')
    ax2.grid(True, which='major', linestyle='-', alpha=0.2)
    ax2.grid(True, which='minor', linestyle=':', alpha=0.1)
    ax2.legend(fontsize=10)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.3)

    # Save plot directly to plots directory
    plt.savefig(plot_dir / 'entropy.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_energy_spectrum_comparison(sp, plot_dir):
    """Create plots showing log1001 and three-group flux spectra for all regions.

    Creates comparison plots for outer_tank, rpv_inner, rpv_outer, and lithium_wall regions.
    Column 1: log-log scale with spectrum ONLY
    Column 2: linear-log scale with spectrum ONLY
    Column 3: three-group ONLY with fractions, log-log scale

    Parameters
    ----------
    sp : openmc.StatePoint
        StatePoint file containing tally results
    plot_dir : Path
        Directory to save plots
    """
    print("\n" + "="*70)
    print("Creating Energy Spectrum Comparison Plots")
    print("="*70)

    # Calculate normalization factor
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)

    # Get dimensions
    derived = get_derived_dimensions()

    # Define regions to plot - include both RPV layers
    regions = ['outer_tank', 'rpv_inner', 'rpv_outer', 'lithium_blanket']
    region_titles = {
        'outer_tank': 'Outer Tank (Cold Coolant)',
        'rpv_inner': 'RPV Inner Layer',
        'rpv_outer': 'RPV Outer Layer',
        'lithium_blanket': 'Lithium Breeding Blanket'
    }

    # Add moderator region if enabled
    if inputs['enable_moderator_region']:
        regions.insert(-1, 'moderator')  # Insert before lithium_blanket
        region_titles['moderator'] = 'Moderator Region'

    # Calculate cell volumes for each region
    # These are cylindrical shell volumes
    r_core = inputs['r_core']
    r_outer_tank = derived['r_outer_tank']
    r_rpv_1 = derived['r_rpv_1']
    r_rpv_2 = derived['r_rpv_2']
    r_lithium = derived['r_lithium']
    r_lithium_wall = derived['r_lithium_wall']
    height = derived['z_top'] - derived['z_bottom']

    # Calculate volumes (cylindrical shells: π * (r_outer² - r_inner²) * height)
    # Determine inner radius for lithium blanket based on moderator configuration
    if inputs['enable_moderator_region']:
        r_wall_divider = derived['r_wall_divider']
        lithium_blanket_r_inner = r_wall_divider
    else:
        lithium_blanket_r_inner = r_rpv_2

    region_volumes = {
        'outer_tank': np.pi * (r_outer_tank**2 - r_core**2) * height,
        'rpv_inner': np.pi * (r_rpv_1**2 - r_outer_tank**2) * height,
        'rpv_outer': np.pi * (r_rpv_2**2 - r_rpv_1**2) * height,
        'lithium_blanket': np.pi * (r_lithium**2 - lithium_blanket_r_inner**2) * height
    }

    # Add moderator volume if enabled
    if inputs['enable_moderator_region']:
        r_moderator = derived['r_moderator']
        region_volumes['moderator'] = np.pi * (r_moderator**2 - r_rpv_2**2) * height

    # Energy boundaries for three-group structure (in eV, will convert to MeV for plotting)
    energy_bins_eV = {
        'Thermal': (0.0, 0.625),  # eV
        'Epithermal': (0.625, 1e5),  # eV
        'Fast': (1e5, 2e7)  # eV
    }

    # Convert to MeV for plotting
    energy_bins_MeV = {
        name: (e_min/1e6, e_max/1e6) for name, (e_min, e_max) in energy_bins_eV.items()
    }

    # Create figure with 4 rows x 3 columns
    fig, axes = plt.subplots(len(regions), 3, figsize=(20, 4*len(regions)))

    for row_idx, region in enumerate(regions):
        # Get cell volume for this region
        cell_volume = region_volumes[region]

        # Get log1001 tally
        tally_name = f'{region}_flux_log1001'
        try:
            tally = sp.get_tally(name=tally_name)
        except Exception as e:
            print(f"Warning: Could not find tally {tally_name}, skipping...")
            continue

        # Get energy filter and extract data
        energy_filter = tally.find_filter(openmc.EnergyFilter)
        energy_bins_array = energy_filter.bins  # Shape: (n_bins, 2) - in eV
        energy_mids_eV = np.sqrt(energy_bins_array[:, 0] * energy_bins_array[:, 1])  # eV
        energy_mids_MeV = energy_mids_eV / 1e6  # Convert to MeV

        # Apply normalization AND divide by cell volume to get flux density
        flux = tally.mean.flatten() * norm_factor / cell_volume
        flux_std = tally.std_dev.flatten() * norm_factor / cell_volume

        # Get three-group tallies and normalize
        three_group_fluxes = {}
        for group in ['thermal', 'epithermal', 'fast']:
            tally_name_3g = f'{region}_flux_{group}'
            try:
                tally_3g = sp.get_tally(name=tally_name_3g)
                three_group_fluxes[group.capitalize()] = tally_3g.mean[0, 0, 0] * norm_factor / cell_volume
            except:
                three_group_fluxes[group.capitalize()] = 0.0

        # Calculate normalized fractions
        total_flux = sum(three_group_fluxes.values())
        fractions = {name: flux/total_flux if total_flux > 0 else 0
                    for name, flux in three_group_fluxes.items()}

        # --- Column 1: log-log scale (SPECTRUM ONLY) ---
        ax_left = axes[row_idx, 0] if len(regions) > 1 else axes[0]

        ax_left.loglog(energy_mids_MeV, flux, 'b-', linewidth=2, label='log1001 Spectrum')

        ax_left.set_xlabel('Energy [MeV]', fontsize=11)
        ax_left.set_ylabel('Flux [n/cm²/s]', fontsize=11)
        ax_left.set_title(f'{region_titles[region]} - Log-Log', fontsize=12, fontweight='bold')
        ax_left.legend(loc='best', fontsize=8)
        ax_left.grid(True, which='both', alpha=0.3)
        ax_left.set_xlim(1e-12, 20)  # 1e-6 eV to 20 MeV in MeV units

        # --- Column 2: linear-log scale (SPECTRUM ONLY) ---
        ax_middle = axes[row_idx, 1] if len(regions) > 1 else axes[1]

        ax_middle.semilogx(energy_mids_MeV, flux, 'b-', linewidth=2, label='log1001 Spectrum')

        ax_middle.set_xlabel('Energy [MeV]', fontsize=11)
        ax_middle.set_ylabel('Flux [n/cm²/s]', fontsize=11)
        ax_middle.set_title(f'{region_titles[region]} - Linear-Log', fontsize=12, fontweight='bold')
        ax_middle.legend(loc='best', fontsize=8)
        ax_middle.grid(True, which='both', alpha=0.3)
        ax_middle.set_xlim(1e-12, 20)  # 1e-6 eV to 20 MeV in MeV units

        # --- Column 3: three-group ONLY with fractions, log-log ---
        ax_right = axes[row_idx, 2] if len(regions) > 1 else axes[2]

        # Plot ONLY three-group data with solid vertical connectors
        colors_3g = {'Thermal': 'green', 'Epithermal': 'orange', 'Fast': 'red'}

        # Store flux values for drawing connectors
        group_order = ['Thermal', 'Epithermal', 'Fast']
        flux_levels = []

        for group_name in group_order:
            e_min, e_max = energy_bins_MeV[group_name]
            if group_name in three_group_fluxes and three_group_fluxes[group_name] > 0:
                flux_val = three_group_fluxes[group_name]
                frac = fractions[group_name]
                flux_levels.append((group_name, flux_val, e_min, e_max))

                # Horizontal line
                ax_right.hlines(flux_val, e_min, e_max,
                              colors=colors_3g[group_name], linewidth=4,
                              label=f'{group_name}: {frac:.3f}', alpha=0.8)

        # Draw solid vertical connectors between adjacent groups
        for i in range(len(flux_levels) - 1):
            curr_group, curr_flux, _, curr_e_max = flux_levels[i]
            next_group, next_flux, next_e_min, _ = flux_levels[i + 1]

            # Draw solid vertical line at the boundary connecting the two flux levels
            ax_right.vlines(curr_e_max, curr_flux, next_flux,
                          colors=colors_3g[curr_group], linewidth=2.5,
                          linestyles='-', alpha=0.8)

        ax_right.set_xlabel('Energy [MeV]', fontsize=11)
        ax_right.set_ylabel('Flux [n/cm²/s]', fontsize=11)
        ax_right.set_title(f'{region_titles[region]} - Three-Group', fontsize=12, fontweight='bold')
        ax_right.set_xscale('log')
        ax_right.set_yscale('log')

        # Move legend outside to the right
        ax_right.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)

        ax_right.grid(True, which='both', alpha=0.3)
        ax_right.set_xlim(1e-12, 20)  # 1e-6 eV to 20 MeV in MeV units

    plt.tight_layout()
    plt.savefig(plot_dir / 'flux_energy_spectrum_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_mesh_flux(sp, plot_dir):
    """Create comprehensive mesh flux visualization plots with correct XY orientation.

    CRITICAL: Assumes mesh is stored as (nz, ny, nx) - Z (axial) first, then Y, then X

    Creates:
    1. 2x2 Heatmaps: XY axially averaged for Total, Fast, Epithermal, Thermal
    2. Radial profiles: Axially averaged and midplane (all 4 flux types on same plot)
    3. Axial profiles: At different radial locations (all 4 flux types on same plot)
    """
    print("\n" + "="*70)
    print("Creating Mesh Flux Visualization Plots")
    print("="*70)

    # Calculate normalization factor
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)

    # Get dimensions
    derived = get_derived_dimensions()

    # Get the mesh tally
    mesh_tally = sp.get_tally(name='mesh_total_flux')
    mesh_filter = mesh_tally.find_filter(openmc.MeshFilter)
    mesh = mesh_filter.mesh
    shape = mesh.dimension  # OpenMC gives [nx, ny, nz] but data might be stored as [nz, ny, nx]

    # Calculate mesh volume
    dx = 2 * derived['r_lithium_wall'] / shape[0]
    dy = 2 * derived['r_lithium_wall'] / shape[1]
    dz = (derived['z_top'] - derived['z_bottom']) / shape[2]
    mesh_volume = dx * dy * dz

    # Get all flux data and normalize
    total_flux_tally = sp.get_tally(name='mesh_total_flux')
    thermal_flux_tally = sp.get_tally(name='mesh_thermal_flux')
    epithermal_flux_tally = sp.get_tally(name='mesh_epithermal_flux')
    fast_flux_tally = sp.get_tally(name='mesh_fast_flux')

    # Reshape data - CRITICAL: Check if data is stored as (nz, ny, nx) instead of (nx, ny, nz)
    # We'll reshape to (nz, ny, nx) assuming Z is the fastest varying in storage
    nx, ny, nz = shape
    flux_total_raw = total_flux_tally.mean.reshape((nz, ny, nx)) * norm_factor / mesh_volume
    flux_thermal_raw = thermal_flux_tally.mean.reshape((nz, ny, nx)) * norm_factor / mesh_volume
    flux_epithermal_raw = epithermal_flux_tally.mean.reshape((nz, ny, nx)) * norm_factor / mesh_volume
    flux_fast_raw = fast_flux_tally.mean.reshape((nz, ny, nx)) * norm_factor / mesh_volume

    # Transpose to get (nx, ny, nz) for easier slicing
    flux_total = flux_total_raw.transpose(2, 1, 0)
    flux_thermal = flux_thermal_raw.transpose(2, 1, 0)
    flux_epithermal = flux_epithermal_raw.transpose(2, 1, 0)
    flux_fast = flux_fast_raw.transpose(2, 1, 0)

    flux_data = {
        'Total': flux_total,
        'Fast': flux_fast,
        'Epithermal': flux_epithermal,
        'Thermal': flux_thermal
    }

    # Create coordinates
    r_max = derived['r_lithium_wall']
    height = derived['z_top'] - derived['z_bottom']
    z_bottom = derived['z_bottom']
    z_top = derived['z_top']
    z = np.linspace(z_bottom, z_top, nz)

    # Get z-index for midplane
    mid_z = nz // 2

    # Define boundary radii
    r_core = inputs['r_core']
    r_outer_tank = derived['r_outer_tank']
    r_rpv = derived['r_rpv_2']  # Outer RPV radius
    r_lithium = derived['r_lithium']
    r_lithium_wall = derived['r_lithium_wall']

    # Moderator and wall divider radii (if enabled)
    r_moderator = None
    r_wall_divider = None
    if inputs['enable_moderator_region']:
        r_moderator = derived['r_moderator']
        r_wall_divider = derived['r_wall_divider']

    # =================================================================
    # PLOT 1: SPATIAL HEATMAPS (2x2 grid, axially averaged only)
    # =================================================================
    fig1 = plt.figure(figsize=(16, 16))
    gs1 = plt.GridSpec(2, 2, figure=fig1, hspace=0.25, wspace=0.25)

    flux_order = ['Total', 'Fast', 'Epithermal', 'Thermal']

    for idx, flux_name in enumerate(flux_order):
        row = idx // 2
        col = idx % 2
        ax = fig1.add_subplot(gs1[row, col])

        flux_3d = flux_data[flux_name]

        # Average over Z axis
        flux_avg = np.mean(flux_3d, axis=2)  # Shape: (nx, ny)

        x_edges = np.linspace(-r_max, r_max, nx + 1)
        y_edges = np.linspace(-r_max, r_max, ny + 1)

        im = ax.pcolormesh(x_edges, y_edges, flux_avg.T,
                          cmap='viridis',
                          norm=plt.matplotlib.colors.LogNorm(vmin=np.max(flux_avg)*1e-3,
                                                             vmax=np.max(flux_avg)),
                          shading='flat')

        # Add boundary circles
        circle_core = plt.Circle((0, 0), r_core, fill=False,
                                color='red', linestyle='--', linewidth=2, label='Core', alpha=0.8)
        circle_tank = plt.Circle((0, 0), r_outer_tank, fill=False,
                                color='cyan', linestyle='--', linewidth=2, label='Tank', alpha=0.8)
        circle_rpv = plt.Circle((0, 0), r_rpv, fill=False,
                                color='orange', linestyle=':', linewidth=2, label='RPV', alpha=0.8)
        circle_lithium = plt.Circle((0, 0), r_lithium, fill=False,
                                   color='magenta', linestyle='-.', linewidth=1.5, label='Lithium', alpha=0.7)
        circle_wall = plt.Circle((0, 0), r_lithium_wall, fill=False,
                                color='white', linestyle='-', linewidth=2, label='Wall', alpha=0.9)

        ax.add_patch(circle_core)
        ax.add_patch(circle_tank)
        ax.add_patch(circle_rpv)
        ax.add_patch(circle_lithium)
        ax.add_patch(circle_wall)

        # Add moderator and wall divider circles if enabled
        if inputs['enable_moderator_region'] and r_moderator is not None:
            circle_moderator = plt.Circle((0, 0), r_moderator, fill=False,
                                         color='darkblue', linestyle='--', linewidth=1.5, label='Moderator', alpha=0.7)
            ax.add_patch(circle_moderator)
        if inputs['enable_moderator_region'] and r_wall_divider is not None:
            circle_wall_divider = plt.Circle((0, 0), r_wall_divider, fill=False,
                                            color='lime', linestyle=':', linewidth=1.5, label='Wall Divider', alpha=0.7)
            ax.add_patch(circle_wall_divider)

        ax.set_xlabel('X [cm]', fontsize=12)
        ax.set_ylabel('Y [cm]', fontsize=12)
        ax.set_title(f'{flux_name} Flux - Axially Averaged (XY Plane)',
                     fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Flux [n/cm²/s]', fontsize=11)

        if idx == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    plt.savefig(plot_dir / 'flux_spatial_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =================================================================
    # PLOT 2: RADIAL PROFILES (2 subplots)
    # Left: Axially averaged radial profile
    # Right: Midplane radial profile
    # All 4 flux types on same plot
    # =================================================================
    fig2, (ax_avg, ax_mid) = plt.subplots(1, 2, figsize=(16, 6))

    colors_flux = {
        'Total': 'black',
        'Fast': 'red',
        'Epithermal': 'orange',
        'Thermal': 'blue'
    }

    center_x = nx // 2
    center_y = ny // 2
    x_centers = np.linspace(-r_max, r_max, nx)
    r_centers = np.abs(x_centers)

    # --- Left: Axially averaged radial profile ---
    for flux_name, flux_3d in flux_data.items():
        # Average over z
        flux_avg_z = np.mean(flux_3d, axis=2)  # (nx, ny)

        # Get radial profile through center
        flux_radial_avg = flux_avg_z[:, center_y]

        # Average left and right
        flux_r_avg = (flux_radial_avg[:center_x][::-1] + flux_radial_avg[center_x:]) / 2
        r_plot = r_centers[center_x:]

        ax_avg.semilogy(r_plot, flux_r_avg, label=flux_name,
                       color=colors_flux[flux_name], linewidth=2.5, alpha=0.8)

    # Add boundary lines
    ax_avg.axvline(r_core, color='red', linestyle='--', linewidth=1.5, alpha=0.4, label='Core')
    ax_avg.axvline(r_outer_tank, color='cyan', linestyle='--', linewidth=1.5, alpha=0.4, label='Tank')
    ax_avg.axvline(r_rpv, color='orange', linestyle=':', linewidth=1.5, alpha=0.4, label='RPV')
    # Add moderator and wall divider lines if enabled
    if inputs['enable_moderator_region'] and r_moderator is not None:
        ax_avg.axvline(r_moderator, color='darkblue', linestyle='--', linewidth=1.5, alpha=0.4, label='Moderator')
    if inputs['enable_moderator_region'] and r_wall_divider is not None:
        ax_avg.axvline(r_wall_divider, color='lime', linestyle=':', linewidth=1.5, alpha=0.4, label='Wall Divider')
    ax_avg.axvline(r_lithium, color='magenta', linestyle='-.', linewidth=1.5, alpha=0.4, label='Lithium')

    ax_avg.set_xlabel('Radius [cm]', fontsize=12)
    ax_avg.set_ylabel('Flux [n/cm²/s]', fontsize=12)
    ax_avg.set_title('Radial Flux Profile - Axially Averaged', fontsize=14, fontweight='bold')
    ax_avg.legend(loc='best', fontsize=10, ncol=2)
    ax_avg.grid(True, alpha=0.3)
    ax_avg.set_xlim(0, r_max)

    # --- Right: Midplane radial profile ---
    for flux_name, flux_3d in flux_data.items():
        # Get midplane slice
        flux_mid = flux_3d[:, :, mid_z]

        # Get radial profile through center
        flux_radial_mid = flux_mid[:, center_y]

        # Average left and right
        flux_r_mid = (flux_radial_mid[:center_x][::-1] + flux_radial_mid[center_x:]) / 2
        r_plot = r_centers[center_x:]

        ax_mid.semilogy(r_plot, flux_r_mid, label=flux_name,
                       color=colors_flux[flux_name], linewidth=2.5, alpha=0.8)

    # Add boundary lines
    ax_mid.axvline(r_core, color='red', linestyle='--', linewidth=1.5, alpha=0.4, label='Core')
    ax_mid.axvline(r_outer_tank, color='cyan', linestyle='--', linewidth=1.5, alpha=0.4, label='Tank')
    ax_mid.axvline(r_rpv, color='orange', linestyle=':', linewidth=1.5, alpha=0.4, label='RPV')
    # Add moderator and wall divider lines if enabled
    if inputs['enable_moderator_region'] and r_moderator is not None:
        ax_mid.axvline(r_moderator, color='darkblue', linestyle='--', linewidth=1.5, alpha=0.4, label='Moderator')
    if inputs['enable_moderator_region'] and r_wall_divider is not None:
        ax_mid.axvline(r_wall_divider, color='lime', linestyle=':', linewidth=1.5, alpha=0.4, label='Wall Divider')
    ax_mid.axvline(r_lithium, color='magenta', linestyle='-.', linewidth=1.5, alpha=0.4, label='Lithium')

    ax_mid.set_xlabel('Radius [cm]', fontsize=12)
    ax_mid.set_ylabel('Flux [n/cm²/s]', fontsize=12)
    ax_mid.set_title('Radial Flux Profile - Z Midplane', fontsize=14, fontweight='bold')
    ax_mid.legend(loc='best', fontsize=10, ncol=2)
    ax_mid.grid(True, alpha=0.3)
    ax_mid.set_xlim(0, r_max)

    plt.tight_layout()
    plt.savefig(plot_dir / 'flux_radial_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()

    # =================================================================
    # PLOT 3: AXIAL PROFILES AT DIFFERENT RADIAL LOCATIONS
    # 5 subplots: center, core edge, tank, RPV, lithium
    # All 4 flux types on same plot
    # =================================================================
    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
    axes3 = axes3.flatten()

    # Calculate mesh cell centers
    x_centers = np.linspace(-r_max, r_max, nx)
    dx_cell = 2 * r_max / nx
    center_x = nx // 2
    center_y = ny // 2

    # Define radial locations with exact coordinates
    radial_locations_nominal = {
        'Center': 0,
        'Core Edge': r_core,
        'Outer Tank': r_outer_tank,
        'RPV': r_rpv,
        'Lithium': r_lithium
    }

    # Calculate actual radial indices and exact r values
    radial_locations = {}
    for name, r_nom in radial_locations_nominal.items():
        r_idx = int(r_nom / r_max * nx // 2)
        # Calculate actual r from mesh center
        if r_idx == 0:
            r_exact = 0.0
        else:
            r_exact = r_idx * dx_cell
        radial_locations[name] = (r_idx, r_exact)

    for ax_idx, (location_name, (r_idx, r_exact)) in enumerate(radial_locations.items()):
        ax = axes3[ax_idx]

        for flux_name, flux_3d in flux_data.items():
            # Get circular average at this radius
            if r_idx == 0:
                flux_axial = flux_3d[center_x, center_y, :]
            else:
                # Average around circle at this radius
                theta = np.linspace(0, 2*np.pi, 360)
                x_indices = np.round(center_x + r_idx * np.cos(theta)).astype(int)
                y_indices = np.round(center_y + r_idx * np.sin(theta)).astype(int)

                valid = (x_indices >= 0) & (x_indices < nx) & \
                       (y_indices >= 0) & (y_indices < ny)
                x_indices = x_indices[valid]
                y_indices = y_indices[valid]

                fluxes = np.zeros((nz, len(x_indices)))
                for i, (xi, yi) in enumerate(zip(x_indices, y_indices)):
                    fluxes[:, i] = flux_3d[xi, yi, :]

                flux_axial = np.mean(fluxes, axis=1)

            ax.plot(z, flux_axial, label=flux_name,
                   color=colors_flux[flux_name], linewidth=2.5, alpha=0.8)

        ax.set_xlabel('Z [cm]', fontsize=11)
        ax.set_ylabel('Flux [n/cm²/s]', fontsize=11)
        ax.set_title(f'Axial Profile - {location_name} (r = {r_exact:.1f} cm)',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(z_bottom, z_top)

    # Remove extra subplot
    fig3.delaxes(axes3[5])

    plt.tight_layout()
    plt.savefig(plot_dir / 'flux_axial_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to generate all plots."""
    print("\n" + "="*70)
    print("PWR FUSION BREEDER REACTOR - VISUALIZATION SCRIPT")
    print("="*70)

    # Define paths
    statepoint_file = 'simulation_raw/statepoint.250.h5'
    plot_dir = Path('visualization_figures')
    plot_dir.mkdir(exist_ok=True)

    # Load statepoint
    print(f"\nLoading statepoint file: {statepoint_file}")
    try:
        sp = openmc.StatePoint(statepoint_file)
        print(f"✓ Successfully loaded statepoint")
    except Exception as e:
        print(f"✗ Error loading statepoint: {e}")
        sys.exit(1)

    # Generate plots and print results
    print_tritium_production(sp)
    plot_keff_entropy(sp, plot_dir)  # Saves to plots/entropy.png
    plot_energy_spectrum_comparison(sp, plot_dir)  # Saves to plots/flux_energy_spectrum_comparison.png
    plot_surface_current_spectrum_comparison(sp, plot_dir)  # NEW: Surface current plots
    plot_mesh_flux(sp, plot_dir)  # Saves to plots/flux_*.png

if __name__ == '__main__':
    main()
