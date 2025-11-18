"""
Plot tallies from PWR Fusion Breeder Reactor simulation
Creates plots for tritium breeder assembly surface currents and fluxes
"""
import openmc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from inputs import inputs, get_derived_dimensions
from tallies import calc_norm_factor

def plot_tritium_breeder_tallies(statepoint_path='simulation_raw/statepoint.250.h5', output_dir='tally_figures'):
    """Plot surface currents and fluxes for tritium breeder assembly.

    Parameters
    ----------
    statepoint_path : str
        Path to the statepoint file
    output_dir : str
        Directory to save plots
    """
    # Load statepoint
    print(f"Loading statepoint from: {statepoint_path}")
    sp = openmc.StatePoint(statepoint_path)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Saving plots to: {output_path}")

    # Calculate normalization factor
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)

    # Get dimensions
    derived = get_derived_dimensions()

    # Calculate surface areas and volumes for tritium breeder
    calandria_or = inputs['candu_calandria_or']  # 6.7526 cm
    pt_ir = inputs['candu_pressure_tube_ir']  # 5.1689 cm

    # Height
    height = derived['z_fuel_top'] - derived['z_fuel_bottom']

    # Surface areas (cylindrical surfaces: 2*pi*r*h)
    calandria_outer_area = 2 * np.pi * calandria_or * height  # cm²
    pt_inner_area = 2 * np.pi * pt_ir * height  # cm²

    # Volume of breeder region (cylinder: pi*r²*h)
    breeder_volume = np.pi * pt_ir**2 * height  # cm³

    print(f"\nGeometry:")
    print(f"  Calandria outer area: {calandria_outer_area:.2e} cm²")
    print(f"  PT inner area: {pt_inner_area:.2e} cm²")
    print(f"  Breeder volume: {breeder_volume:.2e} cm³")

    # Energy bins for plotting
    energy_bins_eV = {
        'Thermal': (0.0, 0.625),
        'Epithermal': (0.625, 1e5),
        'Fast': (1e5, 2e7)
    }
    energy_bins_MeV = {
        name: (e_min/1e6, e_max/1e6) for name, (e_min, e_max) in energy_bins_eV.items()
    }

    # ============================================================
    # Create figure with 3 rows and 3 columns
    # Row 1: Calandria outer surface currents
    # Row 2: Pressure tube inner surface currents
    # Row 3: Breeder region flux
    # Column 1: Log-log
    # Column 2: Linear-log (semilogx)
    # Column 3: Three-group step function
    # ============================================================
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    fig.suptitle('Tritium Breeder Assembly: Surface Currents and Flux', fontsize=18, fontweight='bold')

    colors_3g = {'Thermal': 'green', 'Epithermal': 'orange', 'Fast': 'red'}
    group_order = ['Thermal', 'Epithermal', 'Fast']

    # ============================================================
    # ROW 1: Calandria Outer Surface Currents (from moderator)
    # ============================================================
    print("\nProcessing Calandria Outer Surface Currents...")

    # Get LOG_1001 current spectrum
    try:
        tally_log1001 = sp.get_tally(name='tritium_calandria_outer_current_log1001')

        # Get energy bins
        energy_filter = tally_log1001.filters[2]  # Third filter is energy
        energy_bins = energy_filter.bins

        energy_centers_eV = np.sqrt(energy_bins[:, 0] * energy_bins[:, 1])
        energy_centers_MeV = energy_centers_eV / 1e6

        # Normalize: multiply by norm_factor, divide by area, flip sign for inward
        current_values = -tally_log1001.mean.flatten() * norm_factor
        current_density = current_values / calandria_outer_area
        current_std = tally_log1001.std_dev.flatten() * norm_factor / calandria_outer_area

        has_log1001_calandria = True
    except Exception as e:
        print(f"  Could not get LOG_1001 spectrum: {e}")
        has_log1001_calandria = False

    # Get three-group currents and normalize properly
    calandria_groups = ['thermal', 'epithermal', 'fast']
    three_group_currents = {}
    for group in calandria_groups:
        tally_name = f'tritium_calandria_outer_current_{group}'
        try:
            tally = sp.get_tally(name=tally_name)
            abs_current = tally.mean[0, 0, 0] * norm_factor
            current_density_val = -abs_current / calandria_outer_area
            three_group_currents[group.capitalize()] = current_density_val
            print(f"  {group}: {current_density_val:.4e} n/cm²/s")
        except Exception as e:
            print(f"  Error getting {tally_name}: {e}")
            three_group_currents[group.capitalize()] = 0.0

    # Calculate fractions
    total_current = sum(three_group_currents.values())
    fractions = {name: curr/total_current if total_current > 0 else 0
                for name, curr in three_group_currents.items()}

    # Column 1: Log-log spectrum
    ax = axes[0, 0]
    if has_log1001_calandria:
        ax.loglog(energy_centers_MeV, current_density, 'b-', linewidth=2, label='LOG_1001 Spectrum')
        ax.set_xlabel('Energy [MeV]', fontsize=11)
        ax.set_ylabel('Current Density [n/cm²/s]', fontsize=11)
        ax.set_title('Calandria Outer - Log-Log', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1e-12, 20)
    else:
        ax.text(0.5, 0.5, 'LOG_1001\nnot available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)

    # Column 2: Linear-log spectrum
    ax = axes[0, 1]
    if has_log1001_calandria:
        ax.semilogx(energy_centers_MeV, current_density, 'b-', linewidth=2, label='LOG_1001 Spectrum')
        ax.set_xlabel('Energy [MeV]', fontsize=11)
        ax.set_ylabel('Current Density [n/cm²/s]', fontsize=11)
        ax.set_title('Calandria Outer - Linear-Log', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1e-12, 20)
    else:
        ax.text(0.5, 0.5, 'LOG_1001\nnot available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)

    # Column 3: Three-group step function
    ax = axes[0, 2]
    current_levels = []

    for group_name in group_order:
        e_min, e_max = energy_bins_MeV[group_name]
        if group_name in three_group_currents and three_group_currents[group_name] > 0:
            current_val = three_group_currents[group_name]
            frac = fractions[group_name]
            current_levels.append((group_name, current_val, e_min, e_max))

            ax.hlines(current_val, e_min, e_max,
                     colors=colors_3g[group_name], linewidth=4,
                     label=f'{group_name}: {frac:.3f}', alpha=0.8)

    for i in range(len(current_levels) - 1):
        curr_group, curr_current, _, curr_e_max = current_levels[i]
        next_group, next_current, next_e_min, _ = current_levels[i + 1]
        ax.vlines(curr_e_max, curr_current, next_current,
                 colors=colors_3g[curr_group], linewidth=2.5,
                 linestyles='-', alpha=0.8)

    ax.set_xlabel('Energy [MeV]', fontsize=11)
    ax.set_ylabel('Current Density [n/cm²/s]', fontsize=11)
    ax.set_title('Calandria Outer - Three-Group', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1e-12, 20)

    # ============================================================
    # ROW 2: Pressure Tube Inner Surface Currents (from PT)
    # ============================================================
    print("\nProcessing Pressure Tube Inner Surface Currents...")

    # Get LOG_1001 current spectrum
    try:
        tally_log1001_pt = sp.get_tally(name='tritium_pt_inner_current_log1001')

        # Get energy bins
        energy_filter_pt = tally_log1001_pt.filters[2]  # Third filter is energy
        energy_bins_pt = energy_filter_pt.bins

        energy_centers_eV_pt = np.sqrt(energy_bins_pt[:, 0] * energy_bins_pt[:, 1])
        energy_centers_MeV_pt = energy_centers_eV_pt / 1e6

        # Normalize: multiply by norm_factor, divide by area, flip sign for inward
        current_values_pt = -tally_log1001_pt.mean.flatten() * norm_factor
        current_density_pt = current_values_pt / pt_inner_area
        current_std_pt = tally_log1001_pt.std_dev.flatten() * norm_factor / pt_inner_area

        has_log1001_pt = True
    except Exception as e:
        print(f"  Could not get LOG_1001 spectrum: {e}")
        has_log1001_pt = False

    # Get three-group currents
    pt_groups = ['thermal', 'epithermal', 'fast']
    three_group_currents_pt = {}
    for group in pt_groups:
        tally_name = f'tritium_pt_inner_current_{group}'
        try:
            tally = sp.get_tally(name=tally_name)
            abs_current = tally.mean[0, 0, 0] * norm_factor
            current_density_val = -abs_current / pt_inner_area
            three_group_currents_pt[group.capitalize()] = current_density_val
            print(f"  {group}: {current_density_val:.4e} n/cm²/s")
        except Exception as e:
            print(f"  Error getting {tally_name}: {e}")
            three_group_currents_pt[group.capitalize()] = 0.0

    # Calculate fractions
    total_current_pt = sum(three_group_currents_pt.values())
    fractions_pt = {name: curr/total_current_pt if total_current_pt > 0 else 0
                for name, curr in three_group_currents_pt.items()}

    # Column 1: Log-log spectrum
    ax = axes[1, 0]
    if has_log1001_pt:
        ax.loglog(energy_centers_MeV_pt, current_density_pt, 'b-', linewidth=2, label='LOG_1001 Spectrum')
        ax.set_xlabel('Energy [MeV]', fontsize=11)
        ax.set_ylabel('Current Density [n/cm²/s]', fontsize=11)
        ax.set_title('Pressure Tube Inner - Log-Log', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1e-12, 20)
    else:
        ax.text(0.5, 0.5, 'LOG_1001\nnot available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)

    # Column 2: Linear-log spectrum
    ax = axes[1, 1]
    if has_log1001_pt:
        ax.semilogx(energy_centers_MeV_pt, current_density_pt, 'b-', linewidth=2, label='LOG_1001 Spectrum')
        ax.set_xlabel('Energy [MeV]', fontsize=11)
        ax.set_ylabel('Current Density [n/cm²/s]', fontsize=11)
        ax.set_title('Pressure Tube Inner - Linear-Log', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1e-12, 20)
    else:
        ax.text(0.5, 0.5, 'LOG_1001\nnot available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)

    # Column 3: Three-group step function
    ax = axes[1, 2]
    current_levels_pt = []

    for group_name in group_order:
        e_min, e_max = energy_bins_MeV[group_name]
        if group_name in three_group_currents_pt and three_group_currents_pt[group_name] > 0:
            current_val = three_group_currents_pt[group_name]
            frac = fractions_pt[group_name]
            current_levels_pt.append((group_name, current_val, e_min, e_max))

            ax.hlines(current_val, e_min, e_max,
                     colors=colors_3g[group_name], linewidth=4,
                     label=f'{group_name}: {frac:.3f}', alpha=0.8)

    for i in range(len(current_levels_pt) - 1):
        curr_group, curr_current, _, curr_e_max = current_levels_pt[i]
        next_group, next_current, next_e_min, _ = current_levels_pt[i + 1]
        ax.vlines(curr_e_max, curr_current, next_current,
                 colors=colors_3g[curr_group], linewidth=2.5,
                 linestyles='-', alpha=0.8)

    ax.set_xlabel('Energy [MeV]', fontsize=11)
    ax.set_ylabel('Current Density [n/cm²/s]', fontsize=11)
    ax.set_title('Pressure Tube Inner - Three-Group', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1e-12, 20)

    # ============================================================
    # ROW 3: Flux in Breeder Region
    # ============================================================
    print("\nProcessing Flux Tallies...")

    # Get three-group fluxes for column 3
    flux_groups = ['thermal', 'epithermal', 'fast']
    three_group_fluxes = {}
    for group in flux_groups:
        tally_name = f'tritium_breeder_flux_{group}'
        try:
            tally = sp.get_tally(name=tally_name)
            flux_density = tally.mean[0, 0, 0] * norm_factor / breeder_volume
            three_group_fluxes[group.capitalize()] = flux_density
            print(f"  {group}: {flux_density:.4e} n/cm²/s")
        except Exception as e:
            print(f"  Error getting {tally_name}: {e}")
            three_group_fluxes[group.capitalize()] = 0.0

    # Calculate fractions
    total_flux = sum(three_group_fluxes.values())
    fractions_flux = {name: flux/total_flux if total_flux > 0 else 0
                for name, flux in three_group_fluxes.items()}

    # Get LOG_1001 flux spectrum
    try:
        tally = sp.get_tally(name='tritium_breeder_flux_log1001')

        # Get energy bins
        energy_filter = tally.filters[1]
        energy_bins = energy_filter.bins

        energy_centers_eV = np.sqrt(energy_bins[:, 0] * energy_bins[:, 1])
        energy_centers_MeV = energy_centers_eV / 1e6

        # Sum over all cells
        flux_per_cell = tally.mean[:, :, 0]
        flux_std_per_cell = tally.std_dev[:, :, 0]

        flux_total = np.sum(flux_per_cell, axis=0)
        flux_std_total = np.sqrt(np.sum(flux_std_per_cell**2, axis=0))

        # Normalize
        flux = flux_total * norm_factor / breeder_volume
        flux_std = flux_std_total * norm_factor / breeder_volume

        print(f"  LOG_1001 spectrum: {len(flux)} bins")

        # Column 1: Log-log
        ax = axes[2, 0]
        ax.loglog(energy_centers_MeV, flux, 'b-', linewidth=2, label='LOG_1001 Spectrum')
        ax.set_xlabel('Energy [MeV]', fontsize=11)
        ax.set_ylabel('Flux [n/cm²/s]', fontsize=11)
        ax.set_title('Breeder Region - Log-Log', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1e-12, 20)

        # Column 2: Linear-log (semilogx)
        ax = axes[2, 1]
        ax.semilogx(energy_centers_MeV, flux, 'b-', linewidth=2, label='LOG_1001 Spectrum')
        ax.set_xlabel('Energy [MeV]', fontsize=11)
        ax.set_ylabel('Flux [n/cm²/s]', fontsize=11)
        ax.set_title('Breeder Region - Linear-Log', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1e-12, 20)

    except Exception as e:
        print(f"  Could not create flux spectrum plot: {e}")
        for col_idx in [0, 1]:
            axes[2, col_idx].text(0.5, 0.5, 'LOG_1001 flux\nnot available',
                                 ha='center', va='center', transform=axes[2, col_idx].transAxes, fontsize=12)

    # Column 3: Three-group step function
    ax = axes[2, 2]
    flux_levels = []

    for group_name in group_order:
        e_min, e_max = energy_bins_MeV[group_name]
        if group_name in three_group_fluxes and three_group_fluxes[group_name] > 0:
            flux_val = three_group_fluxes[group_name]
            frac = fractions_flux[group_name]
            flux_levels.append((group_name, flux_val, e_min, e_max))

            ax.hlines(flux_val, e_min, e_max,
                     colors=colors_3g[group_name], linewidth=4,
                     label=f'{group_name}: {frac:.3f}', alpha=0.8)

    for i in range(len(flux_levels) - 1):
        curr_group, curr_flux, _, curr_e_max = flux_levels[i]
        next_group, next_flux, next_e_min, _ = flux_levels[i + 1]
        ax.vlines(curr_e_max, curr_flux, next_flux,
                 colors=colors_3g[curr_group], linewidth=2.5,
                 linestyles='-', alpha=0.8)

    ax.set_xlabel('Energy [MeV]', fontsize=11)
    ax.set_ylabel('Flux [n/cm²/s]', fontsize=11)
    ax.set_title('Breeder Region - Three-Group', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1e-12, 20)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    output_file = output_path / 'tritium_breeder_tallies.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()

    print("\n" + "="*60)
    print("Plots created successfully!")
    print("="*60)


def plot_core_radial_flux(sp, output_dir='tally_figures'):
    """Plot radial flux profile from core mesh (energy discretized)."""
    output_path = Path(output_dir)

    print("\nCreating core radial flux plot...")

    # Calculate normalization
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)
    derived = get_derived_dimensions()

    # Get mesh tallies
    try:
        total_tally = sp.get_tally(name='core_mesh_total_flux')
        thermal_tally = sp.get_tally(name='core_mesh_thermal_flux')
        epithermal_tally = sp.get_tally(name='core_mesh_epithermal_flux')
        fast_tally = sp.get_tally(name='core_mesh_fast_flux')
    except Exception as e:
        print(f"  Could not find core mesh tallies: {e}")
        return

    # Get mesh info
    mesh_filter = total_tally.find_filter(openmc.MeshFilter)
    mesh = mesh_filter.mesh
    nx, ny, nz = mesh.dimension

    # Calculate mesh volume
    r_max = derived['r_lithium_wall']
    dx = 2 * r_max / nx
    dy = 2 * r_max / ny
    dz = (derived['z_top'] - derived['z_bottom']) / nz
    mesh_volume = dx * dy * dz

    # Reshape and normalize flux data
    flux_total = total_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume
    flux_thermal = thermal_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume
    flux_epithermal = epithermal_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume
    flux_fast = fast_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume

    # Average over Z axis
    flux_total_avg = np.mean(flux_total, axis=2)
    flux_thermal_avg = np.mean(flux_thermal, axis=2)
    flux_epithermal_avg = np.mean(flux_epithermal, axis=2)
    flux_fast_avg = np.mean(flux_fast, axis=2)

    # Get radial profile through center
    center_x = nx // 2
    center_y = ny // 2
    x_centers = np.linspace(-r_max, r_max, nx)
    r_centers = np.abs(x_centers)

    flux_radial_total = flux_total_avg[:, center_y]
    flux_radial_thermal = flux_thermal_avg[:, center_y]
    flux_radial_epithermal = flux_epithermal_avg[:, center_y]
    flux_radial_fast = flux_fast_avg[:, center_y]

    # Average left and right sides
    flux_r_total = (flux_radial_total[:center_x][::-1] + flux_radial_total[center_x:]) / 2
    flux_r_thermal = (flux_radial_thermal[:center_x][::-1] + flux_radial_thermal[center_x:]) / 2
    flux_r_epithermal = (flux_radial_epithermal[:center_x][::-1] + flux_radial_epithermal[center_x:]) / 2
    flux_r_fast = (flux_radial_fast[:center_x][::-1] + flux_radial_fast[center_x:]) / 2
    r_plot = r_centers[center_x:]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    colors_flux = {
        'Total': 'black',
        'Fast': 'red',
        'Epithermal': 'orange',
        'Thermal': 'blue'
    }

    ax.semilogy(r_plot, flux_r_total, label='Total', color=colors_flux['Total'], linewidth=2.5, alpha=0.8)
    ax.semilogy(r_plot, flux_r_fast, label='Fast', color=colors_flux['Fast'], linewidth=2.5, alpha=0.8)
    ax.semilogy(r_plot, flux_r_epithermal, label='Epithermal', color=colors_flux['Epithermal'], linewidth=2.5, alpha=0.8)
    ax.semilogy(r_plot, flux_r_thermal, label='Thermal', color=colors_flux['Thermal'], linewidth=2.5, alpha=0.8)

    # Add boundary lines
    r_core = inputs['r_core']
    r_rpv_2 = derived['r_rpv_2']
    r_lithium = derived['r_lithium']

    ax.axvline(r_core, color='red', linestyle='--', linewidth=1.5, alpha=0.4, label='Core')
    ax.axvline(r_rpv_2, color='orange', linestyle=':', linewidth=1.5, alpha=0.4, label='RPV')
    if inputs['enable_moderator_region']:
        r_moderator = derived['r_moderator']
        r_wall_divider = derived['r_wall_divider']
        ax.axvline(r_moderator, color='darkblue', linestyle='--', linewidth=1.5, alpha=0.4, label='Moderator')
        ax.axvline(r_wall_divider, color='lime', linestyle=':', linewidth=1.5, alpha=0.4, label='Wall Divider')
    ax.axvline(r_lithium, color='magenta', linestyle='-.', linewidth=1.5, alpha=0.4, label='Lithium')

    ax.set_xlabel('Radius [cm]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Flux [n/cm²/s]', fontsize=14, fontweight='bold')
    ax.set_title('Core Radial Flux Profile - Axially Averaged', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_max)

    plt.tight_layout()
    output_file = output_path / 'core_radial_flux.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_tritium_assembly_heatmap(sp, output_dir='tally_figures'):
    """Plot 3x3 assembly mesh heatmap centered on tritium breeder (2x2 subplot)."""
    output_path = Path(output_dir)

    print("\nCreating tritium assembly heatmaps...")

    # Calculate normalization
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)
    derived = get_derived_dimensions()

    # Get mesh tallies
    try:
        total_tally = sp.get_tally(name='tritium_assembly_mesh_total')
        thermal_tally = sp.get_tally(name='tritium_assembly_mesh_thermal')
        epithermal_tally = sp.get_tally(name='tritium_assembly_mesh_epithermal')
        fast_tally = sp.get_tally(name='tritium_assembly_mesh_fast')
    except Exception as e:
        print(f"  Could not find tritium assembly mesh tallies: {e}")
        return

    # Get mesh info
    mesh_filter = total_tally.find_filter(openmc.MeshFilter)
    mesh = mesh_filter.mesh
    nx, ny, nz = mesh.dimension  # Should be 50, 50, 1

    # Find T_1 position in lattice (same calculation as in tallies.py)
    assembly_width = derived['assembly_width']
    if inputs['assembly_type'] == 'candu':
        core_lattice = inputs['candu_lattice']
    else:
        core_lattice = inputs['ap1000_lattice']

    n_rows = len(core_lattice)
    n_cols = len(core_lattice[0])

    # Find T_1 position
    t1_row, t1_col = None, None
    for row_idx, row in enumerate(core_lattice):
        for col_idx, symbol in enumerate(row):
            if symbol == 'T_1' or symbol == 'T':
                t1_row, t1_col = row_idx, col_idx
                break
        if t1_row is not None:
            break

    # Calculate T_1 center position (must match tallies.py)
    if t1_row is not None and t1_col is not None:
        x_center = (t1_col - n_cols/2 + 0.5) * assembly_width
        y_center = (t1_row - n_rows/2 + 0.5) * assembly_width
    else:
        x_center, y_center = 0.0, 0.0

    mesh_width = 3 * assembly_width

    # Calculate mesh volume
    dx = mesh_width / nx
    dy = mesh_width / ny
    dz = (derived['z_fuel_top'] - derived['z_fuel_bottom']) / nz
    mesh_volume = dx * dy * dz

    # Reshape and normalize all flux types
    flux_total = total_tally.mean.reshape((nz, ny, nx))[0, :, :] * norm_factor / mesh_volume
    flux_thermal = thermal_tally.mean.reshape((nz, ny, nx))[0, :, :] * norm_factor / mesh_volume
    flux_epithermal = epithermal_tally.mean.reshape((nz, ny, nx))[0, :, :] * norm_factor / mesh_volume
    flux_fast = fast_tally.mean.reshape((nz, ny, nx))[0, :, :] * norm_factor / mesh_volume

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle('Tritium Breeder Assembly: 3×3 Flux Distribution', fontsize=18, fontweight='bold')

    # Create coordinate arrays centered on T_1
    x_edges = np.linspace(x_center - mesh_width/2, x_center + mesh_width/2, nx + 1)
    y_edges = np.linspace(y_center - mesh_width/2, y_center + mesh_width/2, ny + 1)

    # Plot data
    flux_data = [
        ('Total', flux_total, axes[0, 0]),
        ('Fast', flux_fast, axes[0, 1]),
        ('Epithermal', flux_epithermal, axes[1, 0]),
        ('Thermal', flux_thermal, axes[1, 1])
    ]

    calandria_or = inputs['candu_calandria_or']

    for flux_name, flux_2d, ax in flux_data:
        # Plot heatmap with log scale
        vmin = np.max(flux_2d) * 1e-3  # Set lower limit to avoid zero issues
        vmax = np.max(flux_2d)

        im = ax.pcolormesh(x_edges, y_edges, flux_2d,
                          cmap='viridis',
                          norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                          shading='flat')

        # Add assembly boundaries (3x3 grid centered on T_1)
        for i in range(-1, 2):
            for j in range(-1, 2):
                x_asm = x_center + i * assembly_width
                y_asm = y_center + j * assembly_width
                square = plt.Rectangle((x_asm - assembly_width/2, y_asm - assembly_width/2),
                                      assembly_width, assembly_width,
                                      fill=False, edgecolor='white', linewidth=2, linestyle='--', alpha=0.8)
                ax.add_patch(square)

        ax.set_xlabel('X [cm]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y [cm]', fontsize=12, fontweight='bold')
        ax.set_title(f'{flux_name} Flux', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(x_center - mesh_width/2, x_center + mesh_width/2)
        ax.set_ylim(y_center - mesh_width/2, y_center + mesh_width/2)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Flux [n/cm²/s]', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_file = output_path / 'tritium_assembly_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    import sys

    # Default statepoint path
    statepoint_path = 'simulation_raw/statepoint.250.h5'

    # Check if user provided a different path
    if len(sys.argv) > 1:
        statepoint_path = sys.argv[1]

    # Check if file exists
    if not Path(statepoint_path).exists():
        print(f"Error: Statepoint file not found: {statepoint_path}")
        print("\nUsage: python plot_tallies.py [statepoint_path]")
        print("Example: python plot_tallies.py simulation_raw/statepoint.500.h5")
        sys.exit(1)

    # Load statepoint once for all plots
    print("\n" + "="*60)
    print("TRITIUM BREEDER TALLY VISUALIZATION")
    print("="*60)
    sp = openmc.StatePoint(statepoint_path)

    # Create all plots
    plot_tritium_breeder_tallies(statepoint_path)
    plot_core_radial_flux(sp)
    plot_tritium_assembly_heatmap(sp)

    print("\n" + "="*60)
    print("ALL PLOTS COMPLETED!")
    print("="*60)
