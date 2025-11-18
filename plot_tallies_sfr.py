"""
Plot tallies for SFR Tritium Breeder Assembly
Handles hexagonal geometry with 4 energy groups
"""
import openmc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from inputs import inputs, get_derived_dimensions
from tallies import calc_norm_factor


def plot_sfr_tritium_breeder_tallies(sp, output_dir='tally_figures'):
    """Plot surface currents and fluxes for SFR tritium breeder assembly.

    Parameters
    ----------
    sp : openmc.StatePoint
        Loaded statepoint file
    output_dir : str
        Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Calculate normalization factor
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)

    print("\n" + "="*70)
    print("PLOTTING SFR TRITIUM BREEDER TALLIES")
    print("="*70)

    # Get dimensions
    hex_edge = inputs['sfr_tritium_breeder_edge']
    breeder_hex_edge = hex_edge - 1.0  # Minus 1 cm cladding
    axial_height = inputs['sfr_axial_height'] * 2  # Full height

    # Calculate hexagonal surface area: perimeter × height
    # Hexagon perimeter = 6 × edge_length
    breeder_perimeter = 6 * breeder_hex_edge  # cm
    breeder_surface_area = breeder_perimeter * axial_height  # cm²

    # Calculate hexagonal breeder volume
    # Hexagon area = (3√3/2) × edge²
    breeder_area = (3 * np.sqrt(3) / 2) * breeder_hex_edge**2  # cm²
    breeder_volume = breeder_area * axial_height  # cm³

    # Subtract volume of 7 cooling tubes
    coolant_tube_or = 1.25  # cm
    cooling_tube_volume = 7 * np.pi * coolant_tube_or**2 * axial_height
    breeder_volume_net = breeder_volume - cooling_tube_volume

    print(f"\nGeometry:")
    print(f"  Hexagonal edge (breeder): {breeder_hex_edge:.3f} cm")
    print(f"  Breeder surface area: {breeder_surface_area:.2e} cm²")
    print(f"  Breeder volume (net): {breeder_volume_net:.2e} cm³")
    print(f"  Axial height: {axial_height:.1f} cm")

    # Energy bins for plotting (updated with new boundaries)
    energy_bins_eV = {
        'Thermal': (0.0, 0.625),
        'Epithermal': (0.625, 100e3),
        'Fast': (100e3, 3e6),
        'Very-Fast': (3e6, 20e6)
    }
    energy_bins_MeV = {
        name: (e_min/1e6, e_max/1e6) for name, (e_min, e_max) in energy_bins_eV.items()
    }

    # Create figure with 2 rows and 3 columns
    # Row 1: Surface currents (cladding → breeder)
    # Row 2: Flux in breeder material
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle('SFR Tritium Breeder: Surface Currents and Flux (4-Group)',
                 fontsize=18, fontweight='bold')

    colors_4g = {
        'Thermal': 'green',
        'Epithermal': 'orange',
        'Fast': 'red',
        'Very-Fast': 'darkred'
    }
    group_order = ['Thermal', 'Epithermal', 'Fast', 'Very-Fast']

    # ============================================================
    # ROW 1: Surface Currents (Cladding → Breeder)
    # ============================================================
    print("\nProcessing Surface Currents (Cladding → Breeder)...")

    # Get LOG_1001 current spectrum
    try:
        tally_log1001 = sp.get_tally(name='sfr_tritium_current_log1001')

        # Get energy bins
        energy_filter = tally_log1001.filters[2]  # Third filter is energy
        energy_bins = energy_filter.bins

        energy_centers_eV = np.sqrt(energy_bins[:, 0] * energy_bins[:, 1])
        energy_centers_MeV = energy_centers_eV / 1e6

        # Normalize: multiply by norm_factor, divide by surface area
        current_values = tally_log1001.mean.flatten() * norm_factor
        current_density = current_values / breeder_surface_area
        current_std = tally_log1001.std_dev.flatten() * norm_factor / breeder_surface_area

        has_log1001_current = True
    except Exception as e:
        print(f"  Warning: Could not get LOG_1001 current spectrum: {e}")
        has_log1001_current = False

    # Get four-group currents
    current_groups = ['thermal', 'epithermal', 'fast', 'veryfast']
    four_group_currents = {}
    for group in current_groups:
        tally_name = f'sfr_tritium_current_{group}'
        try:
            tally = sp.get_tally(name=tally_name)
            abs_current = tally.mean[0, 0, 0] * norm_factor
            current_density_val = abs_current / breeder_surface_area
            group_label = group.replace('veryfast', 'Very-Fast').capitalize()
            four_group_currents[group_label] = current_density_val
            print(f"  {group}: {current_density_val:.4e} n/cm²/s")
        except Exception as e:
            print(f"  Warning: Could not get {tally_name}: {e}")
            group_label = group.replace('veryfast', 'Very-Fast').capitalize()
            four_group_currents[group_label] = 0.0

    # Calculate fractions
    total_current = sum(four_group_currents.values())
    fractions = {name: curr/total_current if total_current > 0 else 0
                for name, curr in four_group_currents.items()}

    # Column 1: Log-log spectrum
    ax = axes[0, 0]
    if has_log1001_current:
        ax.loglog(energy_centers_MeV, current_density, 'b-', linewidth=2, label='LOG_1001 Spectrum')
        ax.set_xlabel('Energy [MeV]', fontsize=11)
        ax.set_ylabel('Current Density [n/cm²/s]', fontsize=11)
        ax.set_title('Cladding → Breeder - Log-Log', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1e-12, 20)
    else:
        ax.text(0.5, 0.5, 'LOG_1001\nnot available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)

    # Column 2: Linear-log spectrum
    ax = axes[0, 1]
    if has_log1001_current:
        ax.semilogx(energy_centers_MeV, current_density, 'b-', linewidth=2, label='LOG_1001 Spectrum')
        ax.set_xlabel('Energy [MeV]', fontsize=11)
        ax.set_ylabel('Current Density [n/cm²/s]', fontsize=11)
        ax.set_title('Cladding → Breeder - Linear-Log', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1e-12, 20)
    else:
        ax.text(0.5, 0.5, 'LOG_1001\nnot available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)

    # Column 3: Four-group step function
    ax = axes[0, 2]
    current_levels = []

    for group_name in group_order:
        e_min, e_max = energy_bins_MeV[group_name]
        if group_name in four_group_currents and four_group_currents[group_name] > 0:
            current_val = four_group_currents[group_name]
            frac = fractions[group_name]
            current_levels.append((group_name, current_val, e_min, e_max))

            ax.hlines(current_val, e_min, e_max,
                     colors=colors_4g[group_name], linewidth=4,
                     label=f'{group_name}: {frac:.3f}', alpha=0.8)

    for i in range(len(current_levels) - 1):
        curr_group, curr_current, _, curr_e_max = current_levels[i]
        next_group, next_current, next_e_min, _ = current_levels[i + 1]
        ax.vlines(curr_e_max, curr_current, next_current,
                 colors=colors_4g[curr_group], linewidth=2.5,
                 linestyles='-', alpha=0.8)

    ax.set_xlabel('Energy [MeV]', fontsize=11)
    ax.set_ylabel('Current Density [n/cm²/s]', fontsize=11)
    ax.set_title('Cladding → Breeder - Four-Group', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1e-12, 20)

    # ============================================================
    # ROW 2: Flux in Breeder Material
    # ============================================================
    print("\nProcessing Flux in Breeder Material...")

    # Get four-group fluxes
    flux_groups = ['thermal', 'epithermal', 'fast', 'veryfast']
    four_group_fluxes = {}
    for group in flux_groups:
        tally_name = f'sfr_tritium_flux_{group}'
        try:
            tally = sp.get_tally(name=tally_name)
            flux_density = tally.mean[0, 0, 0] * norm_factor / breeder_volume_net
            group_label = group.replace('veryfast', 'Very-Fast').capitalize()
            four_group_fluxes[group_label] = flux_density
            print(f"  {group}: {flux_density:.4e} n/cm²/s")
        except Exception as e:
            print(f"  Warning: Could not get {tally_name}: {e}")
            group_label = group.replace('veryfast', 'Very-Fast').capitalize()
            four_group_fluxes[group_label] = 0.0

    # Calculate fractions
    total_flux = sum(four_group_fluxes.values())
    fractions_flux = {name: flux/total_flux if total_flux > 0 else 0
                for name, flux in four_group_fluxes.items()}

    # Get LOG_1001 flux spectrum
    try:
        tally = sp.get_tally(name='sfr_tritium_flux_log1001')

        # Get energy bins
        energy_filter = tally.filters[1]  # Second filter is energy
        energy_bins = energy_filter.bins

        energy_centers_eV = np.sqrt(energy_bins[:, 0] * energy_bins[:, 1])
        energy_centers_MeV = energy_centers_eV / 1e6

        # Normalize
        flux_values = tally.mean.flatten() * norm_factor
        flux = flux_values / breeder_volume_net
        flux_std = tally.std_dev.flatten() * norm_factor / breeder_volume_net

        has_log1001_flux = True

        # Column 1: Log-log
        ax = axes[1, 0]
        ax.loglog(energy_centers_MeV, flux, 'b-', linewidth=2, label='LOG_1001 Spectrum')
        ax.set_xlabel('Energy [MeV]', fontsize=11)
        ax.set_ylabel('Flux [n/cm²/s]', fontsize=11)
        ax.set_title('Breeder Material - Log-Log', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1e-12, 20)

        # Column 2: Linear-log (semilogx)
        ax = axes[1, 1]
        ax.semilogx(energy_centers_MeV, flux, 'b-', linewidth=2, label='LOG_1001 Spectrum')
        ax.set_xlabel('Energy [MeV]', fontsize=11)
        ax.set_ylabel('Flux [n/cm²/s]', fontsize=11)
        ax.set_title('Breeder Material - Linear-Log', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim(1e-12, 20)

    except Exception as e:
        print(f"  Warning: Could not create flux spectrum plot: {e}")
        for col_idx in [0, 1]:
            axes[1, col_idx].text(0.5, 0.5, 'LOG_1001 flux\nnot available',
                                 ha='center', va='center', transform=axes[1, col_idx].transAxes,
                                 fontsize=12)

    # Column 3: Four-group step function
    ax = axes[1, 2]
    flux_levels = []

    for group_name in group_order:
        e_min, e_max = energy_bins_MeV[group_name]
        if group_name in four_group_fluxes and four_group_fluxes[group_name] > 0:
            flux_val = four_group_fluxes[group_name]
            frac = fractions_flux[group_name]
            flux_levels.append((group_name, flux_val, e_min, e_max))

            ax.hlines(flux_val, e_min, e_max,
                     colors=colors_4g[group_name], linewidth=4,
                     label=f'{group_name}: {frac:.3f}', alpha=0.8)

    for i in range(len(flux_levels) - 1):
        curr_group, curr_flux, _, curr_e_max = flux_levels[i]
        next_group, next_flux, next_e_min, _ = flux_levels[i + 1]
        ax.vlines(curr_e_max, curr_flux, next_flux,
                 colors=colors_4g[curr_group], linewidth=2.5,
                 linestyles='-', alpha=0.8)

    ax.set_xlabel('Energy [MeV]', fontsize=11)
    ax.set_ylabel('Flux [n/cm²/s]', fontsize=11)
    ax.set_title('Breeder Material - Four-Group', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1e-12, 20)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    output_file = output_path / 'sfr_tritium_breeder_tallies.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()


def plot_sfr_core_radial_flux(sp, output_dir='tally_figures'):
    """Plot radial flux profile from SFR core mesh (4 energy groups)."""
    output_path = Path(output_dir)

    print("\nCreating SFR core radial flux plot...")

    # Calculate normalization
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)

    # Get mesh tallies
    try:
        total_tally = sp.get_tally(name='sfr_core_mesh_total')
        thermal_tally = sp.get_tally(name='sfr_core_mesh_thermal')
        epithermal_tally = sp.get_tally(name='sfr_core_mesh_epithermal')
        fast_tally = sp.get_tally(name='sfr_core_mesh_fast')
        vfast_tally = sp.get_tally(name='sfr_core_mesh_veryfast')
    except Exception as e:
        print(f"  Warning: Could not find SFR core mesh tallies: {e}")
        return

    # Get mesh info
    mesh_filter = total_tally.find_filter(openmc.MeshFilter)
    mesh = mesh_filter.mesh
    nx, ny, nz = mesh.dimension

    # Calculate mesh volume
    core_edge = inputs['sfr_core_edge'] + inputs['sfr_ss316_wall_thickness']
    axial_height = inputs['sfr_axial_height'] + inputs['sfr_axial_reflector_thickness']

    dx = 2 * core_edge / nx
    dy = 2 * core_edge / ny
    dz = 2 * axial_height / nz
    mesh_volume = dx * dy * dz

    # Reshape and normalize flux data
    flux_total = total_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume
    flux_thermal = thermal_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume
    flux_epithermal = epithermal_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume
    flux_fast = fast_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume
    flux_vfast = vfast_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume

    # Average over Z axis
    flux_total_avg = np.mean(flux_total, axis=2)
    flux_thermal_avg = np.mean(flux_thermal, axis=2)
    flux_epithermal_avg = np.mean(flux_epithermal, axis=2)
    flux_fast_avg = np.mean(flux_fast, axis=2)
    flux_vfast_avg = np.mean(flux_vfast, axis=2)

    # Get radial profile through center
    center_x = nx // 2
    center_y = ny // 2
    x_centers = np.linspace(-core_edge, core_edge, nx)
    r_centers = np.abs(x_centers)

    flux_radial_total = flux_total_avg[:, center_y]
    flux_radial_thermal = flux_thermal_avg[:, center_y]
    flux_radial_epithermal = flux_epithermal_avg[:, center_y]
    flux_radial_fast = flux_fast_avg[:, center_y]
    flux_radial_vfast = flux_vfast_avg[:, center_y]

    # Average left and right sides
    flux_r_total = (flux_radial_total[:center_x][::-1] + flux_radial_total[center_x:]) / 2
    flux_r_thermal = (flux_radial_thermal[:center_x][::-1] + flux_radial_thermal[center_x:]) / 2
    flux_r_epithermal = (flux_radial_epithermal[:center_x][::-1] + flux_radial_epithermal[center_x:]) / 2
    flux_r_fast = (flux_radial_fast[:center_x][::-1] + flux_radial_fast[center_x:]) / 2
    flux_r_vfast = (flux_radial_vfast[:center_x][::-1] + flux_radial_vfast[center_x:]) / 2
    r_plot = r_centers[center_x:]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    colors_flux = {
        'Total': 'black',
        'Very-Fast': 'darkred',
        'Fast': 'red',
        'Epithermal': 'orange',
        'Thermal': 'blue'
    }

    ax.semilogy(r_plot, flux_r_total, label='Total', color=colors_flux['Total'], linewidth=2.5, alpha=0.8)
    ax.semilogy(r_plot, flux_r_vfast, label='Very-Fast (3-20 MeV)', color=colors_flux['Very-Fast'], linewidth=2.5, alpha=0.8)
    ax.semilogy(r_plot, flux_r_fast, label='Fast (100 keV-3 MeV)', color=colors_flux['Fast'], linewidth=2.5, alpha=0.8)
    ax.semilogy(r_plot, flux_r_epithermal, label='Epithermal', color=colors_flux['Epithermal'], linewidth=2.5, alpha=0.8)
    ax.semilogy(r_plot, flux_r_thermal, label='Thermal', color=colors_flux['Thermal'], linewidth=2.5, alpha=0.8)

    # Add boundary lines for SFR geometry
    # Calculate ring boundaries (approximate radial positions)
    assembly_pitch = inputs['sfr_assembly_pitch']

    # Inner fuel outer edge (ring 9)
    r_inner_fuel = assembly_pitch * (inputs['sfr_inner_fuel_rings'] + 0.5)
    # Outer fuel outer edge (ring 12)
    r_outer_fuel = assembly_pitch * (inputs['sfr_inner_fuel_rings'] + inputs['sfr_outer_fuel_rings'] + 0.5)
    # Reflector outer edge (ring 15)
    r_reflector = inputs['sfr_core_edge']
    # SS316 wall outer edge
    r_ss316_wall = r_reflector + inputs['sfr_ss316_wall_thickness']

    ax.axvline(r_inner_fuel, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.5, label='Inner Fuel Edge')
    ax.axvline(r_outer_fuel, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Outer Fuel Edge')
    ax.axvline(r_reflector, color='blue', linestyle=':', linewidth=1.5, alpha=0.5, label='Na Reflector')
    ax.axvline(r_ss316_wall, color='gray', linestyle='-.', linewidth=1.5, alpha=0.5, label='SS316 Wall')

    ax.set_xlabel('Radius [cm]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Flux [n/cm²/s]', fontsize=14, fontweight='bold')
    ax.set_title('SFR Core Radial Flux Profile - Axially Averaged (4-Group)', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, core_edge)

    plt.tight_layout()
    output_file = output_path / 'sfr_core_radial_flux.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_sfr_tritium_assembly_heatmap(sp, output_dir='tally_figures'):
    """Plot 3x3 assembly mesh heatmap centered on SFR tritium breeder."""
    output_path = Path(output_dir)

    print("\nCreating SFR tritium assembly heatmaps...")

    # Calculate normalization
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)

    # Get mesh tallies
    try:
        total_tally = sp.get_tally(name='sfr_tritium_assembly_mesh_total')
        thermal_tally = sp.get_tally(name='sfr_tritium_assembly_mesh_thermal')
        epithermal_tally = sp.get_tally(name='sfr_tritium_assembly_mesh_epithermal')
        fast_tally = sp.get_tally(name='sfr_tritium_assembly_mesh_fast')
        vfast_tally = sp.get_tally(name='sfr_tritium_assembly_mesh_veryfast')
    except Exception as e:
        print(f"  Warning: Could not find SFR assembly mesh tallies: {e}")
        return

    # Get mesh info
    mesh_filter = total_tally.find_filter(openmc.MeshFilter)
    mesh = mesh_filter.mesh
    nx, ny, nz = mesh.dimension  # Should be 300, 300, 1

    # Assembly width and mesh width
    derived = get_derived_dimensions()
    assembly_width = derived['assembly_width']
    mesh_width = 3 * assembly_width
    hex_edge = inputs['sfr_tritium_breeder_edge']

    # Calculate mesh volume
    dx = mesh_width / nx
    dy = mesh_width / ny
    axial_height = inputs['sfr_axial_height'] * 2
    dz = axial_height / nz
    mesh_volume = dx * dy * dz

    # Reshape and normalize all flux types
    flux_total = total_tally.mean.reshape((nz, ny, nx))[0, :, :] * norm_factor / mesh_volume
    flux_thermal = thermal_tally.mean.reshape((nz, ny, nx))[0, :, :] * norm_factor / mesh_volume
    flux_epithermal = epithermal_tally.mean.reshape((nz, ny, nx))[0, :, :] * norm_factor / mesh_volume
    flux_fast = fast_tally.mean.reshape((nz, ny, nx))[0, :, :] * norm_factor / mesh_volume
    flux_vfast = vfast_tally.mean.reshape((nz, ny, nx))[0, :, :] * norm_factor / mesh_volume

    # Create 2x3 subplot for 5 plots (total + 4 groups)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('SFR Tritium Breeder: 3×3 Assembly Flux Distribution (4-Group)',
                 fontsize=18, fontweight='bold')

    # Create coordinate arrays centered on (0, 0) - tritium breeder location
    x_edges = np.linspace(-mesh_width/2, mesh_width/2, nx + 1)
    y_edges = np.linspace(-mesh_width/2, mesh_width/2, ny + 1)

    # Plot data: Total + 4 groups
    flux_data = [
        ('Total', flux_total, axes[0, 0]),
        ('Fast (100 keV - 3 MeV)', flux_fast, axes[0, 1]),
        ('Very-Fast (3 - 20 MeV)', flux_vfast, axes[0, 2]),
        ('Epithermal', flux_epithermal, axes[1, 0]),
        ('Thermal', flux_thermal, axes[1, 1])
    ]

    for flux_name, flux_2d, ax in flux_data:
        # Plot heatmap with log scale
        vmin = np.max(flux_2d) * 1e-3  # Set lower limit to avoid zero issues
        vmax = np.max(flux_2d)

        if vmin > 0:
            im = ax.pcolormesh(x_edges, y_edges, flux_2d,
                              cmap='viridis',
                              norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                              shading='flat')

            # Add hexagonal assembly boundaries
            # Draw hexagons using proper hexagonal lattice positions
            from matplotlib.patches import RegularPolygon

            # Center hexagon (tritium breeder at origin)
            hex_circumradius = hex_edge * 2 / np.sqrt(3)

            hexagon_center = RegularPolygon(
                (0, 0),
                numVertices=6,
                radius=hex_circumradius,
                orientation=0,  # flat-top (orientation='x')
                fill=False,
                edgecolor='white',
                linewidth=3,
                linestyle='-',
                alpha=0.9,
                label='Tritium Breeder'
            )
            ax.add_patch(hexagon_center)

            # Ring 1: 6 hexagons around center
            # For flat-topped hex lattice, nearest neighbors are at:
            pitch = assembly_width
            hex_positions = [
                (pitch, 0),                          # Right
                (pitch/2, pitch * np.sqrt(3)/2),     # Upper-right
                (-pitch/2, pitch * np.sqrt(3)/2),    # Upper-left
                (-pitch, 0),                         # Left
                (-pitch/2, -pitch * np.sqrt(3)/2),   # Lower-left
                (pitch/2, -pitch * np.sqrt(3)/2),    # Lower-right
            ]

            for (x_pos, y_pos) in hex_positions:
                hexagon = RegularPolygon(
                    (x_pos, y_pos),
                    numVertices=6,
                    radius=hex_circumradius,
                    orientation=0,
                    fill=False,
                    edgecolor='white',
                    linewidth=2,
                    linestyle='--',
                    alpha=0.6
                )
                ax.add_patch(hexagon)

            ax.set_xlabel('X [cm]', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y [cm]', fontsize=12, fontweight='bold')
            ax.set_title(f'{flux_name} Flux', fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.set_xlim(-mesh_width/2, mesh_width/2)
            ax.set_ylim(-mesh_width/2, mesh_width/2)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Flux [n/cm²/s]', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'{flux_name}\nNo data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)

    # Hide the empty subplot (2,2)
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_file = output_path / 'sfr_tritium_assembly_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_sfr_core_flux_heatmaps(sp, output_dir='tally_figures'):
    """Plot 3x2 heatmap of axially-averaged flux across entire SFR core (5 flux types).

    Parameters
    ----------
    sp : openmc.StatePoint
        Loaded statepoint file
    output_dir : str
        Directory to save plots
    """
    output_path = Path(output_dir)

    print("\nCreating SFR core flux heatmaps (axially averaged)...")

    # Calculate normalization
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)

    # Get mesh tallies
    try:
        total_tally = sp.get_tally(name='sfr_core_mesh_total')
        thermal_tally = sp.get_tally(name='sfr_core_mesh_thermal')
        epithermal_tally = sp.get_tally(name='sfr_core_mesh_epithermal')
        fast_tally = sp.get_tally(name='sfr_core_mesh_fast')
        vfast_tally = sp.get_tally(name='sfr_core_mesh_veryfast')
    except Exception as e:
        print(f"  Warning: Could not find SFR core mesh tallies: {e}")
        return

    # Get mesh info
    mesh_filter = total_tally.find_filter(openmc.MeshFilter)
    mesh = mesh_filter.mesh
    nx, ny, nz = mesh.dimension

    # Calculate mesh volume
    core_edge = inputs['sfr_core_edge'] + inputs['sfr_ss316_wall_thickness']
    axial_height = inputs['sfr_axial_height'] + inputs['sfr_axial_reflector_thickness']

    dx = 2 * core_edge / nx
    dy = 2 * core_edge / ny
    dz = 2 * axial_height / nz
    mesh_volume = dx * dy * dz

    # Reshape and normalize flux data
    flux_total = total_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume
    flux_thermal = thermal_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume
    flux_epithermal = epithermal_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume
    flux_fast = fast_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume
    flux_vfast = vfast_tally.mean.reshape((nz, ny, nx)).transpose(2, 1, 0) * norm_factor / mesh_volume

    # Average over Z axis (axially averaged)
    flux_total_2d = np.mean(flux_total, axis=2)
    flux_thermal_2d = np.mean(flux_thermal, axis=2)
    flux_epithermal_2d = np.mean(flux_epithermal, axis=2)
    flux_fast_2d = np.mean(flux_fast, axis=2)
    flux_vfast_2d = np.mean(flux_vfast, axis=2)

    # Create coordinate arrays
    x_edges = np.linspace(-core_edge, core_edge, nx + 1)
    y_edges = np.linspace(-core_edge, core_edge, ny + 1)

    # Create 3x2 subplot
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle('SFR Core: Axially-Averaged Flux Distribution (4-Group)',
                 fontsize=20, fontweight='bold')

    # Plot data: 5 flux types
    flux_data = [
        ('Total Flux', flux_total_2d, axes[0, 0]),
        ('Thermal Flux (0 - 0.625 eV)', flux_thermal_2d, axes[0, 1]),
        ('Epithermal Flux (0.625 eV - 100 keV)', flux_epithermal_2d, axes[1, 0]),
        ('Fast Flux (100 keV - 3 MeV)', flux_fast_2d, axes[1, 1]),
        ('Very-Fast Flux (3 - 20 MeV)', flux_vfast_2d, axes[2, 0])
    ]

    for flux_name, flux_2d, ax in flux_data:
        # Plot heatmap with log scale
        flux_max = np.max(flux_2d[flux_2d > 0]) if np.any(flux_2d > 0) else 1.0
        flux_min = flux_max * 1e-4  # 4 orders of magnitude dynamic range

        if flux_max > 0:
            im = ax.pcolormesh(x_edges, y_edges, flux_2d.T,  # Transpose for correct orientation
                              cmap='hot',
                              norm=plt.matplotlib.colors.LogNorm(vmin=flux_min, vmax=flux_max),
                              shading='flat')

            # Add boundary circles for radial regions
            assembly_pitch = inputs['sfr_assembly_pitch']

            # Inner fuel outer edge
            r_inner_fuel = assembly_pitch * (inputs['sfr_inner_fuel_rings'] + 0.5)
            circle1 = plt.Circle((0, 0), r_inner_fuel, fill=False,
                                edgecolor='cyan', linewidth=2, linestyle='--', alpha=0.7,
                                label='Inner Fuel')
            ax.add_patch(circle1)

            # Outer fuel outer edge
            r_outer_fuel = assembly_pitch * (inputs['sfr_inner_fuel_rings'] + inputs['sfr_outer_fuel_rings'] + 0.5)
            circle2 = plt.Circle((0, 0), r_outer_fuel, fill=False,
                                edgecolor='lime', linewidth=2, linestyle='--', alpha=0.7,
                                label='Outer Fuel')
            ax.add_patch(circle2)

            # Reflector edge
            r_reflector = inputs['sfr_core_edge']
            circle3 = plt.Circle((0, 0), r_reflector, fill=False,
                                edgecolor='blue', linewidth=2, linestyle=':', alpha=0.7,
                                label='Na Reflector')
            ax.add_patch(circle3)

            # SS316 wall edge
            r_ss316 = r_reflector + inputs['sfr_ss316_wall_thickness']
            circle4 = plt.Circle((0, 0), r_ss316, fill=False,
                                edgecolor='white', linewidth=2.5, linestyle='-.', alpha=0.9,
                                label='SS316 Wall')
            ax.add_patch(circle4)

            ax.set_xlabel('X [cm]', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y [cm]', fontsize=12, fontweight='bold')
            ax.set_title(flux_name, fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.set_xlim(-core_edge, core_edge)
            ax.set_ylim(-core_edge, core_edge)

            # Add legend only to first plot
            if flux_name == 'Total Flux':
                ax.legend(loc='upper right', fontsize=9, framealpha=0.8)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Flux [n/cm²/s]', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'{flux_name}\nNo data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)

    # Hide the empty subplot (2,1)
    axes[2, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_file = output_path / 'sfr_core_flux_heatmaps.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_all_sfr_tallies(statepoint_path='simulation_raw/statepoint.250.h5', output_dir='tally_figures'):
    """Plot all SFR tallies from a statepoint file.

    Parameters
    ----------
    statepoint_path : str
        Path to the statepoint file
    output_dir : str
        Directory to save plots
    """
    # Check if file exists
    if not Path(statepoint_path).exists():
        print(f"Error: Statepoint file not found: {statepoint_path}")
        return

    print("\n" + "="*70)
    print("SFR TALLY VISUALIZATION")
    print("="*70)

    # Load statepoint
    print(f"\nLoading statepoint from: {statepoint_path}")
    sp = openmc.StatePoint(statepoint_path)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate all plots
    plot_sfr_tritium_breeder_tallies(sp, output_dir)
    plot_sfr_core_radial_flux(sp, output_dir)
    plot_sfr_core_flux_heatmaps(sp, output_dir)
    plot_sfr_tritium_assembly_heatmap(sp, output_dir)

    print("\n" + "="*70)
    print("ALL SFR PLOTS COMPLETED!")
    print("="*70)


if __name__ == '__main__':
    import sys

    # Default statepoint path
    statepoint_path = 'simulation_raw/statepoint.250.h5'

    # Check if user provided a different path
    if len(sys.argv) > 1:
        statepoint_path = sys.argv[1]

    plot_all_sfr_tallies(statepoint_path)
