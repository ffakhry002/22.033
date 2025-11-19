"""
Convert OpenMC log1001 flux AND surface current tallies to UKAEA-1102 energy group structure.
Enhanced version with detailed energy breakdown tables in summary.

This script:
1. Extracts log1001 flux tallies from OpenMC statepoint files
2. Extracts log1001 surface current tallies
3. Rebins them to UKAEA-1102 energy structure (1102 groups)
4. Outputs in FISPACT-II format (6 values per line, scientific notation)
5. Creates detailed summary with thermal/epithermal/fast breakdowns
"""
import numpy as np
import openmc
from pathlib import Path
import sys

sys.path.append('.')
from inputs import inputs, get_derived_dimensions
from tallies import calc_norm_factor


def parse_ukaea_1102_file(filepath='ukaea_1102_fluxes/UKAEA1102Groups.txt'):
    """Parse UKAEA-1102 energy group structure from file."""
    boundaries = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Format: "group_number    energy_value"
    for line in lines:
        line = line.strip()
        if line and '\t' in line:  # Tab-separated values
            parts = line.split('\t')
            if len(parts) == 2:
                try:
                    # Try to parse as group number and energy
                    group_num = int(parts[0])
                    energy = float(parts[1])
                    boundaries.append(energy)
                except (ValueError, IndexError):
                    continue

    if len(boundaries) == 1102:
        boundaries.append(1.0e-5)

    print(f"Parsed {len(boundaries)} energy boundaries from UKAEA-1102 file")

    if len(boundaries) != 1103:
        print(f"Warning: Expected 1103 boundaries but got {len(boundaries)}")

    return np.array(boundaries)


def rebin_flux_conservative(old_energies, old_flux, new_energies):
    """Rebin flux/current conservatively from log1001 to UKAEA-1102 structure."""

    # Convert new_energies to ascending order for easier processing
    new_energies_asc = new_energies[::-1]
    n_new = len(new_energies_asc) - 1
    new_flux = np.zeros(n_new)

    # For each new bin (in ascending order)
    for i in range(n_new):
        new_low = new_energies_asc[i]
        new_high = new_energies_asc[i+1]

        # Find overlapping old bins
        for j in range(len(old_flux)):
            old_low = old_energies[j]
            old_high = old_energies[j+1]

            # Calculate overlap
            overlap_low = max(new_low, old_low)
            overlap_high = min(new_high, old_high)

            if overlap_high > overlap_low:
                # Calculate the fraction of old bin that overlaps with new bin
                # Using logarithmic weighting (lethargy)
                if old_high > old_low and overlap_high > overlap_low:
                    overlap_lethargy = np.log(overlap_high / overlap_low)
                    old_lethargy = np.log(old_high / old_low)
                    fraction = overlap_lethargy / old_lethargy

                    # Add weighted contribution to new bin
                    new_flux[i] += old_flux[j] * fraction

    # Reverse to match descending order of input new_energies
    return new_flux[::-1]


def calculate_energy_groups(data, energies):
    """Calculate thermal, epithermal, and fast components from UKAEA-1102 data.

    Energy groups (from inputs.py):
    - Thermal: < thermal_cutoff eV
    - Epithermal: thermal_cutoff eV - epithermal_cutoff eV
    - Fast: > epithermal_cutoff eV
    """
    # Energy boundaries from inputs (in eV)
    thermal_cutoff = inputs['thermal_cutoff']  # Default: 0.625 eV
    fast_cutoff = inputs['epithermal_cutoff']  # Default: 100 keV = 1e5 eV

    thermal_sum = 0
    epithermal_sum = 0
    fast_sum = 0

    # energies are in descending order
    for i in range(len(data)):
        e_high = energies[i]
        e_low = energies[i+1] if i+1 < len(energies) else 1e-5

        # Classify based on energy
        if e_high <= thermal_cutoff:
            thermal_sum += data[i]
        elif e_low >= fast_cutoff:
            fast_sum += data[i]
        else:
            # Could be epithermal or split across boundaries
            # For simplicity, assign to epithermal
            epithermal_sum += data[i]

    return {
        'thermal': thermal_sum,
        'epithermal': epithermal_sum,
        'fast': fast_sum,
        'total': thermal_sum + epithermal_sum + fast_sum
    }


def extract_cell_fluxes(sp, norm_factor, ukaea_energies, output_path):
    """Extract and convert cell-based flux tallies."""

    # Get derived dimensions for volume calculations
    derived = get_derived_dimensions()

    # Calculate cell volumes (cylindrical shells)
    r_core = inputs['r_core']
    r_outer_tank = derived['r_outer_tank']
    r_rpv_1 = derived['r_rpv_1']
    r_rpv_2 = derived['r_rpv_2']
    r_lithium = derived['r_lithium']
    height = derived['z_top'] - derived['z_bottom']

    # Determine lithium blanket inner radius based on moderator configuration
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

    # Define regions to process
    regions = ['outer_tank', 'rpv_inner', 'rpv_outer']
    if inputs['enable_moderator_region']:
        regions.append('moderator')
    regions.append('lithium_blanket')

    print("\n" + "="*70)
    print("Processing Cell Flux Tallies")
    print("="*70)

    # Store results for summary
    results = {}

    # Process each region
    for region in regions:
        print(f"\nProcessing flux in: {region}")
        print("-" * 40)

        # Get log1001 tally
        tally_name = f'{region}_flux_log1001'
        try:
            tally = sp.get_tally(name=tally_name)
        except:
            print(f"  ERROR: Could not find tally {tally_name}")
            continue

        # Extract energy filter and flux data
        energy_filter = tally.find_filter(openmc.EnergyFilter)
        energy_bins = energy_filter.bins  # Shape: (1001, 2) in eV, ascending

        # Get bin boundaries (ascending order)
        log1001_boundaries = np.concatenate(([energy_bins[0, 0]], energy_bins[:, 1]))

        # Get flux values and normalize (per source particle -> absolute flux)
        cell_volume = region_volumes[region]
        flux_values = tally.mean.flatten() * norm_factor / cell_volume  # n/cm²/s
        flux_std = tally.std_dev.flatten() * norm_factor / cell_volume

        print(f"  Cell volume: {cell_volume:.2e} cm³")
        print(f"  Original energy bins: {len(flux_values)}")
        print(f"  Energy range: {log1001_boundaries[0]:.2e} to {log1001_boundaries[-1]:.2e} eV")
        print(f"  Total flux: {np.sum(flux_values):.4e} n/cm²/s")

        # Rebin to UKAEA-1102 structure
        ukaea_flux = rebin_flux_conservative(log1001_boundaries, flux_values, ukaea_energies)

        print(f"  Rebinned groups: {len(ukaea_flux)}")
        print(f"  Total flux (after rebinning): {np.sum(ukaea_flux):.4e} n/cm²/s")

        # Check conservation
        conservation_ratio = np.sum(ukaea_flux) / np.sum(flux_values) if np.sum(flux_values) > 0 else 0
        print(f"  Flux conservation: {conservation_ratio:.4f}")

        # Calculate energy groups
        energy_breakdown = calculate_energy_groups(ukaea_flux, ukaea_energies)

        # Store results with energy breakdown
        results[region] = {
            'flux': ukaea_flux,
            'total': np.sum(ukaea_flux),
            'max': np.max(ukaea_flux),
            'volume': cell_volume,
            'thermal': energy_breakdown['thermal'],
            'epithermal': energy_breakdown['epithermal'],
            'fast': energy_breakdown['fast']
        }

        # Write output file in FISPACT-II format
        output_file = output_path / 'cell_fluxes' / f'{region}_flux_ukaea1102.txt'
        output_file.parent.mkdir(exist_ok=True)
        write_fispact_format(output_file, ukaea_flux, ukaea_energies,
                           f"{region} flux", cell_volume, "n/cm²/s", "Cell volume")
        print(f"  Output saved to: {output_file}")

    return results


def extract_surface_currents(sp, norm_factor, ukaea_energies, output_path):
    """Extract and convert surface current tallies (outward only)."""

    # Get derived dimensions
    derived = get_derived_dimensions()

    # Define surfaces and their radii
    surfaces = {
        'core': inputs['r_core'],
        'outer_tank': derived['r_outer_tank'],
        'rpv_inner': derived['r_rpv_1'],
        'rpv_outer': derived['r_rpv_2'],
        'lithium': derived['r_lithium']
    }

    # Add moderator region surfaces if enabled
    if inputs['enable_moderator_region']:
        surfaces['moderator'] = derived['r_moderator']
        surfaces['wall_divider'] = derived['r_wall_divider']

    # Calculate surface areas (cylindrical surfaces)
    # Using active fuel height for consistency
    height = derived['z_fuel_top'] - derived['z_fuel_bottom']
    surface_areas = {}
    for surf_name, radius in surfaces.items():
        surface_areas[surf_name] = 2 * np.pi * radius * height  # cm²

    print("\n" + "="*70)
    print("Processing Surface Current Tallies (Outward Only)")
    print("="*70)

    # Store results
    results = {}

    # Process each surface
    for surf_name, radius in surfaces.items():
        print(f"\nProcessing outward current at: {surf_name} (r={radius:.1f} cm)")
        print("-" * 40)

        # Get log1001 surface current tally
        tally_name = f'surface_{surf_name}_current_log1001'
        try:
            tally = sp.get_tally(name=tally_name)
        except:
            print(f"  ERROR: Could not find tally {tally_name}")
            continue

        # Extract energy filter and current data
        energy_filter = tally.find_filter(openmc.EnergyFilter)
        energy_bins = energy_filter.bins  # Shape: (1001, 2) in eV, ascending

        # Get bin boundaries (ascending order)
        log1001_boundaries = np.concatenate(([energy_bins[0, 0]], energy_bins[:, 1]))

        # Get current values and normalize
        surface_area = surface_areas[surf_name]
        # Current is already outward-only due to CellFromFilter
        current_values = tally.mean.flatten() * norm_factor  # n/s (total current)
        current_density = current_values / surface_area  # n/cm²/s (current density)
        current_std = tally.std_dev.flatten() * norm_factor / surface_area

        print(f"  Surface area: {surface_area:.2e} cm²")
        print(f"  Radius: {radius:.1f} cm")
        print(f"  Height: {height:.1f} cm")
        print(f"  Original energy bins: {len(current_density)}")
        print(f"  Energy range: {log1001_boundaries[0]:.2e} to {log1001_boundaries[-1]:.2e} eV")
        print(f"  Total outward current: {np.sum(current_values):.4e} n/s")
        print(f"  Total outward current density: {np.sum(current_density):.4e} n/cm²/s")

        # Rebin to UKAEA-1102 structure
        ukaea_current = rebin_flux_conservative(log1001_boundaries, current_density, ukaea_energies)

        print(f"  Rebinned groups: {len(ukaea_current)}")
        print(f"  Total current density (after rebinning): {np.sum(ukaea_current):.4e} n/cm²/s")

        # Check conservation
        conservation_ratio = np.sum(ukaea_current) / np.sum(current_density) if np.sum(current_density) > 0 else 0
        print(f"  Current conservation: {conservation_ratio:.4f}")

        # Calculate energy groups
        energy_breakdown = calculate_energy_groups(ukaea_current, ukaea_energies)

        # Store results with energy breakdown
        results[surf_name] = {
            'current_density': ukaea_current,
            'total_density': np.sum(ukaea_current),
            'total_current': np.sum(ukaea_current) * surface_area,
            'max_density': np.max(ukaea_current),
            'area': surface_area,
            'radius': radius,
            'thermal_density': energy_breakdown['thermal'],
            'epithermal_density': energy_breakdown['epithermal'],
            'fast_density': energy_breakdown['fast'],
            'thermal_current': energy_breakdown['thermal'] * surface_area,
            'epithermal_current': energy_breakdown['epithermal'] * surface_area,
            'fast_current': energy_breakdown['fast'] * surface_area
        }

        # Write output file for current density
        output_file = output_path / 'surface_currents' / f'surface_{surf_name}_current_ukaea1102.txt'
        output_file.parent.mkdir(exist_ok=True)
        write_fispact_format(output_file, ukaea_current, ukaea_energies,
                           f"{surf_name} surface outward current", surface_area,
                           "n/cm²/s", "Surface area")
        print(f"  Output saved to: {output_file}")

        # Also write total current (not density) file
        output_file_total = output_path / 'surface_currents' / f'surface_{surf_name}_current_total_ukaea1102.txt'
        ukaea_current_total = ukaea_current * surface_area  # Convert to total n/s
        write_fispact_format(output_file_total, ukaea_current_total, ukaea_energies,
                           f"{surf_name} surface outward current (total)", surface_area,
                           "n/s", "Surface area")
        print(f"  Total current saved to: {output_file_total}")

    return results


def extract_tritium_breeder_currents(sp, norm_factor, ukaea_energies, output_path):
    """Extract and convert tritium breeder assembly surface current tallies (INWARD)."""

    derived = get_derived_dimensions()

    # Check if this is SFR or CANDU
    is_sfr = inputs.get('assembly_type') == 'sodium'

    if is_sfr:
        # SFR tritium breeder - single hexagonal surface
        # Current enters from cladding into breeder bundle
        surfaces = {
            'sfr_tritium_breeder': {
                'edge_length': inputs['sfr_tritium_breeder_edge'] - 1.0,  # Inner hex edge (minus cladding)
                'description': 'SFR Tritium Breeder Hexagonal Surface (INWARD)',
                'height': inputs['sfr_axial_height'] * 2  # Full height
            }
        }

        # Calculate hexagonal surface area
        # Perimeter = 6 * edge_length, Area = Perimeter * height
        for surf_info in surfaces.values():
            edge = surf_info['edge_length']
            height = surf_info['height']
            perimeter = 6 * edge
            surf_info['area'] = perimeter * height  # cm²
            surf_info['radius'] = edge  # For compatibility, store edge as "radius"

    else:
        # CANDU tritium breeder - cylindrical surfaces
        surfaces = {
            'tritium_calandria_outer': {
                'radius': inputs['candu_calandria_or'],  # 6.7526 cm
                'description': 'Calandria Outer (from moderator, INWARD)'
            },
            'tritium_pt_inner': {
                'radius': inputs['candu_pressure_tube_ir'],  # 5.1689 cm
                'description': 'Pressure Tube Inner (from PT, INWARD)'
            }
        }

    # Calculate surface areas for CANDU (already done for SFR above)
    if not is_sfr:
        height = derived['z_fuel_top'] - derived['z_fuel_bottom']
        for surf_info in surfaces.values():
            surf_info['area'] = 2 * np.pi * surf_info['radius'] * height  # cm²

    print("\n" + "="*70)
    if is_sfr:
        print("Processing SFR Tritium Breeder Surface Current Tallies (INWARD)")
    else:
        print("Processing CANDU Tritium Breeder Surface Current Tallies (INWARD)")
    print("="*70)

    results = {}

    for surf_name, surf_info in surfaces.items():
        print(f"\nProcessing INWARD current at: {surf_name}")
        print(f"  Description: {surf_info['description']}")
        print("-" * 40)

        # Get log1001 surface current tally
        if is_sfr:
            # SFR uses single unified tally name
            tally_name = 'sfr_tritium_current_log1001'
        else:
            # CANDU uses per-surface tally names
            tally_name = f'{surf_name}_current_log1001'

        try:
            tally = sp.get_tally(name=tally_name)
        except:
            print(f"  ERROR: Could not find tally {tally_name}")
            continue

        # Extract energy filter and current data
        energy_filter = tally.find_filter(openmc.EnergyFilter)
        energy_bins = energy_filter.bins  # Shape: (1001, 2) in eV, ascending

        # Get bin boundaries (ascending order)
        log1001_boundaries = np.concatenate(([energy_bins[0, 0]], energy_bins[:, 1]))

        # Get current values and normalize
        surface_area = surf_info['area']
        radius = surf_info['radius']

        # Current is NEGATIVE for inward (CellFromFilter gives negative for inward)
        # Multiply by -1 to get positive inward current
        current_values = -tally.mean.flatten() * norm_factor  # n/s (total INWARD current, now positive)
        current_density = current_values / surface_area  # n/cm²/s (current density)
        current_std = tally.std_dev.flatten() * norm_factor / surface_area

        print(f"  Surface area: {surface_area:.2e} cm²")
        print(f"  Radius: {radius:.4f} cm")
        print(f"  Height: {height:.1f} cm")
        print(f"  Original energy bins: {len(current_density)}")
        print(f"  Energy range: {log1001_boundaries[0]:.2e} to {log1001_boundaries[-1]:.2e} eV")
        print(f"  Total INWARD current: {np.sum(current_values):.4e} n/s")
        print(f"  Total INWARD current density: {np.sum(current_density):.4e} n/cm²/s")

        # Rebin to UKAEA-1102 structure
        ukaea_current = rebin_flux_conservative(log1001_boundaries, current_density, ukaea_energies)

        print(f"  Rebinned groups: {len(ukaea_current)}")
        print(f"  Total current density (after rebinning): {np.sum(ukaea_current):.4e} n/cm²/s")

        # Check conservation
        conservation_ratio = np.sum(ukaea_current) / np.sum(current_density) if np.sum(current_density) > 0 else 0
        print(f"  Current conservation: {conservation_ratio:.4f}")

        # Calculate energy groups
        energy_breakdown = calculate_energy_groups(ukaea_current, ukaea_energies)

        # Store results
        results[surf_name] = {
            'current_density': ukaea_current,
            'total_density': np.sum(ukaea_current),
            'total_current': np.sum(ukaea_current) * surface_area,
            'max_density': np.max(ukaea_current),
            'area': surface_area,
            'radius': radius,
            'description': surf_info['description'],
            'thermal_density': energy_breakdown['thermal'],
            'epithermal_density': energy_breakdown['epithermal'],
            'fast_density': energy_breakdown['fast'],
            'thermal_current': energy_breakdown['thermal'] * surface_area,
            'epithermal_current': energy_breakdown['epithermal'] * surface_area,
            'fast_current': energy_breakdown['fast'] * surface_area
        }

        # Write output file for current density
        output_file = output_path / 'tritium_breeder_currents' / f'{surf_name}_current_ukaea1102.txt'
        output_file.parent.mkdir(exist_ok=True)
        write_fispact_format(output_file, ukaea_current, ukaea_energies,
                           f"{surf_info['description']}", surface_area,
                           "n/cm²/s", "Surface area")
        print(f"  Output saved to: {output_file}")

        # Also write total current (not density) file
        output_file_total = output_path / 'tritium_breeder_currents' / f'{surf_name}_current_total_ukaea1102.txt'
        ukaea_current_total = ukaea_current * surface_area  # Convert to total n/s
        write_fispact_format(output_file_total, ukaea_current_total, ukaea_energies,
                           f"{surf_info['description']} (total)", surface_area,
                           "n/s", "Surface area")
        print(f"  Total current saved to: {output_file_total}")

    return results


def write_fispact_format(filename, flux_values, energies, description, geometry_param, units, param_name):
    """Write data in FISPACT-II format."""
    with open(filename, 'w') as f:
        # Write header comments
        f.write(f"# UKAEA-1102 spectrum: {description}\n")
        f.write(f"# Number of groups: {len(flux_values)}\n")
        f.write(f"# Energy range: {energies[-1]:.4e} to {energies[0]:.4e} eV\n")
        f.write(f"# {param_name}: {geometry_param:.4e} cm{'³' if 'volume' in param_name.lower() else '²'}\n")
        f.write(f"# Units: {units}\n")
        f.write(f"# Total: {np.sum(flux_values):.4e} {units}\n")
        f.write("#\n")

        # Write values, 6 per line
        for i in range(0, len(flux_values), 6):
            line_values = flux_values[i:min(i+6, len(flux_values))]
            # Format each value
            formatted_values = []
            for val in line_values:
                if val == 0:
                    formatted_values.append('0.0000e+00')
                else:
                    exp_str = f'{val:.4e}'
                    formatted_values.append(exp_str)

            line = ' '.join(formatted_values)
            f.write(line + ' \n')


def write_detailed_summary(output_path, statepoint_file, keff, power_mw, norm_factor,
                          flux_results, current_results, tritium_current_results=None):
    """Write detailed summary file with energy breakdowns."""

    # Get energy boundaries from inputs
    thermal_cutoff_eV = inputs['thermal_cutoff']
    epithermal_cutoff_eV = inputs['epithermal_cutoff']
    epithermal_cutoff_keV = epithermal_cutoff_eV / 1e3  # Convert to keV for display

    summary_file = output_path / 'conversion_summary.txt'
    with open(summary_file, 'w') as f:
        # Header
        f.write("="*100 + "\n")
        f.write(" "*35 + "UKAEA-1102 CONVERSION SUMMARY\n")
        f.write("="*100 + "\n")

        # Basic info
        f.write("\nSIMULATION PARAMETERS:\n")
        f.write("-"*50 + "\n")
        f.write(f"Statepoint:    {statepoint_file}\n")
        f.write(f"k-effective:   {keff.n:.6f} +/- {keff.s:.6f}\n")
        f.write(f"Power:         {power_mw} MW\n")
        f.write(f"Normalization: {norm_factor:.4e} n/s\n")

        # Energy group definitions
        f.write("\nENERGY GROUP DEFINITIONS:\n")
        f.write("-"*50 + "\n")
        f.write(f"Thermal:     < {thermal_cutoff_eV} eV\n")
        f.write(f"Epithermal:  {thermal_cutoff_eV} eV - {epithermal_cutoff_keV} keV\n")
        f.write(f"Fast:        > {epithermal_cutoff_keV} keV\n")

        # Cell flux results with energy breakdown
        if flux_results:
            f.write("\n" + "="*100 + "\n")
            f.write("CELL FLUX RESULTS\n")
            f.write("="*100 + "\n")

            # Summary table
            f.write("\nOverview:\n")
            f.write("-"*50 + "\n")
            f.write(f"{'Region':<15} {'Volume [cm³]':<15} {'Total Flux [n/cm²/s]':<20}\n")
            f.write("-"*50 + "\n")
            for region, data in flux_results.items():
                f.write(f"{region:<15} {data['volume']:<15.2e} {data['total']:<20.4e}\n")

            # Detailed energy breakdown table - ABSOLUTE VALUES
            f.write("\n" + "="*100 + "\n")
            f.write("ENERGY BREAKDOWN - ABSOLUTE FLUX [n/cm²/s]\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Region':<15} {'Thermal':<22} {'Epithermal':<22} {'Fast':<22} {'Total':<22}\n")
            f.write(f"{'':15} {'(<' + str(thermal_cutoff_eV) + ' eV)':<22} "
                   f"{'(' + str(thermal_cutoff_eV) + 'eV-' + str(epithermal_cutoff_keV) + 'keV)':<22} "
                   f"{'(>' + str(epithermal_cutoff_keV) + 'keV)':<22} {'':<22}\n")
            f.write("-"*100 + "\n")

            for region, data in flux_results.items():
                f.write(f"{region:<15} {data['thermal']:<22.4e} {data['epithermal']:<22.4e} "
                       f"{data['fast']:<22.4e} {data['total']:<22.4e}\n")

            # Energy breakdown table - PERCENTAGES
            f.write("\n" + "="*100 + "\n")
            f.write("ENERGY BREAKDOWN - PERCENTAGE\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Region':<15} {'Thermal %':<22} {'Epithermal %':<22} {'Fast %':<22} {'Total':<22}\n")
            f.write("-"*100 + "\n")

            for region, data in flux_results.items():
                total = data['total']
                t_pct = (data['thermal']/total*100) if total > 0 else 0
                e_pct = (data['epithermal']/total*100) if total > 0 else 0
                f_pct = (data['fast']/total*100) if total > 0 else 0
                f.write(f"{region:<15} {t_pct:<22.2f} {e_pct:<22.2f} {f_pct:<22.2f} {'100.00':<22}\n")

        # Surface current results with energy breakdown
        if current_results:
            f.write("\n" + "="*100 + "\n")
            f.write("SURFACE CURRENT RESULTS (OUTWARD ONLY)\n")
            f.write("="*100 + "\n")

            # Overview table
            f.write("\nOverview:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Surface':<15} {'Radius [cm]':<12} {'Area [cm²]':<15} "
                   f"{'Total Current [n/s]':<22} {'Current Density [n/cm²/s]':<25}\n")
            f.write("-"*80 + "\n")
            for surface, data in current_results.items():
                f.write(f"{surface:<15} {data['radius']:<12.1f} {data['area']:<15.2e} "
                       f"{data['total_current']:<22.4e} {data['total_density']:<25.4e}\n")

            # Energy breakdown - CURRENT DENSITY [n/cm²/s]
            f.write("\n" + "="*100 + "\n")
            f.write("ENERGY BREAKDOWN - CURRENT DENSITY [n/cm²/s]\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Surface':<15} {'Radius':<10} {'Thermal':<20} {'Epithermal':<20} {'Fast':<20} {'Total':<20}\n")
            f.write(f"{'':15} {'[cm]':<10} {'(<' + str(thermal_cutoff_eV) + ' eV)':<20} "
                   f"{'(' + str(thermal_cutoff_eV) + 'eV-' + str(epithermal_cutoff_keV) + 'keV)':<20} "
                   f"{'(>' + str(epithermal_cutoff_keV) + 'keV)':<20} {'':<20}\n")
            f.write("-"*100 + "\n")

            for surface, data in current_results.items():
                f.write(f"{surface:<15} {data['radius']:<10.1f} {data['thermal_density']:<20.4e} "
                       f"{data['epithermal_density']:<20.4e} {data['fast_density']:<20.4e} "
                       f"{data['total_density']:<20.4e}\n")

            # Energy breakdown - TOTAL CURRENT [n/s]
            f.write("\n" + "="*100 + "\n")
            f.write("ENERGY BREAKDOWN - TOTAL CURRENT [n/s]\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Surface':<15} {'Radius':<10} {'Thermal':<20} {'Epithermal':<20} {'Fast':<20} {'Total':<20}\n")
            f.write(f"{'':15} {'[cm]':<10} {'(<' + str(thermal_cutoff_eV) + ' eV)':<20} "
                   f"{'(' + str(thermal_cutoff_eV) + 'eV-' + str(epithermal_cutoff_keV) + 'keV)':<20} "
                   f"{'(>' + str(epithermal_cutoff_keV) + 'keV)':<20} {'':<20}\n")
            f.write("-"*100 + "\n")

            for surface, data in current_results.items():
                f.write(f"{surface:<15} {data['radius']:<10.1f} {data['thermal_current']:<20.4e} "
                       f"{data['epithermal_current']:<20.4e} {data['fast_current']:<20.4e} "
                       f"{data['total_current']:<20.4e}\n")

            # Energy breakdown - PERCENTAGES
            f.write("\n" + "="*100 + "\n")
            f.write("ENERGY BREAKDOWN - PERCENTAGE\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Surface':<15} {'Radius':<10} {'Thermal %':<20} {'Epithermal %':<20} {'Fast %':<20} {'Total':<20}\n")
            f.write("-"*100 + "\n")

            for surface, data in current_results.items():
                total = data['total_density']
                t_pct = (data['thermal_density']/total*100) if total > 0 else 0
                e_pct = (data['epithermal_density']/total*100) if total > 0 else 0
                f_pct = (data['fast_density']/total*100) if total > 0 else 0
                f.write(f"{surface:<15} {data['radius']:<10.1f} {t_pct:<20.2f} "
                       f"{e_pct:<20.2f} {f_pct:<20.2f} {'100.00':<20}\n")

        # Add tritium breeder currents section if available
        if tritium_current_results:
            f.write("\n" + "="*100 + "\n")
            f.write("TRITIUM BREEDER SURFACE CURRENTS (INWARD)\n")
            f.write("="*100 + "\n")

            # Energy breakdown - CURRENT DENSITY [n/cm²/s]
            f.write("\n" + "="*100 + "\n")
            f.write("ENERGY BREAKDOWN - CURRENT DENSITY [n/cm²/s]\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Surface':<30} {'Radius':<10} {'Thermal':<20} {'Epithermal':<20} {'Fast':<20} {'Total':<20}\n")
            f.write(f"{'':30} {'[cm]':<10} {'(<' + str(thermal_cutoff_eV) + ' eV)':<20} "
                   f"{'(' + str(thermal_cutoff_eV) + 'eV-' + str(epithermal_cutoff_keV) + 'keV)':<20} "
                   f"{'(>' + str(epithermal_cutoff_keV) + 'keV)':<20} {'':<20}\n")
            f.write("-"*100 + "\n")

            for surface, data in tritium_current_results.items():
                f.write(f"{data['description']:<30} {data['radius']:<10.4f} {data['thermal_density']:<20.4e} "
                       f"{data['epithermal_density']:<20.4e} {data['fast_density']:<20.4e} "
                       f"{data['total_density']:<20.4e}\n")

            # Energy breakdown - TOTAL CURRENT [n/s]
            f.write("\n" + "="*100 + "\n")
            f.write("ENERGY BREAKDOWN - TOTAL CURRENT [n/s]\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Surface':<30} {'Radius':<10} {'Thermal':<20} {'Epithermal':<20} {'Fast':<20} {'Total':<20}\n")
            f.write(f"{'':30} {'[cm]':<10} {'(<' + str(thermal_cutoff_eV) + ' eV)':<20} "
                   f"{'(' + str(thermal_cutoff_eV) + 'eV-' + str(epithermal_cutoff_keV) + 'keV)':<20} "
                   f"{'(>' + str(epithermal_cutoff_keV) + 'keV)':<20} {'':<20}\n")
            f.write("-"*100 + "\n")

            for surface, data in tritium_current_results.items():
                f.write(f"{data['description']:<30} {data['radius']:<10.4f} {data['thermal_current']:<20.4e} "
                       f"{data['epithermal_current']:<20.4e} {data['fast_current']:<20.4e} "
                       f"{data['total_current']:<20.4e}\n")

            # Energy breakdown - PERCENTAGES
            f.write("\n" + "="*100 + "\n")
            f.write("ENERGY BREAKDOWN - PERCENTAGE\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Surface':<30} {'Radius':<10} {'Thermal %':<20} {'Epithermal %':<20} {'Fast %':<20} {'Total':<20}\n")
            f.write("-"*100 + "\n")

            for surface, data in tritium_current_results.items():
                total = data['total_density']
                t_pct = (data['thermal_density']/total*100) if total > 0 else 0
                e_pct = (data['epithermal_density']/total*100) if total > 0 else 0
                f_pct = (data['fast_density']/total*100) if total > 0 else 0
                f.write(f"{data['description']:<30} {data['radius']:<10.4f} {t_pct:<20.2f} "
                       f"{e_pct:<20.2f} {f_pct:<20.2f} {'100.00':<20}\n")

        # File locations
        f.write("\n" + "="*100 + "\n")
        f.write("OUTPUT FILE LOCATIONS\n")
        f.write("-"*100 + "\n")
        f.write(f"Cell fluxes:      {output_path}/cell_fluxes/\n")
        f.write(f"Surface currents: {output_path}/surface_currents/\n")
        if tritium_current_results:
            f.write(f"Tritium breeder currents: {output_path}/tritium_breeder_currents/\n")
        f.write("\n" + "="*100 + "\n")

    print(f"\nDetailed summary saved to: {summary_file}")


def extract_and_convert_all(statepoint_file='simulation_raw/statepoint.250.h5',
                            output_dir='ukaea_1102_fluxes',
                            ukaea_file='ukaea_1102_fluxes/UKAEA1102Groups.txt'):
    """Extract and convert both flux and surface current tallies."""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load UKAEA-1102 energy structure
    print("="*70)
    print("Loading UKAEA-1102 energy structure...")
    print("="*70)
    ukaea_energies = parse_ukaea_1102_file(ukaea_file)
    print(f"Energy range: {ukaea_energies[0]:.2e} eV (max) to {ukaea_energies[-1]:.2e} eV (min)")
    print(f"Number of groups: {len(ukaea_energies) - 1}")

    # Load statepoint
    print("\n" + "="*70)
    print(f"Loading statepoint: {statepoint_file}")
    print("="*70)
    sp = openmc.StatePoint(statepoint_file)

    # Calculate normalization factor
    power_mw = inputs['core_power']
    norm_factor = calc_norm_factor(power_mw, sp)
    print(f"Reactor power: {power_mw} MW")
    print(f"Normalization factor: {norm_factor:.4e} n/s")

    # Get k-effective
    keff = sp.keff
    print(f"k-effective: {keff}")

    # Extract cell fluxes
    flux_results = extract_cell_fluxes(sp, norm_factor, ukaea_energies, output_path)

    # Extract surface currents
    current_results = extract_surface_currents(sp, norm_factor, ukaea_energies, output_path)

    # Extract tritium breeder surface currents
    tritium_current_results = extract_tritium_breeder_currents(sp, norm_factor, ukaea_energies, output_path)

    # Print quick summary
    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)

    # Write detailed summary with energy breakdowns
    write_detailed_summary(output_path, statepoint_file, keff, power_mw, norm_factor,
                          flux_results, current_results, tritium_current_results)

    print("\nConversion complete!")
    print(f"Check {output_path}/conversion_summary.txt for detailed energy breakdowns")

    return flux_results, current_results, tritium_current_results


if __name__ == '__main__':
    # Run the conversion for both fluxes and currents
    extract_and_convert_all()
