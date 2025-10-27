"""
Convert OpenMC log1001 flux tallies to UKAEA-1102 energy group structure.

This script:
1. Extracts log1001 tallies from OpenMC statepoint files
2. Rebins them to UKAEA-1102 energy structure (1102 groups)
3. Outputs in FISPACT-II format (6 values per line, scientific notation)
"""
import numpy as np
import openmc
from pathlib import Path
import sys

sys.path.append('.')
from inputs import inputs, get_derived_dimensions
from tallies import calc_norm_factor


def parse_ukaea_1102_file(filepath='simulation_raw/UKAEA1102Groups.txt'):
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
                # Calcula the fraction of old bin that overlaps with new bin
                # Using logarithmic weighting (lethargy)
                if old_high > old_low and overlap_high > overlap_low:
                    overlap_lethargy = np.log(overlap_high / overlap_low)
                    old_lethargy = np.log(old_high / old_low)
                    fraction = overlap_lethargy / old_lethargy

                    # Add weighted contribution to new bin
                    new_flux[i] += old_flux[j] * fraction

    # Reverse to match descending order of input new_energies
    return new_flux[::-1]


def extract_and_convert_tallies(statepoint_file='simulation_raw/statepoint.250.h5',
                                output_dir='ukaea_1102_fluxes',
                                ukaea_file='simulation_raw/UKAEA1102Groups.txt'):
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

    # Get derived dimensions for volume calculations
    derived = get_derived_dimensions()

    # Calculate cell volumes (cylindrical shells)
    r_core = inputs['r_core']
    r_outer_tank = derived['r_outer_tank']
    r_rpv_1 = derived['r_rpv_1']
    r_rpv_2 = derived['r_rpv_2']
    r_lithium = derived['r_lithium']
    r_lithium_wall = derived['r_lithium_wall']
    height = derived['z_top'] - derived['z_bottom']

    region_volumes = {
        'outer_tank': np.pi * (r_outer_tank**2 - r_core**2) * height,
        'rpv_inner': np.pi * (r_rpv_1**2 - r_outer_tank**2) * height,
        'rpv_outer': np.pi * (r_rpv_2**2 - r_rpv_1**2) * height,
        'lithium_wall': np.pi * (r_lithium_wall**2 - r_lithium**2) * height
    }

    # Define regions to process
    regions = ['outer_tank', 'rpv_inner', 'rpv_outer', 'lithium_wall']

    print("\n" + "="*70)
    print("Processing Regions")
    print("="*70)

    # Store results for summary
    results = {}

    # Process each region
    for region in regions:
        print(f"\nProcessing: {region}")
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
        conservation_ratio = np.sum(ukaea_flux) / np.sum(flux_values)
        print(f"  Flux conservation: {conservation_ratio:.4f}")

        # Store results
        results[region] = {
            'flux': ukaea_flux,
            'total': np.sum(ukaea_flux),
            'max': np.max(ukaea_flux),
            'volume': cell_volume
        }

        # Write output file in FISPACT-II format
        output_file = output_path / f'{region}_flux_ukaea1102.txt'
        write_fispact_format(output_file, ukaea_flux, ukaea_energies, region, cell_volume)
        print(f"  Output saved to: {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"\nProcessed {len(results)} regions:")
    print(f"{'Region':<20} {'Total Flux [n/cm²/s]':<20} {'Peak Flux [n/cm²/s]':<20}")
    print("-" * 60)
    for region, data in results.items():
        print(f"{region:<20} {data['total']:<20.4e} {data['max']:<20.4e}")

    print(f"\nAll files saved to: {output_path}/")
    print("\nConversion complete!")


def write_fispact_format(filename, flux_values, energies, region_name, volume):
    with open(filename, 'w') as f:
        # Write header comments
        f.write(f"# UKAEA-1102 flux spectrum for {region_name}\n")
        f.write(f"# Number of groups: {len(flux_values)}\n")
        f.write(f"# Energy range: {energies[-1]:.4e} to {energies[0]:.4e} eV\n")
        f.write(f"# Cell volume: {volume:.4e} cm³\n")
        f.write(f"# Units: n/cm²/s\n")
        f.write(f"# Total flux: {np.sum(flux_values):.4e} n/cm²/s\n")
        f.write("#\n")

        # Write flux values, 6 per line in the exact format requested
        for i in range(0, len(flux_values), 6):
            line_values = flux_values[i:min(i+6, len(flux_values))]
            # Format each value as X.XXXXe±YY with space separation
            formatted_values = []
            for val in line_values:
                if val == 0:
                    formatted_values.append('0.0000e+00')
                else:
                    # Format with 4 decimal places in mantissa
                    exp_str = f'{val:.4e}'
                    formatted_values.append(exp_str)

            line = ' '.join(formatted_values)
            f.write(line + ' \n')  # Add space before newline as in example


if __name__ == '__main__':
    # Check if we're in the right directory
    if not Path('simulation_raw/statepoint.250.h5').exists():
        print("ERROR: Cannot find statepoint file!")
        print("Make sure you run this script from the directory containing 'simulation_raw/'")
        sys.exit(1)

    # Run the conversion
    extract_and_convert_tallies()
