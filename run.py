"""
Run script for PWR Fusion Breeder Reactor simulation
Modified to handle surface tallies
"""
import openmc
import numpy as np
import os
from pathlib import Path
from materials import make_materials
from geometry import create_core
from tallies import create_all_tallies, calc_norm_factor
from inputs import inputs, get_derived_dimensions


def print_simulation_summary(sim_dir):
    """Print summary of simulation results including tritium production.

    Parameters
    ----------
    sim_dir : Path
        Directory containing the statepoint files
    """
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)

    # Get power and breeder material from inputs
    power_mw = inputs['core_power']
    breeder_material = inputs['breeder_material']

    print(f"\nReactor Configuration:")
    print(f"  Power: {power_mw} MW")
    print(f"  Breeder Material: {breeder_material}")

    # Find the latest statepoint file
    statepoint_files = list(sim_dir.glob('statepoint.*.h5'))
    if not statepoint_files:
        print("\n  Warning: No statepoint files found. Cannot calculate tritium production.")
        print("="*70)
        return

    # Get the last statepoint (highest batch number)
    latest_sp_file = max(statepoint_files, key=lambda f: int(f.stem.split('.')[-1]))

    try:
        # Load statepoint
        sp = openmc.StatePoint(str(latest_sp_file))

        # Calculate normalization factor
        norm_factor = calc_norm_factor(power_mw, sp)

        # Get tritium breeding tally
        tbr_tally = sp.get_tally(name='tritium_breeding_ratio')
        tritium_per_source = tbr_tally.mean[0, 0, 0]  # T atoms/source neutron
        tritium_per_source_std = tbr_tally.std_dev[0, 0, 0]

        # Calculate absolute tritium production rate
        t_production = tritium_per_source * norm_factor  # T atoms/s
        t_production_std = tritium_per_source_std * norm_factor

        # Convert to grams (tritium-3 has atomic mass of 3.0 g/mol)
        atoms_to_grams = 3.0 / 6.022e23  # g/atom

        # Calculate rates
        g_per_s = t_production * atoms_to_grams
        g_per_year = g_per_s * 86400 * 365.25

        # Calculate uncertainties
        g_per_s_std = t_production_std * atoms_to_grams
        g_per_year_std = g_per_s_std * 86400 * 365.25

        print(f"\nTritium Production:")
        print(f"  Per Second: {g_per_s:.6e} ± {g_per_s_std:.6e} g/s")
        print(f"  Per Year:   {g_per_year:.6e} ± {g_per_year_std:.6e} g/year")

        # Also print TBR for reference
        print(f"\nTritium Breeding Ratio (TBR):")
        print(f"  {tritium_per_source:.6f} ± {tritium_per_source_std:.6f} T atoms/source neutron")

    except Exception as e:
        print(f"\n  Warning: Could not calculate tritium production: {e}")

    print("="*70)


def run_simulation():
    """Set up and run the OpenMC simulation."""

    # Create simulation_raw directory if it doesn't exist
    sim_dir = Path('simulation_raw')
    sim_dir.mkdir(exist_ok=True)
    print(f"\nSaving simulation files to '{sim_dir}' directory...")

    # Create materials
    mat_dict = make_materials()
    materials = openmc.Materials([mat for mat in mat_dict.values()])

    # Create geometry (now returns both geometry and surfaces_dict)
    geometry, surfaces_dict = create_core(mat_dict)
    print(f"\nGeometry created with surfaces: {list(surfaces_dict.keys())}")

    # Create settings
    settings = openmc.Settings()
    settings.batches = inputs['batches']
    settings.inactive = inputs['inactive']
    settings.particles = inputs['particles']
    settings.max_particle_events = inputs['max_particle_events']
    settings.run_mode = 'eigenvalue'

    # Source distribution
    derived = get_derived_dimensions()
    source_dist = openmc.stats.Box(
        [-inputs['r_core'], -inputs['r_core'], derived['z_fuel_bottom']],
        [inputs['r_core'], inputs['r_core'], derived['z_fuel_top']],
        only_fissionable=True
    )
    source = openmc.IndependentSource()
    source.space = source_dist
    settings.source = source

    # Entropy mesh for convergence
    entropy_mesh = openmc.RegularMesh()
    entropy_mesh.lower_left = [-inputs['r_core'], -inputs['r_core'], derived['z_fuel_bottom']]
    entropy_mesh.upper_right = [inputs['r_core'], inputs['r_core'], derived['z_fuel_top']]
    entropy_mesh.dimension = inputs['entropy_mesh_dimension']
    settings.entropy_mesh = entropy_mesh
    settings.temperature = {'method': 'interpolation'}

    # Create tallies (now passing surfaces_dict)
    tallies = create_all_tallies(geometry, surfaces_dict)

    # Export XML files to simulation_raw directory
    materials.export_to_xml(str(sim_dir / 'materials.xml'))
    geometry.export_to_xml(str(sim_dir / 'geometry.xml'))
    settings.export_to_xml(str(sim_dir / 'settings.xml'))
    tallies.export_to_xml(str(sim_dir / 'tallies.xml'))

    # Run OpenMC in the simulation_raw directory
    print("\n" + "="*70)
    print("Running OpenMC Simulation...")
    print("="*70 + "\n")

    # Save current directory and change to simulation directory
    original_dir = os.getcwd()
    os.chdir(sim_dir)

    try:
        openmc.run(geometry_debug=True)
    finally:
        # Return to original directory
        os.chdir(original_dir)

    print("\n" + "="*70)
    print("Simulation Complete!")
    print("="*70)

    # Print summary of results
    print_simulation_summary(sim_dir)


if __name__ == '__main__':
    run_simulation()
