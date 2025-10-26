"""
Run script for PWR Fusion Breeder Reactor simulation
"""
import openmc
import numpy as np
import os
from pathlib import Path
from materials import make_materials
from geometry import create_core
from tallies import create_all_tallies
from inputs import inputs, get_derived_dimensions


def run_simulation():
    """Set up and run the OpenMC simulation."""

    # Create simulation_raw directory if it doesn't exist
    sim_dir = Path('simulation_raw')
    sim_dir.mkdir(exist_ok=True)
    print(f"\nSaving simulation files to '{sim_dir}' directory...")

    # Create materials
    mat_dict = make_materials()
    materials = openmc.Materials([mat for mat in mat_dict.values()])

    # Create geometry
    geometry = create_core(mat_dict)

    # Create settings
    settings = openmc.Settings()
    settings.batches = inputs['batches']
    settings.inactive = inputs['inactive']
    settings.particles = inputs['particles']
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

    # Create tallies
    tallies = create_all_tallies(geometry)

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
        openmc.run()
    finally:
        # Return to original directory
        os.chdir(original_dir)

    print("\n" + "="*70)
    print("Simulation Complete!")
    print("="*70)


if __name__ == '__main__':
    run_simulation()
