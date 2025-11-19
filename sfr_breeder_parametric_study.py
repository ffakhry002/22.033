"""
SFR Tritium Breeder Parametric Study

Cycles through all available breeder materials and tests them at:
1. Center position (Ring 17)
2. Ring 7, position 30 (left side outer position)

Calculates tritium bred per year for each configuration.
"""

import openmc
import numpy as np
from pathlib import Path
import shutil
from inputs import inputs, get_derived_dimensions
from materials import make_materials
from geometry import create_SFR_core
from tallies import create_all_tallies

# List of all available breeder materials
BREEDER_MATERIALS = [
    'natural_lithium',
    'enriched_lithium',
    'natural_flibe',
    'enriched_flibe',
    'natural_pbli',
    'enriched_pbli',
    'double_enriched_pbli'
]

# Locations to test
LOCATIONS = ['center', 'ring7']

def run_single_case(mat_dict, breeder_material, location, sim_dir):
    """Run a single simulation case with specified breeder material and location."""

    print(f"\n{'='*80}")
    print(f"Running: {breeder_material} at {location}")
    print(f"{'='*80}")

    # Create geometry with specified breeder and location
    geometry, surfaces_dict = create_SFR_core(mat_dict, location, breeder_material)

    # Create settings
    settings = openmc.Settings()
    settings.batches = inputs['batches']
    settings.inactive = inputs['inactive']
    settings.survival_biasing = False
    settings.particles = inputs['particles']
    settings.max_particle_events = inputs['max_particle_events']
    settings.run_mode = 'eigenvalue'

    # Source distribution
    derived = get_derived_dimensions()
    core_edge = inputs['sfr_core_edge']
    source_dist = openmc.stats.Box(
        [-core_edge, -core_edge, -inputs['sfr_axial_height']],
        [core_edge, core_edge, inputs['sfr_axial_height']]
    )

    # Entropy mesh
    entropy_mesh = openmc.RegularMesh()
    entropy_mesh.lower_left = [-core_edge, -core_edge, -inputs['sfr_axial_height']]
    entropy_mesh.upper_right = [core_edge, core_edge, inputs['sfr_axial_height']]
    entropy_mesh.dimension = inputs['entropy_mesh_dimension']

    source = openmc.IndependentSource()
    source.space = source_dist
    source.constraints = {'fissionable': True}
    source.strength = 1.0
    settings.source = source
    settings.entropy_mesh = entropy_mesh
    settings.temperature = {'method': 'interpolation'}

    # Create tallies
    tallies = create_all_tallies(geometry, surfaces_dict)

    # Create materials collection
    materials = openmc.Materials([mat for mat in mat_dict.values()])

    # Export to specific directory
    case_dir = sim_dir / f"{breeder_material}_{location}"
    case_dir.mkdir(exist_ok=True, parents=True)

    materials.export_to_xml(str(case_dir / 'materials.xml'))
    geometry.export_to_xml(str(case_dir / 'geometry.xml'))
    settings.export_to_xml(str(case_dir / 'settings.xml'))
    tallies.export_to_xml(str(case_dir / 'tallies.xml'))

    # Run OpenMC
    openmc.run(cwd=str(case_dir))

    # Extract results
    sp = openmc.StatePoint(str(case_dir / f'statepoint.{inputs["batches"]}.h5'))

    # Get k-eff
    keff = sp.keff.nominal_value
    keff_std = sp.keff.std_dev

    # Calculate tritium production rate
    try:
        # Get Li-6 (n,t) tally - use correct name from tallies_sfr.py
        tritium_tally = sp.get_tally(name='sfr_tritium_production')
        tritium_rate_per_source = tritium_tally.mean.flatten()[0]
        tritium_std_per_source = tritium_tally.std_dev.flatten()[0]

        # Calculate normalization factor (using k-eff = 1.2 as per user request)
        kappa_fission = 200.0  # MeV per fission
        keff_for_norm = 1.2  # Fixed k-eff for normalization

        # Calculate nu from tallies
        nu_fission_tally = sp.get_tally(name='nu-fission')
        fission_tally = sp.get_tally(name='fission')
        nu = nu_fission_tally.mean.flatten()[0] / fission_tally.mean.flatten()[0]

        # Normalization factor (neutrons/second)
        power_mw = inputs['core_power']
        C = power_mw * 6.2415e18 * nu / (kappa_fission * keff_for_norm)

        # Tritium production rate (reactions/s)
        tritium_per_second = tritium_rate_per_source * C

        # Convert to tritium bred per year
        seconds_per_year = 365.25 * 24 * 3600
        tritium_per_year = tritium_per_second * seconds_per_year

        # Convert to grams (1 mole tritium = 3 grams, Avogadro's number)
        avogadro = 6.022e23
        tritium_grams_per_year = (tritium_per_year / avogadro) * 3.0

        # Also calculate as kg/year
        tritium_kg_per_year = tritium_grams_per_year / 1000.0

    except Exception as e:
        print(f"  Warning: Could not calculate tritium production: {e}")
        tritium_kg_per_year = 0.0
        tritium_grams_per_year = 0.0

    results = {
        'material': breeder_material,
        'location': location,
        'keff': keff,
        'keff_std': keff_std,
        'tritium_kg_per_year': tritium_kg_per_year,
        'tritium_g_per_year': tritium_grams_per_year,
    }

    print(f"\n  Results:")
    print(f"    k-eff: {keff:.5f} ± {keff_std:.5f}")
    print(f"    Tritium: {tritium_kg_per_year:.6f} kg/year ({tritium_grams_per_year:.3f} g/year)")

    return results


def main():
    """Run parametric study for all breeder materials and locations."""

    print("\n" + "="*80)
    print("SFR TRITIUM BREEDER PARAMETRIC STUDY")
    print("="*80)
    print(f"\nTesting {len(BREEDER_MATERIALS)} breeder materials at {len(LOCATIONS)} locations")
    print(f"Total simulations: {len(BREEDER_MATERIALS) * len(LOCATIONS)}")
    print(f"\nBreeder materials: {', '.join(BREEDER_MATERIALS)}")
    print(f"Locations: {', '.join(LOCATIONS)}")

    # Check if we're running SFR
    if inputs['assembly_type'] != 'sodium':
        print(f"\nError: assembly_type = '{inputs['assembly_type']}', expected 'sodium'")
        print("This script only runs for SFR configurations.")
        return

    # Create base directory for results
    base_dir = Path('sfr_parametric_study')
    base_dir.mkdir(exist_ok=True)

    # Create materials once
    print("\nCreating materials...")
    mat_dict = make_materials()

    # Storage for all results
    all_results = []

    # Run all cases
    for breeder_material in BREEDER_MATERIALS:
        for location in LOCATIONS:
            try:
                results = run_single_case(mat_dict, breeder_material, location, base_dir)
                all_results.append(results)
            except Exception as e:
                print(f"\n  ERROR in {breeder_material} at {location}: {e}")
                import traceback
                traceback.print_exc()

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: TRITIUM PRODUCTION RESULTS")
    print("="*80)
    print(f"\n{'Material':<25} {'Location':<10} {'k-eff':<12} {'Tritium (kg/yr)':<18} {'Tritium (g/yr)':<15}")
    print("-"*80)

    for res in all_results:
        print(f"{res['material']:<25} {res['location']:<10} "
              f"{res['keff']:.5f} ± {res['keff_std']:.5f}  "
              f"{res['tritium_kg_per_year']:>10.6f}        "
              f"{res['tritium_g_per_year']:>10.3f}")

    # Find best configurations
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)

    # Best for center
    center_results = [r for r in all_results if r['location'] == 'center']
    if center_results:
        best_center = max(center_results, key=lambda x: x['tritium_kg_per_year'])
        print(f"\nBest at CENTER:")
        print(f"  Material: {best_center['material']}")
        print(f"  k-eff: {best_center['keff']:.5f}")
        print(f"  Tritium: {best_center['tritium_kg_per_year']:.6f} kg/year")

    # Best for ring7
    ring7_results = [r for r in all_results if r['location'] == 'ring7']
    if ring7_results:
        best_ring7 = max(ring7_results, key=lambda x: x['tritium_kg_per_year'])
        print(f"\nBest at RING 7:")
        print(f"  Material: {best_ring7['material']}")
        print(f"  k-eff: {best_ring7['keff']:.5f}")
        print(f"  Tritium: {best_ring7['tritium_kg_per_year']:.6f} kg/year")

    # Best overall
    if all_results:
        best_overall = max(all_results, key=lambda x: x['tritium_kg_per_year'])
        print(f"\nBEST OVERALL:")
        print(f"  Material: {best_overall['material']}")
        print(f"  Location: {best_overall['location']}")
        print(f"  k-eff: {best_overall['keff']:.5f}")
        print(f"  Tritium: {best_overall['tritium_kg_per_year']:.6f} kg/year")

    # Save results to file
    results_file = base_dir / 'parametric_study_results.txt'
    with open(results_file, 'w') as f:
        f.write("SFR TRITIUM BREEDER PARAMETRIC STUDY RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Material':<25} {'Location':<10} {'k-eff':<12} {'Tritium (kg/yr)':<18} {'Tritium (g/yr)':<15}\n")
        f.write("-"*80 + "\n")
        for res in all_results:
            f.write(f"{res['material']:<25} {res['location']:<10} "
                   f"{res['keff']:.5f} ± {res['keff_std']:.5f}  "
                   f"{res['tritium_kg_per_year']:>10.6f}        "
                   f"{res['tritium_g_per_year']:>10.3f}\n")

    print(f"\nResults saved to: {results_file}")
    print("\nParametric study complete!")


if __name__ == '__main__':
    main()
