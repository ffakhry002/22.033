"""
Tallies for PWR Fusion Breeder Reactor Analysis
Modified for tritium breeder assembly tallies
"""
import openmc
import numpy as np
from inputs import inputs, get_derived_dimensions

def calc_norm_factor(power_mw, sp):
    """Calculate the normalization factor based on reactor parameters.

    This converts OpenMC's per-source-particle values to absolute values (n/s)
    based on the actual reactor power and fixed values.

    Parameters
    ----------
    power_mw : float
        Reactor power in MW
    sp : openmc.StatePoint
        StatePoint file containing the tally results

    Returns
    -------
    float
        Normalization factor (neutrons/second)

    Notes
    -----
    Uses the equation:
    C = P * 6.2415e18 * nu / (kappa * keff)
    where:
    - P is reactor power [MW]
    - 6.2415e18 is MeV/MW-s conversion
    - kappa is energy per fission [MeV] (fixed at 200 MeV)
    - keff is from the simulation
    - nu is calculated from tallies as (nu-fission)/(fission)
    """
    # Use fixed value for energy per fission
    kappa_fission = 200.0  # MeV per fission
    keff = sp.keff.nominal_value

    # Calculate nu from tallies
    nu_fission_tally = sp.get_tally(name='nu-fission')
    fission_tally = sp.get_tally(name='fission')

    nu = (nu_fission_tally.mean.flatten()[0] /
          fission_tally.mean.flatten()[0])

    # Calculate normalization factor
    C = power_mw * 6.2415e18 * nu / (kappa_fission * keff)

    return float(C)


def create_normalization_tallies():
    """Create tallies needed for flux normalization."""
    tallies = openmc.Tallies()

    # Create tally for nu-fission
    nu_fission_tally = openmc.Tally(name='nu-fission')
    nu_fission_tally.scores = ['nu-fission']
    tallies.append(nu_fission_tally)

    # Create tally for fission
    fission_tally = openmc.Tally(name='fission')
    fission_tally.scores = ['fission']
    tallies.append(fission_tally)

    print("\nCreated normalization tallies:")
    print("  - nu-fission")
    print("  - fission")

    return tallies


def create_tritium_breeder_surface_tallies(geometry, surfaces_dict, energy_filters):
    """Create surface tallies for tritium breeder assembly.

    Creates:
    1. Surface tally on calandria outer surface (neutrons from moderator entering)
    2. Surface tally on pressure tube inner surface (neutrons from pressure tube entering breeder)

    Parameters
    ----------
    energy_filters : dict
        Pre-created energy filters to avoid ID duplication
    """
    tallies = openmc.Tallies()

    # Get tritium breeder info
    tritium_info = surfaces_dict.get('tritium_info')
    if tritium_info is None:
        print("\nNo tritium breeder assemblies found in geometry")
        return tallies

    # Get cells and surfaces
    cells_dict = tritium_info['cells_dict']
    surf_dict = tritium_info['surfaces_dict']

    # Get the specific cells we need
    moderator_cell = cells_dict['moderator']
    pressure_tube_cell = cells_dict['pressure_tube']

    # Get surfaces
    calandria_outer = surf_dict['calandria_outer']
    pt_inner = surf_dict['pt_inner']

    # Use pre-created energy filters
    thermal_filter = energy_filters['thermal']
    epithermal_filter = energy_filters['epithermal']
    fast_filter = energy_filters['fast']
    log_1001_filter = energy_filters['log_1001']

    # 1. Calandria outer surface - neutrons from moderator entering
    # We use CellFromFilter to get only neutrons coming FROM the moderator cell
    calandria_surf_filter = openmc.SurfaceFilter(calandria_outer)
    moderator_from_filter = openmc.CellFromFilter(moderator_cell)

    # Total current from moderator into calandria
    calandria_total = openmc.Tally(name='tritium_calandria_outer_current_total')
    calandria_total.filters = [calandria_surf_filter, moderator_from_filter]
    calandria_total.scores = ['current']
    tallies.append(calandria_total)

    # Thermal current
    calandria_thermal = openmc.Tally(name='tritium_calandria_outer_current_thermal')
    calandria_thermal.filters = [calandria_surf_filter, moderator_from_filter, thermal_filter]
    calandria_thermal.scores = ['current']
    tallies.append(calandria_thermal)

    # Epithermal current
    calandria_epithermal = openmc.Tally(name='tritium_calandria_outer_current_epithermal')
    calandria_epithermal.filters = [calandria_surf_filter, moderator_from_filter, epithermal_filter]
    calandria_epithermal.scores = ['current']
    tallies.append(calandria_epithermal)

    # Fast current
    calandria_fast = openmc.Tally(name='tritium_calandria_outer_current_fast')
    calandria_fast.filters = [calandria_surf_filter, moderator_from_filter, fast_filter]
    calandria_fast.scores = ['current']
    tallies.append(calandria_fast)

    # LOG_1001 current spectrum
    calandria_log1001 = openmc.Tally(name='tritium_calandria_outer_current_log1001')
    calandria_log1001.filters = [calandria_surf_filter, moderator_from_filter, log_1001_filter]
    calandria_log1001.scores = ['current']
    tallies.append(calandria_log1001)

    # 2. Pressure tube inner surface - neutrons from pressure tube entering breeder
    # We use CellFromFilter to get only neutrons coming FROM the pressure tube cell
    pt_surf_filter = openmc.SurfaceFilter(pt_inner)
    pt_from_filter = openmc.CellFromFilter(pressure_tube_cell)

    # Total current from pressure tube into breeder
    pt_total = openmc.Tally(name='tritium_pt_inner_current_total')
    pt_total.filters = [pt_surf_filter, pt_from_filter]
    pt_total.scores = ['current']
    tallies.append(pt_total)

    # Thermal current
    pt_thermal = openmc.Tally(name='tritium_pt_inner_current_thermal')
    pt_thermal.filters = [pt_surf_filter, pt_from_filter, thermal_filter]
    pt_thermal.scores = ['current']
    tallies.append(pt_thermal)

    # Epithermal current
    pt_epithermal = openmc.Tally(name='tritium_pt_inner_current_epithermal')
    pt_epithermal.filters = [pt_surf_filter, pt_from_filter, epithermal_filter]
    pt_epithermal.scores = ['current']
    tallies.append(pt_epithermal)

    # Fast current
    pt_fast = openmc.Tally(name='tritium_pt_inner_current_fast')
    pt_fast.filters = [pt_surf_filter, pt_from_filter, fast_filter]
    pt_fast.scores = ['current']
    tallies.append(pt_fast)

    # LOG_1001 current spectrum
    pt_log1001 = openmc.Tally(name='tritium_pt_inner_current_log1001')
    pt_log1001.filters = [pt_surf_filter, pt_from_filter, log_1001_filter]
    pt_log1001.scores = ['current']
    tallies.append(pt_log1001)

    print("\nCreated tritium breeder surface tallies:")
    print("  - Calandria outer surface: Total, Thermal, Epithermal, Fast, LOG_1001 (from moderator)")
    print("  - Pressure tube inner surface: Total, Thermal, Epithermal, Fast, LOG_1001 (from pressure tube)")
    print(f"  Total: {len(tallies)} surface tallies")

    return tallies


def create_tritium_breeding_tally(geometry, surfaces_dict):
    """Create tritium breeding tally for the tritium breeder assembly.

    This tallies tritium production in the breeding region (breeder material + coolant).
    """
    tallies = openmc.Tallies()

    # Get tritium breeder info
    tritium_info = surfaces_dict.get('tritium_info')
    if tritium_info is None:
        print("\nNo tritium breeder assemblies found in geometry")
        return tallies

    # Find all cells with names matching the tritium breeder pattern
    # We want: tritium_breeder_material, tritium_breeder_coolant_inner, tritium_breeder_coolant_wall
    tritium_cells = []
    for cell in geometry.get_all_cells().values():
        if cell.name in ['tritium_breeder_material', 'tritium_breeder_coolant_inner', 'tritium_breeder_coolant_wall']:
            tritium_cells.append(cell)

    if not tritium_cells:
        print("\nWarning: No tritium breeder cells found in geometry")
        return tallies

    # Create cell filter for all tritium breeder cells (breeder + coolant)
    # Assign unique ID to avoid conflicts
    tritium_filter = openmc.CellFilter(tritium_cells)

    # Create tritium production tally
    # Using (n,Xt) which captures all tritium production reactions:
    # - Li-6 (n,t) He-4
    # - Li-7 (n,n't) He-4
    tbr_tally = openmc.Tally(name='tritium_breeder_production')
    tbr_tally.filters = [tritium_filter]
    tbr_tally.scores = ['(n,Xt)']  # Total tritium production
    tallies.append(tbr_tally)

    # Also create separate tallies for Li-6 and Li-7 contributions
    # Li-6 contribution
    tbr_li6_tally = openmc.Tally(name='tritium_breeder_production_li6')
    tbr_li6_tally.filters = [tritium_filter]
    tbr_li6_tally.scores = ['(n,Xt)']
    tbr_li6_tally.nuclides = ['Li6']
    tallies.append(tbr_li6_tally)

    # Li-7 contribution
    tbr_li7_tally = openmc.Tally(name='tritium_breeder_production_li7')
    tbr_li7_tally.filters = [tritium_filter]
    tbr_li7_tally.scores = ['(n,Xt)']
    tbr_li7_tally.nuclides = ['Li7']
    tallies.append(tbr_li7_tally)

    print("\nCreated tritium breeder production tallies:")
    print(f"  - Total tritium production (in {len(tritium_cells)} cells)")
    print("  - Li-6 contribution")
    print("  - Li-7 contribution")

    return tallies


def create_tritium_breeder_flux_tallies(geometry, surfaces_dict, energy_filters):
    """Create flux tallies for the tritium breeder assembly.

    This tallies neutron flux in the breeding region with energy discretization.

    Parameters
    ----------
    energy_filters : dict
        Pre-created energy filters to avoid ID duplication
    """
    tallies = openmc.Tallies()

    # Get tritium breeder info
    tritium_info = surfaces_dict.get('tritium_info')
    if tritium_info is None:
        print("\nNo tritium breeder assemblies found in geometry")
        return tallies

    # Find all cells with names matching the tritium breeder pattern
    tritium_cells = []
    for cell in geometry.get_all_cells().values():
        if cell.name in ['tritium_breeder_material', 'tritium_breeder_coolant_inner', 'tritium_breeder_coolant_wall']:
            tritium_cells.append(cell)

    if not tritium_cells:
        print("\nWarning: No tritium breeder cells found in geometry")
        return tallies

    # Create cell filter for all tritium breeder cells
    tritium_filter = openmc.CellFilter(tritium_cells)

    # Use pre-created energy filters
    thermal_filter = energy_filters['thermal']
    epithermal_filter = energy_filters['epithermal']
    fast_filter = energy_filters['fast']
    log_1001_filter = energy_filters['log_1001']

    # Total flux
    total_flux = openmc.Tally(name='tritium_breeder_flux_total')
    total_flux.filters = [tritium_filter]
    total_flux.scores = ['flux']
    tallies.append(total_flux)

    # Thermal flux
    thermal_flux = openmc.Tally(name='tritium_breeder_flux_thermal')
    thermal_flux.filters = [tritium_filter, thermal_filter]
    thermal_flux.scores = ['flux']
    tallies.append(thermal_flux)

    # Epithermal flux
    epithermal_flux = openmc.Tally(name='tritium_breeder_flux_epithermal')
    epithermal_flux.filters = [tritium_filter, epithermal_filter]
    epithermal_flux.scores = ['flux']
    tallies.append(epithermal_flux)

    # Fast flux
    fast_flux = openmc.Tally(name='tritium_breeder_flux_fast')
    fast_flux.filters = [tritium_filter, fast_filter]
    fast_flux.scores = ['flux']
    tallies.append(fast_flux)

    # LOG_1001 flux spectrum
    log1001_flux = openmc.Tally(name='tritium_breeder_flux_log1001')
    log1001_flux.filters = [tritium_filter, log_1001_filter]
    log1001_flux.scores = ['flux']
    tallies.append(log1001_flux)

    print("\nCreated tritium breeder flux tallies:")
    print(f"  - Total, Thermal, Epithermal, Fast flux (in {len(tritium_cells)} cells)")
    print(f"  - LOG_1001 energy spectrum ({len(inputs['log_1001_bins'])-1} bins)")

    return tallies


def create_core_mesh_tallies(energy_filters):
    """Create full core mesh tallies for radial flux profiles.

    Parameters
    ----------
    energy_filters : dict
        Pre-created energy filters to avoid ID duplication
    """
    tallies = openmc.Tallies()
    derived = get_derived_dimensions()

    # Create rectangular mesh covering entire reactor geometry
    # Use reasonable size to avoid memory issues (400x400x50 instead of 1000x1000x200)
    mesh = openmc.RegularMesh()
    mesh.dimension = [400, 400, 50]
    mesh.lower_left = [-derived['r_lithium_wall'], -derived['r_lithium_wall'], derived['z_bottom']]
    mesh.upper_right = [derived['r_lithium_wall'], derived['r_lithium_wall'], derived['z_top']]

    mesh_filter = openmc.MeshFilter(mesh)

    # Use pre-created energy filters
    thermal_filter = energy_filters['thermal']
    epithermal_filter = energy_filters['epithermal']
    fast_filter = energy_filters['fast']

    # 1. Total flux (all energies)
    total_flux_tally = openmc.Tally(name='core_mesh_total_flux')
    total_flux_tally.filters = [mesh_filter]
    total_flux_tally.scores = ['flux']
    tallies.append(total_flux_tally)

    # 2. Thermal flux
    thermal_flux_tally = openmc.Tally(name='core_mesh_thermal_flux')
    thermal_flux_tally.filters = [mesh_filter, thermal_filter]
    thermal_flux_tally.scores = ['flux']
    tallies.append(thermal_flux_tally)

    # 3. Epithermal flux
    epithermal_flux_tally = openmc.Tally(name='core_mesh_epithermal_flux')
    epithermal_flux_tally.filters = [mesh_filter, epithermal_filter]
    epithermal_flux_tally.scores = ['flux']
    tallies.append(epithermal_flux_tally)

    # 4. Fast flux
    fast_flux_tally = openmc.Tally(name='core_mesh_fast_flux')
    fast_flux_tally.filters = [mesh_filter, fast_filter]
    fast_flux_tally.scores = ['flux']
    tallies.append(fast_flux_tally)

    print("\nCreated core mesh tallies:")
    print("  - Total, Thermal, Epithermal, Fast flux")
    print(f"  Mesh: 400 x 400 x 50")

    return tallies


def create_tritium_assembly_mesh_tally(energy_filters):
    """Create 3x3 assembly mesh centered on tritium breeder (50x50x1).

    Parameters
    ----------
    energy_filters : dict
        Pre-created energy filters to avoid ID duplication
    """
    tallies = openmc.Tallies()
    derived = get_derived_dimensions()

    # Get assembly width
    assembly_width = derived['assembly_width']  # CANDU assembly pitch

    # Find T_1 position in lattice
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

    # Calculate T_1 center position (lattice is centered at origin)
    if t1_row is not None and t1_col is not None:
        # Position relative to center of lattice
        x_center = (t1_col - n_cols/2 + 0.5) * assembly_width
        y_center = (t1_row - n_rows/2 + 0.5) * assembly_width
    else:
        # Default to origin if T_1 not found
        x_center, y_center = 0.0, 0.0
        print("  Warning: T_1 not found in lattice, centering mesh at origin")

    # 3x3 assemblies centered on T_1
    mesh_width = 3 * assembly_width

    # Create mesh: 50x50x1
    mesh = openmc.RegularMesh()
    mesh.dimension = [50, 50, 1]
    mesh.lower_left = [x_center - mesh_width/2, y_center - mesh_width/2, derived['z_fuel_bottom']]
    mesh.upper_right = [x_center + mesh_width/2, y_center + mesh_width/2, derived['z_fuel_top']]

    mesh_filter = openmc.MeshFilter(mesh)

    # Use pre-created energy filters
    thermal_filter = energy_filters['thermal']
    epithermal_filter = energy_filters['epithermal']
    fast_filter = energy_filters['fast']

    # Total flux
    total_tally = openmc.Tally(name='tritium_assembly_mesh_total')
    total_tally.filters = [mesh_filter]
    total_tally.scores = ['flux']
    tallies.append(total_tally)

    # Thermal flux
    thermal_tally = openmc.Tally(name='tritium_assembly_mesh_thermal')
    thermal_tally.filters = [mesh_filter, thermal_filter]
    thermal_tally.scores = ['flux']
    tallies.append(thermal_tally)

    # Epithermal flux
    epithermal_tally = openmc.Tally(name='tritium_assembly_mesh_epithermal')
    epithermal_tally.filters = [mesh_filter, epithermal_filter]
    epithermal_tally.scores = ['flux']
    tallies.append(epithermal_tally)

    # Fast flux
    fast_tally = openmc.Tally(name='tritium_assembly_mesh_fast')
    fast_tally.filters = [mesh_filter, fast_filter]
    fast_tally.scores = ['flux']
    tallies.append(fast_tally)

    print("\nCreated tritium assembly mesh tallies:")
    print(f"  - 3x3 assembly region ({mesh_width:.1f} cm x {mesh_width:.1f} cm)")
    print(f"  - Centered at T_1 position: ({x_center:.1f}, {y_center:.1f}) cm")
    print(f"  - Mesh: 50 x 50 x 1")
    print(f"  - Total, Thermal, Epithermal, Fast flux")

    return tallies


def create_all_tallies(geometry, surfaces_dict):
    """Create all tallies for the simulation.

    Parameters
    ----------
    geometry : openmc.Geometry
        The reactor geometry
    surfaces_dict : dict
        Dictionary of surface objects from geometry creation
    """
    all_tallies = openmc.Tallies()

    # Create energy filters ONCE to avoid ID duplication
    energy_filters = {
        'thermal': openmc.EnergyFilter([0.0, inputs['thermal_cutoff']]),
        'epithermal': openmc.EnergyFilter([inputs['thermal_cutoff'], inputs['epithermal_cutoff']]),
        'fast': openmc.EnergyFilter([inputs['epithermal_cutoff'], inputs['fast_cutoff']]),
        'log_1001': openmc.EnergyFilter(inputs['log_1001_bins'])
    }

    # 1. Normalization tallies (for flux normalization)
    norm_tallies = create_normalization_tallies()
    all_tallies.extend(norm_tallies)

    # 2. Tritium breeder surface tallies
    surface_tallies = create_tritium_breeder_surface_tallies(geometry, surfaces_dict, energy_filters)
    all_tallies.extend(surface_tallies)

    # 3. Tritium breeding tallies
    tbr_tallies = create_tritium_breeding_tally(geometry, surfaces_dict)
    all_tallies.extend(tbr_tallies)

    # 4. Tritium breeder flux tallies
    flux_tallies = create_tritium_breeder_flux_tallies(geometry, surfaces_dict, energy_filters)
    all_tallies.extend(flux_tallies)

    # 5. Core mesh tallies (for radial plots)
    core_mesh_tallies = create_core_mesh_tallies(energy_filters)
    all_tallies.extend(core_mesh_tallies)

    # 6. Tritium assembly mesh tally (3x3 assembly heatmap)
    assembly_mesh_tallies = create_tritium_assembly_mesh_tally(energy_filters)
    all_tallies.extend(assembly_mesh_tallies)

    print("\n" + "="*60)
    print(f"Total tallies created: {len(all_tallies)}")
    print("="*60)

    return all_tallies


if __name__ == '__main__':
    try:
        from materials import make_materials
        from geometry import create_core

        mat_dict = make_materials()
        geometry, surfaces_dict = create_core(mat_dict)

        tallies = create_all_tallies(geometry, surfaces_dict)

        tallies.export_to_xml()

        print("\nTallies exported successfully!")

    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure materials.py and geometry.py are in the same directory")
    except Exception as e:
        print(f"Error creating tallies: {e}")
        import traceback
        traceback.print_exc()
