"""
Tallies for PWR Fusion Breeder Reactor Analysis
Modified to include surface current tallies
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


def create_full_reactor_mesh_tallies():
    """Create full reactor mesh tallies: total, thermal, epithermal, and fast flux."""
    tallies = openmc.Tallies()
    derived = get_derived_dimensions()

    # Create rectangular mesh covering entire reactor geometry
    mesh = openmc.RegularMesh()
    mesh.dimension = [inputs['n_radial_bins'], inputs['n_radial_bins'], inputs['n_axial_bins']]
    mesh.lower_left = [-derived['r_lithium_wall'], -derived['r_lithium_wall'], derived['z_bottom']]
    mesh.upper_right = [derived['r_lithium_wall'], derived['r_lithium_wall'], derived['z_top']]

    mesh_filter = openmc.MeshFilter(mesh)

    # Energy filters for different groups
    thermal_filter = openmc.EnergyFilter([0.0, inputs['thermal_cutoff']])
    epithermal_filter = openmc.EnergyFilter([inputs['thermal_cutoff'], inputs['epithermal_cutoff']])
    fast_filter = openmc.EnergyFilter([inputs['epithermal_cutoff'], inputs['fast_cutoff']])

    # 1. Total flux (all energies)
    total_flux_tally = openmc.Tally(name='mesh_total_flux')
    total_flux_tally.filters = [mesh_filter]
    total_flux_tally.scores = ['flux']
    tallies.append(total_flux_tally)

    # 2. Thermal flux
    thermal_flux_tally = openmc.Tally(name='mesh_thermal_flux')
    thermal_flux_tally.filters = [mesh_filter, thermal_filter]
    thermal_flux_tally.scores = ['flux']
    tallies.append(thermal_flux_tally)

    # 3. Epithermal flux
    epithermal_flux_tally = openmc.Tally(name='mesh_epithermal_flux')
    epithermal_flux_tally.filters = [mesh_filter, epithermal_filter]
    epithermal_flux_tally.scores = ['flux']
    tallies.append(epithermal_flux_tally)

    # 4. Fast flux
    fast_flux_tally = openmc.Tally(name='mesh_fast_flux')
    fast_flux_tally.filters = [mesh_filter, fast_filter]
    fast_flux_tally.scores = ['flux']
    tallies.append(fast_flux_tally)

    print("\nCreated full reactor mesh tallies:")
    print("  - Total flux")
    print("  - Thermal flux (0 to 0.625 eV)")
    print("  - Epithermal flux (0.625 eV to 100 keV)")
    print("  - Fast flux (100 keV to 10 MeV)")
    print(f"  Mesh: {inputs['n_radial_bins']} x {inputs['n_radial_bins']} x {inputs['n_axial_bins']}")

    return tallies


def create_cell_based_energy_tallies(geometry):
    """Create cell-based energy-discretized tallies for key regions."""
    tallies = openmc.Tallies()
    derived = get_derived_dimensions()

    # Find the required cells
    outer_tank_cell = None
    rpv_1_cell = None
    rpv_2_cell = None
    lithium_wall_cell = None

    for cell in geometry.root_universe.cells.values():
        if cell.name == 'outer_tank':
            outer_tank_cell = cell
        elif cell.name == 'rpv_layer_1':
            rpv_1_cell = cell
        elif cell.name == 'rpv_layer_2':
            rpv_2_cell = cell
        elif cell.name == 'breeder_wall':
            lithium_wall_cell = cell

    # Check that all cells were found
    missing_cells = []
    if outer_tank_cell is None:
        missing_cells.append('outer_tank')
    if rpv_1_cell is None:
        missing_cells.append('rpv_layer_1')
    if rpv_2_cell is None:
        missing_cells.append('rpv_layer_2')
    if lithium_wall_cell is None:
        missing_cells.append('breeder_wall')

    if missing_cells:
        raise ValueError(f"Could not find cells: {', '.join(missing_cells)}")

    # Create cell filters
    outer_tank_filter = openmc.CellFilter([outer_tank_cell])
    rpv_1_filter = openmc.CellFilter([rpv_1_cell])
    rpv_2_filter = openmc.CellFilter([rpv_2_cell])
    lithium_wall_filter = openmc.CellFilter([lithium_wall_cell])

    # Define energy filters
    thermal_filter = openmc.EnergyFilter([0.0, inputs['thermal_cutoff']])
    epithermal_filter = openmc.EnergyFilter([inputs['thermal_cutoff'], inputs['epithermal_cutoff']])
    fast_filter = openmc.EnergyFilter([inputs['epithermal_cutoff'], inputs['fast_cutoff']])
    log_1001_filter = openmc.EnergyFilter(inputs['log_1001_bins'])

    # Dictionary mapping cell filters to names
    regions = {
        'outer_tank': outer_tank_filter,
        'rpv_inner': rpv_1_filter,
        'rpv_outer': rpv_2_filter,
        'lithium_wall': lithium_wall_filter
    }

    # Create tallies for each region
    for region_name, cell_filter in regions.items():
        # LOG_1001 flux
        log1001_tally = openmc.Tally(name=f'{region_name}_flux_log1001')
        log1001_tally.filters = [cell_filter, log_1001_filter]
        log1001_tally.scores = ['flux']
        tallies.append(log1001_tally)

        # Thermal flux
        thermal_tally = openmc.Tally(name=f'{region_name}_flux_thermal')
        thermal_tally.filters = [cell_filter, thermal_filter]
        thermal_tally.scores = ['flux']
        tallies.append(thermal_tally)

        # Epithermal flux
        epithermal_tally = openmc.Tally(name=f'{region_name}_flux_epithermal')
        epithermal_tally.filters = [cell_filter, epithermal_filter]
        epithermal_tally.scores = ['flux']
        tallies.append(epithermal_tally)

        # Fast flux
        fast_tally = openmc.Tally(name=f'{region_name}_flux_fast')
        fast_tally.filters = [cell_filter, fast_filter]
        fast_tally.scores = ['flux']
        tallies.append(fast_tally)

    print("\nCreated cell-based energy-discretized tallies:")
    print("  Regions: Outer Tank, RPV Inner, RPV Outer, Lithium Wall")
    print("  For each region: LOG_1001, Thermal, Epithermal, Fast flux")
    print(f"  Total: {len(tallies)} tallies")

    return tallies


def create_surface_current_tallies(geometry):
    """Create surface current tallies to measure ONLY outward neutron leakage at key radii.

    Measures OUTWARD-ONLY neutron current at:
    - Core boundary (r_core) - from fuel region outward
    - Outer tank boundary (r_outer_tank) - from outer tank outward
    - RPV inner boundary (r_rpv_1) - from RPV layer 1 outward
    - RPV outer boundary (r_rpv_2) - from RPV layer 2 outward
    - Lithium blanket boundary (r_lithium) - from breeder blanket outward

    For each surface, creates tallies for:
    - Total current (log1001 energy bins)
    - Thermal current
    - Epithermal current
    - Fast current

    Note: Uses cell filters to get partial (directional) currents - only outward
    """
    tallies = openmc.Tallies()
    derived = get_derived_dimensions()

    # Define the cylindrical surfaces at key radii
    surfaces = {
        'core': openmc.ZCylinder(r=inputs['r_core'], surface_id=10001),
        'outer_tank': openmc.ZCylinder(r=derived['r_outer_tank'], surface_id=10002),
        'rpv_inner': openmc.ZCylinder(r=derived['r_rpv_1'], surface_id=10003),
        'rpv_outer': openmc.ZCylinder(r=derived['r_rpv_2'], surface_id=10004),
        'lithium': openmc.ZCylinder(r=derived['r_lithium'], surface_id=10005)
    }

    # Get cells from geometry for directional current tallies
    # We need to identify which cells are on the "inside" of each surface
    # to measure only outward current
    cell_mapping = {}
    for cell in geometry.root_universe.cells.values():
        if cell.name == 'core_region':  # Contains the fuel lattice
            cell_mapping['core'] = cell
        elif cell.name == 'outer_tank':
            cell_mapping['outer_tank'] = cell
        elif cell.name == 'rpv_layer_1':
            cell_mapping['rpv_inner'] = cell
        elif cell.name == 'rpv_layer_2':
            cell_mapping['rpv_outer'] = cell
        elif cell.name == 'breeder_blanket':
            cell_mapping['lithium'] = cell

    # Energy filters
    thermal_filter = openmc.EnergyFilter([0.0, inputs['thermal_cutoff']])
    epithermal_filter = openmc.EnergyFilter([inputs['thermal_cutoff'], inputs['epithermal_cutoff']])
    fast_filter = openmc.EnergyFilter([inputs['epithermal_cutoff'], inputs['fast_cutoff']])
    log_1001_filter = openmc.EnergyFilter(inputs['log_1001_bins'])

    # Create tallies for each surface
    for surf_name, surface in surfaces.items():
        # Create surface filter
        surf_filter = openmc.SurfaceFilter(surface)

        # Create cell from filter for outward current only
        # This specifies we only want current FROM the inner cell going OUT
        if surf_name in cell_mapping:
            cellfrom_filter = openmc.CellFromFilter(cell_mapping[surf_name])

            # LOG_1001 current (OUTWARD ONLY)
            log1001_tally = openmc.Tally(name=f'surface_{surf_name}_current_log1001')
            log1001_tally.filters = [surf_filter, cellfrom_filter, log_1001_filter]
            log1001_tally.scores = ['current']  # With cellfrom filter, this gives partial current
            tallies.append(log1001_tally)

            # Thermal current (OUTWARD ONLY)
            thermal_tally = openmc.Tally(name=f'surface_{surf_name}_current_thermal')
            thermal_tally.filters = [surf_filter, cellfrom_filter, thermal_filter]
            thermal_tally.scores = ['current']
            tallies.append(thermal_tally)

            # Epithermal current (OUTWARD ONLY)
            epithermal_tally = openmc.Tally(name=f'surface_{surf_name}_current_epithermal')
            epithermal_tally.filters = [surf_filter, cellfrom_filter, epithermal_filter]
            epithermal_tally.scores = ['current']
            tallies.append(epithermal_tally)

            # Fast current (OUTWARD ONLY)
            fast_tally = openmc.Tally(name=f'surface_{surf_name}_current_fast')
            fast_tally.filters = [surf_filter, cellfrom_filter, fast_filter]
            fast_tally.scores = ['current']
            tallies.append(fast_tally)

    print("\nCreated surface current tallies:")
    print("  Surfaces: Core, Outer Tank, RPV Inner, RPV Outer, Lithium")
    print("  For each surface: LOG_1001, Thermal, Epithermal, Fast current (OUTWARD ONLY)")
    print(f"  Total: {len(tallies)} surface tallies")

    return tallies

    print("\nCreated surface current tallies:")
    print("  Surfaces: Core, Outer Tank, RPV Inner, RPV Outer, Lithium")
    print("  For each surface: LOG_1001, Thermal, Epithermal, Fast current")
    print(f"  Total: {len(tallies)} surface tallies")

    return tallies


def create_tritium_breeding_tally(geometry):
    """Create tritium breeding ratio tallies."""
    tallies = openmc.Tallies()

    # Find the breeder blanket cell
    lithium_cell = None
    for cell in geometry.root_universe.cells.values():
        if cell.name == 'breeder_blanket':
            lithium_cell = cell
            break

    if lithium_cell is None:
        raise ValueError("Could not find 'breeder_blanket' cell in geometry")

    # Create cell filter
    lithium_filter = openmc.CellFilter([lithium_cell])

    # Create tritium production tally
    # Using (n,Xt) which captures all tritium production reactions:
    # - Li-6 (n,t) He-4
    # - Li-7 (n,n't) He-4
    tbr_tally = openmc.Tally(name='tritium_breeding_ratio')
    tbr_tally.filters = [lithium_filter]
    tbr_tally.scores = ['(n,Xt)']  # Total tritium production
    tallies.append(tbr_tally)

    # Also create separate tallies for Li-6 and Li-7 contributions
    # Li-6 contribution
    tbr_li6_tally = openmc.Tally(name='tritium_breeding_li6')
    tbr_li6_tally.filters = [lithium_filter]
    tbr_li6_tally.scores = ['(n,Xt)']
    tbr_li6_tally.nuclides = ['Li6']
    tallies.append(tbr_li6_tally)

    # Li-7 contribution
    tbr_li7_tally = openmc.Tally(name='tritium_breeding_li7')
    tbr_li7_tally.filters = [lithium_filter]
    tbr_li7_tally.scores = ['(n,Xt)']
    tbr_li7_tally.nuclides = ['Li7']
    tallies.append(tbr_li7_tally)

    print("\nCreated tritium breeding tallies:")
    print("  - Total TBR (all reactions)")
    print("  - Li-6 contribution")
    print("  - Li-7 contribution")

    return tallies


def create_all_tallies(geometry):
    """Create all tallies for the simulation."""
    all_tallies = openmc.Tallies()

    # 1. Normalization tallies (for flux normalization)
    norm_tallies = create_normalization_tallies()
    all_tallies.extend(norm_tallies)

    # 2. Full reactor mesh tallies
    mesh_tallies = create_full_reactor_mesh_tallies()
    all_tallies.extend(mesh_tallies)

    # 3. Cell-based energy-discretized tallies
    cell_tallies = create_cell_based_energy_tallies(geometry)
    all_tallies.extend(cell_tallies)

    # 4. Surface current tallies (NEW)
    surface_tallies = create_surface_current_tallies(geometry)
    all_tallies.extend(surface_tallies)

    # 5. Tritium breeding tallies
    tbr_tallies = create_tritium_breeding_tally(geometry)
    all_tallies.extend(tbr_tallies)

    print("\n" + "="*60)
    print(f"Total tallies created: {len(all_tallies)}")
    print("="*60)

    return all_tallies


if __name__ == '__main__':
    try:
        from materials import make_materials
        from geometry import create_core

        mat_dict = make_materials()
        geometry = create_core(mat_dict)

        tallies = create_all_tallies(geometry)

        tallies.export_to_xml()

        print("\nTallies exported successfully!")

    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure materials.py and geometry.py are in the same directory")
    except Exception as e:
        print(f"Error creating tallies: {e}")
        import traceback
        traceback.print_exc()
