"""
Tallies for SFR Tritium Breeder Assembly
Handles hexagonal geometry with cooling tubes
"""
import openmc
import numpy as np
from inputs import inputs, get_derived_dimensions


def create_sfr_tritium_breeder_tallies(geometry, surfaces_dict, energy_filters):
    """Create tallies specific to SFR hexagonal tritium breeder.

    Creates:
    1. Surface tally on cladding inner surface (current entering breeder)
    2. Flux tally in breeder material only (excluding cooling tubes)
    3. Tritium production tally in breeder material

    Parameters
    ----------
    geometry : openmc.Geometry
        The reactor geometry
    surfaces_dict : dict
        Dictionary with 'tritium_info' containing cells and surfaces
    energy_filters : dict
        Pre-created energy filters

    Returns
    -------
    tallies : openmc.Tallies
        Collection of SFR tritium breeder tallies
    """
    tallies = openmc.Tallies()

    # Get tritium breeder info
    tritium_info = surfaces_dict.get('tritium_info')
    if tritium_info is None:
        print("\n  Warning: No SFR tritium breeder found")
        return tallies

    try:
        cells_dict = tritium_info['cells_dict']
        surf_dict = tritium_info['surfaces_dict']

        # Get cells
        bundle_cell = cells_dict['bundle']  # Contains breeder universe with cooling tubes
        cladding_cell = cells_dict['cladding']  # Hexagonal cladding

        # Get surfaces
        hex_breeder_inner = surf_dict['hex_breeder_inner']  # Inner edge of breeder region

    except KeyError as e:
        print(f"\n  Warning: Could not find required SFR tritium info: {e}")
        return tallies

    # Find breeder material cells (excluding cooling tubes)
    breeder_cells = []
    for cell in geometry.get_all_cells().values():
        if cell.name == 'sfr_tritium_breeder_material':
            breeder_cells.append(cell)

    if not breeder_cells:
        print("\n  Warning: No SFR tritium breeder material cells found")
        return tallies

    print(f"\nCreating SFR tritium breeder tallies ({len(breeder_cells)} breeder cells)...")

    # Use pre-created energy filters
    thermal_filter = energy_filters['thermal']
    epithermal_filter = energy_filters['epithermal']
    fast_filter = energy_filters['fast']
    very_fast_filter = energy_filters.get('very_fast', fast_filter)  # Fallback to fast if not available
    log_1001_filter = energy_filters['log_1001']

    # ============================================================
    # 1. SURFACE TALLIES: Current from cladding into breeder
    # ============================================================
    print("  - Surface current tallies (cladding → breeder)...")

    # Create surface filter for inner hexagon boundary
    # Note: For hexagonal surfaces, we use the cell-based approach
    # We'll tally current entering the bundle cell from the cladding cell

    cladding_from_filter = openmc.CellFromFilter(cladding_cell)
    bundle_filter = openmc.CellFilter(bundle_cell)

    # Total current entering breeder from cladding
    current_total = openmc.Tally(name='sfr_tritium_current_total')
    current_total.filters = [bundle_filter, cladding_from_filter]
    current_total.scores = ['current']
    tallies.append(current_total)

    # Thermal current
    current_thermal = openmc.Tally(name='sfr_tritium_current_thermal')
    current_thermal.filters = [bundle_filter, cladding_from_filter, thermal_filter]
    current_thermal.scores = ['current']
    tallies.append(current_thermal)

    # Epithermal current
    current_epithermal = openmc.Tally(name='sfr_tritium_current_epithermal')
    current_epithermal.filters = [bundle_filter, cladding_from_filter, epithermal_filter]
    current_epithermal.scores = ['current']
    tallies.append(current_epithermal)

    # Fast current
    current_fast = openmc.Tally(name='sfr_tritium_current_fast')
    current_fast.filters = [bundle_filter, cladding_from_filter, fast_filter]
    current_fast.scores = ['current']
    tallies.append(current_fast)

    # Very-fast current (4th group)
    current_vfast = openmc.Tally(name='sfr_tritium_current_veryfast')
    current_vfast.filters = [bundle_filter, cladding_from_filter, very_fast_filter]
    current_vfast.scores = ['current']
    tallies.append(current_vfast)

    # LOG_1001 current spectrum
    current_log1001 = openmc.Tally(name='sfr_tritium_current_log1001')
    current_log1001.filters = [bundle_filter, cladding_from_filter, log_1001_filter]
    current_log1001.scores = ['current']
    tallies.append(current_log1001)

    # ============================================================
    # 2. FLUX TALLIES: In breeder material only
    # ============================================================
    print("  - Flux tallies (breeder material only)...")

    breeder_filter = openmc.CellFilter(breeder_cells)

    # Total flux
    flux_total = openmc.Tally(name='sfr_tritium_flux_total')
    flux_total.filters = [breeder_filter]
    flux_total.scores = ['flux']
    tallies.append(flux_total)

    # Thermal flux
    flux_thermal = openmc.Tally(name='sfr_tritium_flux_thermal')
    flux_thermal.filters = [breeder_filter, thermal_filter]
    flux_thermal.scores = ['flux']
    tallies.append(flux_thermal)

    # Epithermal flux
    flux_epithermal = openmc.Tally(name='sfr_tritium_flux_epithermal')
    flux_epithermal.filters = [breeder_filter, epithermal_filter]
    flux_epithermal.scores = ['flux']
    tallies.append(flux_epithermal)

    # Fast flux
    flux_fast = openmc.Tally(name='sfr_tritium_flux_fast')
    flux_fast.filters = [breeder_filter, fast_filter]
    flux_fast.scores = ['flux']
    tallies.append(flux_fast)

    # Very-fast flux (4th group)
    flux_vfast = openmc.Tally(name='sfr_tritium_flux_veryfast')
    flux_vfast.filters = [breeder_filter, very_fast_filter]
    flux_vfast.scores = ['flux']
    tallies.append(flux_vfast)

    # LOG_1001 flux spectrum
    flux_log1001 = openmc.Tally(name='sfr_tritium_flux_log1001')
    flux_log1001.filters = [breeder_filter, log_1001_filter]
    flux_log1001.scores = ['flux']
    tallies.append(flux_log1001)

    # ============================================================
    # 3. TRITIUM PRODUCTION: In breeder material only
    # ============================================================
    print("  - Tritium production tallies...")

    # Total tritium production
    tbr_tally = openmc.Tally(name='sfr_tritium_production')
    tbr_tally.filters = [breeder_filter]
    tbr_tally.scores = ['(n,Xt)']
    tallies.append(tbr_tally)

    # Li-6 contribution
    tbr_li6 = openmc.Tally(name='sfr_tritium_production_li6')
    tbr_li6.filters = [breeder_filter]
    tbr_li6.scores = ['(n,Xt)']
    tbr_li6.nuclides = ['Li6']
    tallies.append(tbr_li6)

    # Li-7 contribution
    tbr_li7 = openmc.Tally(name='sfr_tritium_production_li7')
    tbr_li7.filters = [breeder_filter]
    tbr_li7.scores = ['(n,Xt)']
    tbr_li7.nuclides = ['Li7']
    tallies.append(tbr_li7)

    print(f"  Created {len(tallies)} SFR tritium breeder tallies")

    return tallies


def create_sfr_core_mesh_tallies(energy_filters):
    """Create 4-group core mesh tallies for SFR.

    Parameters
    ----------
    energy_filters : dict
        Pre-created energy filters

    Returns
    -------
    tallies : openmc.Tallies
        Collection of SFR core mesh tallies
    """
    tallies = openmc.Tallies()

    # For SFR, use hexagonal core dimensions
    if inputs['assembly_type'] != 'sodium':
        return tallies

    # Create mesh covering entire SFR core
    # Use hexagonal boundary + SS316 wall
    core_edge = inputs['sfr_core_edge'] + inputs['sfr_ss316_wall_thickness']
    axial_height = inputs['sfr_axial_height'] + inputs['sfr_axial_reflector_thickness']

    # Create rectangular mesh (will be clipped by hexagonal boundary)
    core_mesh = openmc.RegularMesh()
    core_mesh.dimension = [400, 400, 50]
    core_mesh.lower_left = [-core_edge, -core_edge, -axial_height]
    core_mesh.upper_right = [core_edge, core_edge, axial_height]

    mesh_filter = openmc.MeshFilter(core_mesh)

    # Use pre-created energy filters
    thermal_filter = energy_filters['thermal']
    epithermal_filter = energy_filters['epithermal']
    fast_filter = energy_filters['fast']
    very_fast_filter = energy_filters.get('very_fast', fast_filter)

    print("\nCreating SFR core mesh tallies (4-group)...")

    # Total flux
    total_tally = openmc.Tally(name='sfr_core_mesh_total')
    total_tally.filters = [mesh_filter]
    total_tally.scores = ['flux']
    tallies.append(total_tally)

    # Thermal flux
    thermal_tally = openmc.Tally(name='sfr_core_mesh_thermal')
    thermal_tally.filters = [mesh_filter, thermal_filter]
    thermal_tally.scores = ['flux']
    tallies.append(thermal_tally)

    # Epithermal flux
    epithermal_tally = openmc.Tally(name='sfr_core_mesh_epithermal')
    epithermal_tally.filters = [mesh_filter, epithermal_filter]
    epithermal_tally.scores = ['flux']
    tallies.append(epithermal_tally)

    # Fast flux
    fast_tally = openmc.Tally(name='sfr_core_mesh_fast')
    fast_tally.filters = [mesh_filter, fast_filter]
    fast_tally.scores = ['flux']
    tallies.append(fast_tally)

    # Very-fast flux (4th group)
    vfast_tally = openmc.Tally(name='sfr_core_mesh_veryfast')
    vfast_tally.filters = [mesh_filter, very_fast_filter]
    vfast_tally.scores = ['flux']
    tallies.append(vfast_tally)

    print(f"  Mesh: 400 x 400 x 50")
    print(f"  Created {len(tallies)} mesh tallies")

    return tallies


def create_sfr_tritium_assembly_mesh(energy_filters):
    """Create 3x3 assembly mesh centered on tritium breeder for SFR.

    For SFR, the tritium breeder is at the center (0, 0).

    Parameters
    ----------
    energy_filters : dict
        Pre-created energy filters

    Returns
    -------
    tallies : openmc.Tallies
        Collection of assembly mesh tallies
    """
    tallies = openmc.Tallies()

    if inputs['assembly_type'] != 'sodium':
        return tallies

    derived = get_derived_dimensions()
    assembly_width = derived['assembly_width']

    # 3x3 assemblies centered on (0, 0) where tritium breeder is
    mesh_width = 3 * assembly_width

    # Create mesh
    assembly_mesh = openmc.RegularMesh()
    assembly_mesh.dimension = [300, 300, 1]
    assembly_mesh.lower_left = [-mesh_width/2, -mesh_width/2, -inputs['sfr_axial_height']]
    assembly_mesh.upper_right = [mesh_width/2, mesh_width/2, inputs['sfr_axial_height']]

    mesh_filter = openmc.MeshFilter(assembly_mesh)

    # Use pre-created energy filters
    thermal_filter = energy_filters['thermal']
    epithermal_filter = energy_filters['epithermal']
    fast_filter = energy_filters['fast']
    very_fast_filter = energy_filters.get('very_fast', fast_filter)

    print("\nCreating SFR tritium assembly mesh tallies...")

    # Total flux
    total_tally = openmc.Tally(name='sfr_tritium_assembly_mesh_total')
    total_tally.filters = [mesh_filter]
    total_tally.scores = ['flux']
    tallies.append(total_tally)

    # Thermal flux
    thermal_tally = openmc.Tally(name='sfr_tritium_assembly_mesh_thermal')
    thermal_tally.filters = [mesh_filter, thermal_filter]
    thermal_tally.scores = ['flux']
    tallies.append(thermal_tally)

    # Epithermal flux
    epithermal_tally = openmc.Tally(name='sfr_tritium_assembly_mesh_epithermal')
    epithermal_tally.filters = [mesh_filter, epithermal_filter]
    epithermal_tally.scores = ['flux']
    tallies.append(epithermal_tally)

    # Fast flux
    fast_tally = openmc.Tally(name='sfr_tritium_assembly_mesh_fast')
    fast_tally.filters = [mesh_filter, fast_filter]
    fast_tally.scores = ['flux']
    tallies.append(fast_tally)

    # Very-fast flux
    vfast_tally = openmc.Tally(name='sfr_tritium_assembly_mesh_veryfast')
    vfast_tally.filters = [mesh_filter, very_fast_filter]
    vfast_tally.scores = ['flux']
    tallies.append(vfast_tally)

    print(f"  3x3 assembly region: {mesh_width:.1f} cm x {mesh_width:.1f} cm")
    print(f"  Centered at (0, 0) - SFR tritium breeder location")
    print(f"  Mesh: 300 x 300 x 1")
    print(f"  Created {len(tallies)} mesh tallies")

    return tallies
