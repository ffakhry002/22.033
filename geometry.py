"""
Geometry for PWR Fusion Breeder Reactor
Modified to return surfaces for tally creation
"""
import openmc
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from inputs import inputs, get_derived_dimensions

# Material colors for plotting
MATERIAL_COLORS = {
    'ap_1000_fuel': 'orange',
    'ap_1000_gap': 'white',
    'ap_1000_cladding': 'gray',
    'ap_1000_coolant': 'lightblue',
    'ap_1000_coolant_outer': 'deepskyblue',
    'rpv_steel': 'darkgray',
    'natural_lithium': 'yellow',
    'enriched_lithium': 'gold',
    'natural_flibe': 'lightgreen',
    'enriched_flibe': 'limegreen',
    'natural_pbli': 'silver',
    'enriched_pbli': 'lightgray',
    'double_enriched_pbli': 'slategray',
    'heavy_water': 'cyan',
    'room_temp_lightwater': 'lightcyan',
    'helium_moderator': 'white',
    # CANDU materials
    'candu_fuel_central': 'lightgray',
    'candu_fuel_inner': 'darkgray',
    'candu_fuel_intermediate': 'gray',
    'candu_fuel_outer': 'dimgray',
    'candu_cladding': 'yellow',
    'candu_coolant': 'blue',
    'candu_moderator': 'lightblue',
    'candu_pressure_tube': 'black',
    'candu_calandria_tube': 'black',
    'candu_gap': 'white',
    'candu_fuel_gap': 'green',
    # Tritium breeder assembly materials
    'ti_grade7': 'purple',
    'vacuum_gap': 'lavender',
    # SFR materials
    'sodium': 'lightblue',
    'ss316': 'silver',
    'mox_inner': 'darkorange',
    'mox_outer': 'orange',
    'sfr_clad': 'goldenrod',
    'mgo_reflector': 'pink',
}

def CANDU_pin(mat_dict, fuel_mat_name):
    """Create a CANDU fuel pin cell universe with specified fuel material."""
    from math import pi, cos, sin

    r_fuel = inputs['candu_r_fuel']
    clad_thickness = inputs['candu_clad_thickness']
    r_clad = inputs['candu_r_clad']

    # Create surfaces
    fuel_surface = openmc.ZCylinder(r=r_fuel)
    clad_fuel = openmc.ZCylinder(r=clad_thickness)
    clad_surface = openmc.ZCylinder(r=r_clad)

    # Create cells
    fuel_cell = openmc.Cell(name='candu_fuel', fill=mat_dict[fuel_mat_name])
    fuel_cell.region = -fuel_surface

    gap_cell = openmc.Cell(name='candu_fuel_gap', fill=mat_dict['candu_fuel_gap'])
    gap_cell.region = +fuel_surface & -clad_fuel

    clad_cell = openmc.Cell(name='candu_cladding', fill=mat_dict['candu_cladding'])
    clad_cell.region = +clad_fuel & -clad_surface

    coolant_cell = openmc.Cell(name='candu_coolant', fill=mat_dict['candu_coolant'])
    coolant_cell.region = +clad_surface

    pin_universe = openmc.Universe(name='candu_pin', cells=[fuel_cell, gap_cell, clad_cell, coolant_cell])
    return pin_universe


def CANDU_bundle(mat_dict):
    """Create a CANDU fuel bundle with ring layout."""
    from math import pi, cos, sin

    ring_radii = np.array(inputs['candu_ring_radii'])
    num_pins = inputs['candu_num_pins']
    angles = inputs['candu_ring_angles']

    r_fuel = inputs['candu_r_fuel']
    clad_thickness = inputs['candu_clad_thickness']
    r_clad = inputs['candu_r_clad']

    # Fuel materials for each ring
    fuel_materials = ['candu_fuel_central', 'candu_fuel_inner', 'candu_fuel_intermediate', 'candu_fuel_outer']

    # Create surfaces that divide rings
    radial_surf = [openmc.ZCylinder(r=r) for r in (ring_radii[:-1] + ring_radii[1:]) / 2]

    # Create coolant cells for each ring (will be modified to exclude pins)
    water_cells = []
    for i in range(len(ring_radii)):
        # Create annular region
        if i == 0:
            water_region = -radial_surf[i]
        elif i == len(ring_radii) - 1:
            water_region = +radial_surf[i-1]
        else:
            water_region = +radial_surf[i-1] & -radial_surf[i]

        water_cell = openmc.Cell(fill=mat_dict['candu_coolant'], region=water_region)
        water_cells.append(water_cell)

    # Create bundle universe
    bundle_universe = openmc.Universe(cells=water_cells)

    # Create fuel pin surfaces (centered at origin, will be translated per pin)
    surf_fuel = openmc.ZCylinder(r=r_fuel)
    # clad_thickness is the cladding inner radius (from fuel surface)
    clad_inner_surf = openmc.ZCylinder(r=clad_thickness)
    # Cladding outer surface (at origin, relative to pin center)
    clad_outer_surf = openmc.ZCylinder(r=r_clad)

    pin_cells = []
    fuel_cells = []
    x_coord = []
    y_coord = []

    # Create pins for each ring
    for i, (r, n, a) in enumerate(zip(ring_radii, num_pins, angles)):
        fuel_mat_name = fuel_materials[i]

        for j in range(n):
            # Determine location of center of pin
            theta = (a + j/n*360.) * pi/180.
            x = r*cos(theta)
            y = r*sin(theta)

            pin_boundary = openmc.ZCylinder(x0=x, y0=y, r=r_clad)
            # Exclude pin from water region (like in notebook)
            water_cells[i].region &= +pin_boundary
            x_coord.append(x)
            y_coord.append(y)

            # Create fuel cell (relative to pin center)
            fuel_cell = openmc.Cell(fill=mat_dict[fuel_mat_name], region=-surf_fuel)
            # Don't assign ID - causes warnings when universe is reused
            # fuel_cell.id = (i + 1) * 1000 + j
            fuel_cells.append(fuel_cell)

            # Create gap cell (between fuel and cladding inner)
            gap_cell = openmc.Cell(fill=mat_dict['candu_fuel_gap'], region=+surf_fuel & -clad_inner_surf)

            # Create cladding cell (between cladding inner and outer)
            clad_cell = openmc.Cell(fill=mat_dict['candu_cladding'], region=+clad_inner_surf & -clad_outer_surf)

            # CRITICAL FIX: Add coolant cell extending infinitely outward (like BEAVRS)
            coolant_cell = openmc.Cell(fill=mat_dict['candu_coolant'], region=+clad_outer_surf)

            # Pin universe now has 4 cells: fuel, gap, clad, coolant (matching BEAVRS style)
            pin_universe = openmc.Universe(cells=(fuel_cell, gap_cell, clad_cell, coolant_cell))

            # Create pin cell
            pin = openmc.Cell(fill=pin_universe, region=-pin_boundary)
            pin.translation = (x, y, 0)
            # Don't assign ID - causes warnings when universe is reused
            # pin.id = (i + 1)*100 + j
            bundle_universe.add_cell(pin)
            pin_cells.append(pin)

    return bundle_universe


def CANDU_assembly(mat_dict):
    """Create a CANDU fuel assembly with pressure tube, calandria, and moderator."""
    from math import pi

    bundle_universe = CANDU_bundle(mat_dict)

    # Tube radii
    pressure_tube_ir = inputs['candu_pressure_tube_ir']
    pressure_tube_or = inputs['candu_pressure_tube_or']
    calandria_ir = inputs['candu_calandria_ir']
    calandria_or = inputs['candu_calandria_or']

    # Create surfaces
    pt_inner = openmc.ZCylinder(r=pressure_tube_ir)
    pt_outer = openmc.ZCylinder(r=pressure_tube_or)
    calandria_inner = openmc.ZCylinder(r=calandria_ir)
    calandria_outer = openmc.ZCylinder(r=calandria_or)

    # Create cells
    # Note: No explicit boundaries needed - lattice automatically handles cell boundaries
    bundle = openmc.Cell(fill=bundle_universe, region=-pt_inner)
    pressure_tube = openmc.Cell(fill=mat_dict['candu_pressure_tube'], region=+pt_inner & -pt_outer)
    gap_cell = openmc.Cell(fill=mat_dict['candu_gap'], region=+pt_outer & -calandria_inner)
    calandria = openmc.Cell(fill=mat_dict['candu_calandria_tube'], region=+calandria_inner & -calandria_outer)
    # Moderator fills everything outside calandria - lattice handles boundaries
    moder = openmc.Cell(fill=mat_dict['candu_moderator'], region=+calandria_outer)

    root_universe = openmc.Universe(cells=[bundle, pressure_tube, gap_cell, calandria, moder])

    return root_universe


def tritium_breeder_assembly(mat_dict):
    """Create a tritium breeding assembly with Ti-0.2Pd pressure tube, vacuum gap,
    calandria tube, and internal coolant tubes.

    Returns
    -------
    root_universe : openmc.Universe
        The complete tritium breeder assembly universe
    cells_dict : dict
        Dictionary of key cells for tally creation:
        - 'breeder_cell': The main breeder cell (including coolant tubes)
        - 'pressure_tube_cell': The pressure tube cell
        - 'calandria_cell': The calandria tube cell
    surfaces_dict : dict
        Dictionary of key surfaces for tally creation:
        - 'pt_inner': Pressure tube inner surface
        - 'pt_outer': Pressure tube outer surface
        - 'calandria_inner': Calandria inner surface
        - 'calandria_outer': Calandria outer surface
    """
    from math import pi, cos, sin

    # Use same outer dimensions as CANDU
    pressure_tube_ir = inputs['candu_pressure_tube_ir']  # 5.1689 cm
    pressure_tube_or = inputs['candu_pressure_tube_or']  # 5.621 cm
    calandria_ir = inputs['candu_calandria_ir']          # 6.6002 cm
    calandria_or = inputs['candu_calandria_or']          # 6.7526 cm

    # Coolant tube dimensions
    coolant_tube_or = 1.0  # cm
    coolant_tube_wall_thickness = 0.1  # cm
    coolant_tube_ir = coolant_tube_or - coolant_tube_wall_thickness

    # Coolant tube positions: center + 4 at two-thirds radius
    two_thirds_radius = pressure_tube_ir * 2.0 / 3.0  # 3.446 cm
    coolant_positions = [
        (0.0, 0.0),                      # Center
        (two_thirds_radius, 0.0),        # 3 o'clock
        (0.0, two_thirds_radius),        # 12 o'clock
        (-two_thirds_radius, 0.0),       # 9 o'clock
        (0.0, -two_thirds_radius),       # 6 o'clock
    ]

    # Create surfaces for main structure
    pt_inner = openmc.ZCylinder(r=pressure_tube_ir)
    pt_outer = openmc.ZCylinder(r=pressure_tube_or)
    calandria_inner = openmc.ZCylinder(r=calandria_ir)
    calandria_outer = openmc.ZCylinder(r=calandria_or)

    # Select breeder material
    breeder_material = inputs['breeder_material']
    breeder_mat = mat_dict[breeder_material]

    # Create breeder region (will exclude coolant tubes)
    breeder_region = -pt_inner

    # Create coolant tubes
    tube_cells = []
    for i, (x, y) in enumerate(coolant_positions):
        tube_outer_surf = openmc.ZCylinder(x0=x, y0=y, r=coolant_tube_or)
        tube_inner_surf = openmc.ZCylinder(x0=x, y0=y, r=coolant_tube_ir)

        # Exclude from breeder region
        breeder_region &= +tube_outer_surf

        # Coolant inside
        coolant_inner = openmc.Cell(name='tritium_breeder_coolant_inner', fill=mat_dict['ap_1000_coolant'], region=-tube_inner_surf)
        # Ti-0.2Pd wall
        tube_wall = openmc.Cell(name='tritium_breeder_coolant_wall', fill=mat_dict['ti_grade7'], region=+tube_inner_surf & -tube_outer_surf)

        tube_cells.extend([coolant_inner, tube_wall])

    # Create breeder cell (excludes coolant tubes but we'll include coolant in tally)
    breeder_cell = openmc.Cell(name='tritium_breeder_material', fill=breeder_mat, region=breeder_region)

    # Create universe containing breeder and coolant tubes
    breeder_universe = openmc.Universe(cells=[breeder_cell] + tube_cells)

    # Main assembly cells
    bundle = openmc.Cell(name='tritium_bundle', fill=breeder_universe, region=-pt_inner)
    pressure_tube = openmc.Cell(name='tritium_pressure_tube', fill=mat_dict['ti_grade7'], region=+pt_inner & -pt_outer)
    gap_cell = openmc.Cell(name='tritium_gap', fill=mat_dict['vacuum_gap'], region=+pt_outer & -calandria_inner)
    calandria = openmc.Cell(name='tritium_calandria', fill=mat_dict['candu_calandria_tube'], region=+calandria_inner & -calandria_outer)
    moder = openmc.Cell(name='tritium_moderator', fill=mat_dict['candu_moderator'], region=+calandria_outer)

    root_universe = openmc.Universe(cells=[bundle, pressure_tube, gap_cell, calandria, moder])

    # Return universe and key cells/surfaces for tallies
    cells_dict = {
        'bundle': bundle,  # Contains the breeder universe
        'pressure_tube': pressure_tube,
        'calandria': calandria,
        'moderator': moder
    }

    surfaces_dict = {
        'pt_inner': pt_inner,
        'pt_outer': pt_outer,
        'calandria_inner': calandria_inner,
        'calandria_outer': calandria_outer
    }

    return root_universe, cells_dict, surfaces_dict


def BEAVRS_pin(mat_dict):
    """Create a BEAVRS fuel pin cell universe."""
    # Create surfaces
    fuel_surface = openmc.ZCylinder(r=inputs['fuel_or'])
    gap_surface = openmc.ZCylinder(r=inputs['clad_ir'])
    clad_surface = openmc.ZCylinder(r=inputs['clad_or'])

    # Create cells
    fuel_cell = openmc.Cell(name='fuel', fill=mat_dict['ap_1000_fuel'])
    fuel_cell.region = -fuel_surface

    gap_cell = openmc.Cell(name='gap', fill=mat_dict['ap_1000_gap'])
    gap_cell.region = +fuel_surface & -gap_surface

    clad_cell = openmc.Cell(name='cladding', fill=mat_dict['ap_1000_cladding'])
    clad_cell.region = +gap_surface & -clad_surface

    coolant_cell = openmc.Cell(name='coolant', fill=mat_dict['ap_1000_coolant'])
    coolant_cell.region = +clad_surface

    pin_universe = openmc.Universe(name='fuel_pin', cells=[fuel_cell, gap_cell, clad_cell, coolant_cell])
    return pin_universe


def BEAVRS_guide_tube(mat_dict):
    """Create a BEAVRS guide tube universe."""
    inner_surface = openmc.ZCylinder(r=inputs['gt_ir'])
    outer_surface = openmc.ZCylinder(r=inputs['gt_or'])

    inner_water_cell = openmc.Cell(name='gt_inner_water', fill=mat_dict['ap_1000_coolant'])
    inner_water_cell.region = -inner_surface

    tube_cell = openmc.Cell(name='gt_tube', fill=mat_dict['ap_1000_cladding'])
    tube_cell.region = +inner_surface & -outer_surface

    outer_water_cell = openmc.Cell(name='gt_outer_water', fill=mat_dict['ap_1000_coolant'])
    outer_water_cell.region = +outer_surface

    gt_universe = openmc.Universe(name='guide_tube', cells=[inner_water_cell, tube_cell, outer_water_cell])
    return gt_universe


def BEAVRS_assembly(mat_dict):
    """Create a BEAVRS 17x17 fuel assembly."""
    fuel_pin = BEAVRS_pin(mat_dict)
    guide_tube = BEAVRS_guide_tube(mat_dict)

    # Create lattice
    assembly_lattice = openmc.RectLattice(name='assembly_lattice')
    assembly_lattice.pitch = (inputs['pin_pitch'], inputs['pin_pitch'])
    assembly_lattice.lower_left = (-inputs['n_pins'] * inputs['pin_pitch'] / 2,
                                   -inputs['n_pins'] * inputs['pin_pitch'] / 2)

    # Fill lattice with fuel pins
    lattice_universes = np.full((inputs['n_pins'], inputs['n_pins']), fuel_pin)

    # Place guide tubes
    for i, j in inputs['guide_tube_positions']:
        lattice_universes[i, j] = guide_tube

    assembly_lattice.universes = lattice_universes

    # Set outer universe to coolant
    coolant_universe = openmc.Universe(name='outer_coolant')
    coolant_cell = openmc.Cell(fill=mat_dict['ap_1000_coolant'])
    coolant_universe.add_cell(coolant_cell)
    assembly_lattice.outer = coolant_universe

    assembly_cell = openmc.Cell(name='assembly_cell', fill=assembly_lattice)
    assembly_universe = openmc.Universe(name='assembly', cells=[assembly_cell])

    return assembly_universe


def SFR_pin(mat_dict, fuel_mat_name):
    """Create a Sodium Fast Reactor fuel pin cell universe.

    Parameters
    ----------
    mat_dict : dict
        Dictionary of materials
    fuel_mat_name : str
        Name of fuel material ('mox_inner' or 'mox_outer')

    Returns
    -------
    pin_universe : openmc.Universe
        The fuel pin universe
    """
    # Create surfaces
    fuel_surface = openmc.ZCylinder(r=inputs['sfr_fuel_or'])
    gap_outer_surface = openmc.ZCylinder(r=inputs['sfr_gap_or'])
    clad_surface = openmc.ZCylinder(r=inputs['sfr_clad_or'])

    # Create cells
    fuel_cell = openmc.Cell(name='sfr_fuel', fill=mat_dict[fuel_mat_name])
    fuel_cell.region = -fuel_surface

    # Gap filled with FUEL (same as benchmark!) - not sodium
    gap_cell = openmc.Cell(name='sfr_gap', fill=mat_dict[fuel_mat_name])
    gap_cell.region = +fuel_surface & -gap_outer_surface

    clad_cell = openmc.Cell(name='sfr_cladding', fill=mat_dict['sfr_clad'])
    clad_cell.region = +gap_outer_surface & -clad_surface

    sodium_cell = openmc.Cell(name='sfr_sodium', fill=mat_dict['sodium'])
    sodium_cell.region = +clad_surface

    pin_universe = openmc.Universe(name='sfr_pin', cells=[fuel_cell, gap_cell, clad_cell, sodium_cell])
    return pin_universe


def SFR_assembly(mat_dict, fuel_mat_name):
    """Create a Sodium Fast Reactor hexagonal assembly with mixed fuel rings.

    Parameters
    ----------
    mat_dict : dict
        Dictionary of materials
    fuel_mat_name : str
        Name of fuel material ('mox_inner' or 'mox_outer')

    Returns
    -------
    assembly_universe : openmc.Universe
        The complete assembly universe
    """
    # Create sodium-only universe for outer region
    sodium_mod_cell = openmc.Cell(fill=mat_dict['sodium'])
    sodium_mod_u = openmc.Universe(cells=(sodium_mod_cell,))

    # Create hexagonal lattice
    assembly_lattice = openmc.HexLattice(name='sfr_assembly_lattice')
    assembly_lattice.center = (0., 0.)
    assembly_lattice.pitch = (inputs['sfr_pin_pitch'],)
    assembly_lattice.orientation = 'x'
    assembly_lattice.outer = sodium_mod_u

    # Create rings - all pins in an assembly use the SAME fuel type
    # (like the notebook: inner assemblies are uniform inner, outer assemblies are uniform outer)
    pins_per_ring = inputs['sfr_pins_per_ring']  # [48, 42, 36, 30, 24, 18, 12, 6, 1]
    lattice_rings = []

    # Create fuel pin with specified fuel type (uniform throughout assembly)
    fuel_pin = SFR_pin(mat_dict, fuel_mat_name)

    for num_pins in pins_per_ring:
        lattice_rings.append([fuel_pin] * num_pins)

    assembly_lattice.universes = lattice_rings

    # Create hexagonal prism boundary - defines the assembly boundary
    hex_boundary = openmc.model.HexagonalPrism(edge_length=inputs['sfr_assembly_edge'], orientation='x')

    # Create main assembly cell (lattice inside hex boundary)
    main_assembly = openmc.Cell(fill=assembly_lattice, region=-hex_boundary)

    # Create outer sodium cell (fills space outside hex boundary, between assemblies)
    outer_sodium_cell = openmc.Cell(fill=mat_dict['sodium'])
    # Don't specify region - it will fill everything not already defined

    # Create universe
    assembly_universe = openmc.Universe(cells=[main_assembly, outer_sodium_cell])

    return assembly_universe


def SFR_reflector_assembly(mat_dict):
    """Create a Sodium Fast Reactor reflector assembly (sodium-filled).

    Parameters
    ----------
    mat_dict : dict
        Dictionary of materials

    Returns
    -------
    reflector_universe : openmc.Universe
        The reflector assembly universe
    """
    # Create hexagonal prism boundary for sodium reflector
    hex_boundary = openmc.model.HexagonalPrism(
        edge_length=inputs['sfr_assembly_edge'],
        orientation='x'
    )

    # Create sodium reflector cell (fills entire assembly)
    ref_cell = openmc.Cell(fill=mat_dict['sodium'], region=-hex_boundary)

    # Create outer sodium cell (fills space outside hex boundary, between assemblies)
    outer_sodium_cell = openmc.Cell(fill=mat_dict['sodium'])
    # Don't specify region - it will fill everything not already defined

    # Create universe
    reflector_universe = openmc.Universe(cells=[ref_cell, outer_sodium_cell])

    return reflector_universe


def SFR_tritium_breeder_assembly(mat_dict, breeder_material_override=None):
    """Create a tritium breeding assembly with hexagonal shape for SFR.

    Simplified design with solid breeder material (no cooling channels).

    Parameters
    ----------
    mat_dict : dict
        Dictionary of materials
    breeder_material_override : str, optional
        Override breeder material name (for parametric studies)

    Returns
    -------
    root_universe : openmc.Universe
        The complete tritium breeder assembly universe
    cells_dict : dict
        Dictionary of key cells for tally creation
    surfaces_dict : dict
        Dictionary of key surfaces for tally creation
    """
    # Assembly hex boundary
    hex_edge = inputs['sfr_tritium_breeder_edge']
    hex_assembly_boundary = openmc.model.HexagonalPrism(edge_length=hex_edge, orientation='x')

    # Cladding thickness = 1 cm
    cladding_thickness = 1.0  # cm

    # Inner hexagonal breeder region (assembly size minus cladding)
    breeder_hex_edge = hex_edge - cladding_thickness
    hex_breeder_inner = openmc.model.HexagonalPrism(edge_length=breeder_hex_edge, orientation='x')

    # Outer hexagonal cladding boundary (same as assembly boundary)
    hex_breeder_outer = openmc.model.HexagonalPrism(edge_length=hex_edge, orientation='x')

    # Select breeder material (allow override for parametric studies)
    if breeder_material_override is not None:
        breeder_material = breeder_material_override
    else:
        breeder_material = inputs['breeder_material']
    breeder_mat = mat_dict[breeder_material]

    # Create breeder cell (solid hexagonal region - NO cooling tubes)
    breeder_cell = openmc.Cell(
        name='sfr_tritium_breeder_material',
        fill=breeder_mat,
        region=-hex_breeder_inner
    )

    # Create universe containing only breeder (simplified design)
    breeder_universe = openmc.Universe(cells=[breeder_cell])

    # Main assembly cells
    # 1. Breeder bundle (hexagonal)
    bundle = openmc.Cell(
        name='sfr_tritium_bundle',
        fill=breeder_universe,
        region=-hex_breeder_inner & -hex_assembly_boundary
    )

    # 2. Hexagonal cladding (Ti-0.2Pd) - fills space between breeder and assembly boundary
    cladding_cell = openmc.Cell(
        name='sfr_tritium_cladding',
        fill=mat_dict['ti_grade7'],
        region=+hex_breeder_inner & -hex_assembly_boundary
    )

    # 4. Outer sodium cell (fills space outside hex boundary, between assemblies)
    outer_sodium_cell = openmc.Cell(
        name='sfr_tritium_outer_sodium',
        fill=mat_dict['sodium']
    )

    root_universe = openmc.Universe(cells=[bundle, cladding_cell, outer_sodium_cell])

    # Return universe and key cells/surfaces for tallies
    cells_dict = {
        'bundle': bundle,
        'cladding': cladding_cell
    }

    surfaces_dict = {
        'hex_breeder_inner': hex_breeder_inner,
        'hex_breeder_outer': hex_breeder_outer,
        'hex_assembly_boundary': hex_assembly_boundary
    }

    return root_universe, cells_dict, surfaces_dict


def build_SFR_core_lattice(mat_dict, hex_boundary, plane_bottom, plane_top,
                           tritium_location='ring7', breeder_material_override=None):
    """Build a hexagonal SFR core lattice matching the reference benchmark pattern.

    Uses the exact ring pattern from the European SFR benchmark with mixed transition zones.
    Tritium breeder can be placed at center or Ring 7 position for parametric studies.

    Parameters
    ----------
    mat_dict : dict
        Dictionary of materials
    hex_boundary : openmc.Surface
        Hexagonal boundary surface
    plane_bottom, plane_top : openmc.Surface
        Axial boundary planes
    tritium_location : str
        'center' or 'ring7' for tritium breeder placement
    breeder_material_override : str, optional
        Override breeder material name (for parametric studies)

    Returns
    -------
    lattice_cell : openmc.Cell
        Cell containing the SFR hexagonal core lattice
    tritium_info : dict or None
        Dictionary with tritium breeder assembly info and location
    """
    # Create assembly types
    inner_fuel = SFR_assembly(mat_dict, 'mox_inner')
    outer_fuel = SFR_assembly(mat_dict, 'mox_outer')
    reflector = SFR_reflector_assembly(mat_dict)  # Sodium reflector

    # Create sodium-only universe for outer region
    sodium_mod_cell = openmc.Cell(fill=mat_dict['sodium'])
    sodium_mod_u = openmc.Universe(cells=(sodium_mod_cell,))

    # Create tritium breeder assembly
    tritium_assembly, tritium_cells, tritium_surfaces = SFR_tritium_breeder_assembly(
        mat_dict, breeder_material_override
    )

    # Track location and material for tallies
    breeder_mat_name = breeder_material_override if breeder_material_override else inputs['breeder_material']
    tritium_info = {
        'cells_dict': tritium_cells,
        'surfaces_dict': tritium_surfaces,
        'location': tritium_location,
        'breeder_material': breeder_mat_name
    }

    # Create hexagonal core lattice
    core_lattice = openmc.HexLattice(name='sfr_core')
    core_lattice.center = (0., 0.)
    core_lattice.pitch = (inputs['sfr_assembly_pitch'],)
    core_lattice.outer = sodium_mod_u
    core_lattice.orientation = 'x'

    # =========================================================================
    # EXACT REFERENCE PATTERN (from European SFR benchmark)
    # =========================================================================
    # Reference has 17 rings total with mixed transition zones
    # Pattern notation: R=reflector, O=outer fuel, I=inner fuel, T=tritium

    lattice_rings = []

    # Ring 1 (96 assemblies): ALL REFLECTOR
    lattice_rings.append([reflector] * 96)

    # Ring 2 (90 assemblies): ALL REFLECTOR
    lattice_rings.append([reflector] * 90)

    # Ring 3 (84 assemblies): ALL REFLECTOR
    lattice_rings.append([reflector] * 84)

    # Ring 4 (78 assemblies): MIXED - [5R, 4O, 4R] × 6-fold symmetry
    ring4 = []
    for _ in range(6):
        ring4.extend([reflector] * 5)
        ring4.extend([outer_fuel] * 4)
        ring4.extend([reflector] * 4)
    lattice_rings.append(ring4)

    # Ring 5 (72 assemblies): MIXED - [1R, 11O] × 6-fold symmetry
    ring5 = []
    for _ in range(6):
        ring5.extend([reflector] * 1)
        ring5.extend([outer_fuel] * 11)
    lattice_rings.append(ring5)

    # Ring 6 (66 assemblies): ALL OUTER FUEL
    lattice_rings.append([outer_fuel] * 66)

    # Ring 7 (60 assemblies): MIXED or ALL OUTER FUEL (depends on tritium location)
    if tritium_location == 'ring7':
        # Place tritium breeder at position 30 (middle of ring for accessibility)
        ring7 = [outer_fuel] * 60
        ring7[30] = tritium_assembly  # Replace one outer fuel with tritium breeder
        lattice_rings.append(ring7)
    else:
        # All outer fuel if tritium is at center
        lattice_rings.append([outer_fuel] * 60)

    # Ring 8 (54 assemblies): MIXED - [2O, 6I, 1O] × 6-fold symmetry
    ring8 = []
    for _ in range(6):
        ring8.extend([outer_fuel] * 2)
        ring8.extend([inner_fuel] * 6)
        ring8.extend([outer_fuel] * 1)
    lattice_rings.append(ring8)

    # Ring 9 (48 assemblies): ALL INNER FUEL
    lattice_rings.append([inner_fuel] * 48)

    # Ring 10 (42 assemblies): ALL INNER FUEL
    lattice_rings.append([inner_fuel] * 42)

    # Ring 11 (36 assemblies): ALL INNER FUEL
    lattice_rings.append([inner_fuel] * 36)

    # Ring 12 (30 assemblies): ALL INNER FUEL
    lattice_rings.append([inner_fuel] * 30)

    # Ring 13 (24 assemblies): ALL INNER FUEL
    lattice_rings.append([inner_fuel] * 24)

    # Ring 14 (18 assemblies): ALL INNER FUEL
    lattice_rings.append([inner_fuel] * 18)

    # Ring 15 (12 assemblies): ALL INNER FUEL
    lattice_rings.append([inner_fuel] * 12)

    # Ring 16 (6 assemblies): ALL INNER FUEL
    lattice_rings.append([inner_fuel] * 6)

    # Ring 17 (1 assembly): CENTER - INNER FUEL or TRITIUM BREEDER (depends on location)
    if tritium_location == 'center':
        lattice_rings.append([tritium_assembly])  # Tritium at center
    else:
        lattice_rings.append([inner_fuel])  # Fuel at center (tritium in Ring 7)

    core_lattice.universes = lattice_rings

    # Create lattice cell with hexagonal boundary
    lattice_cell = openmc.Cell(name='sfr_core_region')
    lattice_cell.region = -hex_boundary & +plane_bottom & -plane_top
    lattice_cell.fill = core_lattice

    return lattice_cell, tritium_info


def build_core_lattice(mat_dict, cyl_core, plane_bottom, plane_top):
    """Build a lattice of assemblies based on core_lattice from inputs.

    Returns
    -------
    lattice_cell : openmc.Cell
        Cell containing the core lattice
    tritium_info : dict or None
        Dictionary with tritium breeder assembly info if present:
        - 'cells_dict': Dictionary of key cells
        - 'surfaces_dict': Dictionary of key surfaces
    """
    derived = get_derived_dimensions()
    assembly_width = derived['assembly_width']

    # Choose assembly type based on inputs
    if inputs['assembly_type'] == 'candu':
        fuel_assembly = CANDU_assembly(mat_dict)
        # For CANDU, use moderator as outer coolant
        coolant_outer_universe = openmc.Universe(name='coolant_outer')
        coolant_cell = openmc.Cell(fill=mat_dict['candu_moderator'])
        coolant_outer_universe.add_cell(coolant_cell)
    else:
        fuel_assembly = BEAVRS_assembly(mat_dict)
        # Outer coolant universe (cold coolant)
        coolant_outer_universe = openmc.Universe(name='coolant_outer')
        coolant_cell = openmc.Cell(fill=mat_dict['ap_1000_coolant_outer'])
        coolant_outer_universe.add_cell(coolant_cell)

    # Create tritium breeder assembly (can be used multiple times)
    # Returns universe, cells_dict, and surfaces_dict
    tritium_assembly, tritium_cells, tritium_surfaces = tritium_breeder_assembly(mat_dict)
    tritium_info = {
        'cells_dict': tritium_cells,
        'surfaces_dict': tritium_surfaces
    }

    # Select appropriate lattice based on assembly type
    if inputs['assembly_type'] == 'candu':
        core_lattice = inputs['candu_lattice']
    else:
        core_lattice = inputs['ap1000_lattice']

    # Build lattice universes array
    n_rows = len(core_lattice)
    n_cols = len(core_lattice[0])
    lattice_universes = []

    for row in core_lattice:
        lattice_row = []
        for symbol in row:
            if symbol == 'F':
                lattice_row.append(fuel_assembly)
            elif symbol == 'C':
                lattice_row.append(coolant_outer_universe)
            elif symbol.startswith('T'):  # T_1, T_2, etc. or just T
                lattice_row.append(tritium_assembly)
            else:
                raise ValueError(f"Unknown core lattice symbol: {symbol}. Use 'C', 'F', or 'T'/'T_1'/'T_2'/...")
        lattice_universes.append(lattice_row)

    lattice_universes = np.array(lattice_universes)

    # Create rectangular lattice
    core_lattice_obj = openmc.RectLattice(name='core_lattice')
    core_lattice_obj.lower_left = (-n_cols * assembly_width / 2, -n_rows * assembly_width / 2)
    core_lattice_obj.pitch = (assembly_width, assembly_width)
    core_lattice_obj.universes = lattice_universes
    core_lattice_obj.outer = coolant_outer_universe

    # Create cell containing the lattice
    lattice_cell = openmc.Cell(name='core_region')
    lattice_cell.region = -cyl_core & +plane_bottom & -plane_top
    lattice_cell.fill = core_lattice_obj

    return lattice_cell, tritium_info


def create_SFR_core(mat_dict, tritium_location='ring7', breeder_material_override=None):
    """Create the full SFR core geometry with hexagonal layout and MgO reflector.

    Parameters
    ----------
    mat_dict : dict
        Dictionary of materials
    tritium_location : str
        'center' or 'ring7' for tritium breeder placement
    breeder_material_override : str, optional
        Override breeder material name (for parametric studies)

    Returns
    -------
    geometry : openmc.Geometry
        The complete SFR reactor geometry
    surfaces_dict : dict
        Dictionary of surfaces for use in tallies
    """
    derived = get_derived_dimensions()

    # Create axial planes with MgO reflectors
    reflector_thickness = inputs['sfr_axial_reflector_thickness']
    plane_bottom_reflector = openmc.ZPlane(
        z0=-(inputs['sfr_axial_height'] + reflector_thickness),
        boundary_type='vacuum'
    )
    plane_fuel_bottom = openmc.ZPlane(z0=-inputs['sfr_axial_height'])
    plane_fuel_top = openmc.ZPlane(z0=inputs['sfr_axial_height'])
    plane_top_reflector = openmc.ZPlane(
        z0=inputs['sfr_axial_height'] + reflector_thickness,
        boundary_type='vacuum'
    )

    # Create hexagonal boundaries
    # Inner boundary: core with sodium reflector assemblies
    hex_core_boundary = openmc.model.HexagonalPrism(
        edge_length=inputs['sfr_core_edge'],
        orientation='x'
    )

    # Outer boundary: MgO reflector wall (excellent for fast spectrum)
    hex_reflector_boundary = openmc.model.HexagonalPrism(
        edge_length=inputs['sfr_core_edge'] + inputs['sfr_ss316_wall_thickness'],
        orientation='x',
        boundary_type='vacuum'
    )

    # Build SFR fuel region with hexagonal lattice
    fuel_region_cell, tritium_info = build_SFR_core_lattice(
        mat_dict,
        hex_core_boundary,
        plane_fuel_bottom,
        plane_fuel_top,
        tritium_location,
        breeder_material_override
    )

    # Bottom axial reflector (MgO - excellent for fast spectrum)
    bottom_reflector_cell = openmc.Cell(name='sfr_bottom_reflector')
    bottom_reflector_cell.region = -hex_reflector_boundary & +plane_bottom_reflector & -plane_fuel_bottom
    bottom_reflector_cell.fill = mat_dict['mgo_reflector']

    # Top axial reflector (MgO)
    top_reflector_cell = openmc.Cell(name='sfr_top_reflector')
    top_reflector_cell.region = -hex_reflector_boundary & +plane_fuel_top & -plane_top_reflector
    top_reflector_cell.fill = mat_dict['mgo_reflector']

    # Radial MgO reflector wall (surrounds core)
    radial_reflector_wall = openmc.Cell(name='sfr_radial_reflector_wall')
    radial_reflector_wall.region = +hex_core_boundary & -hex_reflector_boundary & +plane_fuel_bottom & -plane_fuel_top
    radial_reflector_wall.fill = mat_dict['mgo_reflector']

    # Create root universe and geometry
    root_universe = openmc.Universe(cells=[
        bottom_reflector_cell,
        fuel_region_cell,
        radial_reflector_wall,
        top_reflector_cell
    ])
    geometry = openmc.Geometry(root_universe)

    # Store surfaces in dictionary for tallies
    surfaces_dict = {
        'hex_core_boundary': hex_core_boundary,
        'hex_reflector_boundary': hex_reflector_boundary,
        'plane_bottom_reflector': plane_bottom_reflector,
        'plane_fuel_bottom': plane_fuel_bottom,
        'plane_fuel_top': plane_fuel_top,
        'plane_top_reflector': plane_top_reflector,
        'tritium_info': tritium_info
    }

    return geometry, surfaces_dict


def create_core(mat_dict):
    """Create the full core geometry with reflectors and containment.

    Returns
    -------
    geometry : openmc.Geometry
        The complete reactor geometry
    surfaces_dict : dict
        Dictionary of cylindrical surfaces for use in tallies
    """
    derived = get_derived_dimensions()

    # Check if this is an SFR geometry
    if inputs['assembly_type'] == 'sodium':
        # For run.py, default to center location for tritium breeder
        # (For parametric study, location is passed explicitly)
        tritium_location = inputs.get('sfr_tritium_location', 'center')
        return create_SFR_core(mat_dict, tritium_location=tritium_location)

    # Select breeder material based on configuration
    valid_materials = [
        'natural_lithium', 'enriched_lithium',
        'natural_flibe', 'enriched_flibe',
        'natural_pbli', 'enriched_pbli', 'double_enriched_pbli'
    ]

    breeder_material = inputs['breeder_material']
    if breeder_material not in valid_materials:
        raise ValueError(f"Invalid breeder_material: {breeder_material}. Must be one of {valid_materials}")

    breeder_mat = mat_dict[breeder_material]

    # Create cylindrical surfaces
    cyl_core = openmc.ZCylinder(r=inputs['r_core'])
    cyl_outer_tank = openmc.ZCylinder(r=derived['r_outer_tank'])
    cyl_rpv_1 = openmc.ZCylinder(r=derived['r_rpv_1'])
    cyl_rpv_2 = openmc.ZCylinder(r=derived['r_rpv_2'])

    # Create surfaces for moderator region (if enabled)
    if inputs['enable_moderator_region']:
        cyl_moderator = openmc.ZCylinder(r=derived['r_moderator'])
        cyl_wall_divider = openmc.ZCylinder(r=derived['r_wall_divider'])
        cyl_lithium = openmc.ZCylinder(r=derived['r_lithium'])
    else:
        cyl_moderator = None
        cyl_wall_divider = None
        cyl_lithium = openmc.ZCylinder(r=derived['r_lithium'])

    cyl_lithium_wall = openmc.ZCylinder(r=derived['r_lithium_wall'], boundary_type='vacuum')

    # Store surfaces in dictionary for tallies
    surfaces_dict = {
        'cyl_core': cyl_core,
        'cyl_outer_tank': cyl_outer_tank,
        'cyl_rpv_1': cyl_rpv_1,
        'cyl_rpv_2': cyl_rpv_2,
        'cyl_lithium': cyl_lithium,
        'cyl_lithium_wall': cyl_lithium_wall
    }

    # Add moderator surfaces to dictionary if enabled
    if inputs['enable_moderator_region']:
        surfaces_dict['cyl_moderator'] = cyl_moderator
        surfaces_dict['cyl_wall_divider'] = cyl_wall_divider

    # Create axial planes
    plane_bottom = openmc.ZPlane(z0=derived['z_bottom'], boundary_type='vacuum')
    plane_fuel_bottom = openmc.ZPlane(z0=derived['z_fuel_bottom'])
    plane_fuel_top = openmc.ZPlane(z0=derived['z_fuel_top'])
    plane_top = openmc.ZPlane(z0=derived['z_top'], boundary_type='vacuum')

    # Build fuel region (returns lattice cell and tritium breeder info)
    fuel_region_cell, tritium_info = build_core_lattice(mat_dict, cyl_core, plane_fuel_bottom, plane_fuel_top)

    # Bottom reflector
    bottom_reflector_cell = openmc.Cell(name='bottom_reflector')
    bottom_reflector_cell.region = -cyl_lithium_wall & +plane_bottom & -plane_fuel_bottom
    bottom_reflector_cell.fill = mat_dict['rpv_steel']

    # Outer tank (cold coolant between core and RPV)
    outer_tank_cell = openmc.Cell(name='outer_tank')
    outer_tank_cell.region = +cyl_core & -cyl_outer_tank & +plane_fuel_bottom & -plane_fuel_top
    # Use appropriate coolant material based on assembly type
    if inputs['assembly_type'] == 'candu':
        outer_tank_cell.fill = mat_dict['candu_moderator']
    else:
        outer_tank_cell.fill = mat_dict['ap_1000_coolant_outer']

    # RPV Layer 1
    rpv_1_cell = openmc.Cell(name='rpv_layer_1')
    rpv_1_cell.region = +cyl_outer_tank & -cyl_rpv_1 & +plane_fuel_bottom & -plane_fuel_top
    rpv_1_cell.fill = mat_dict['rpv_steel']

    # RPV Layer 2
    rpv_2_cell = openmc.Cell(name='rpv_layer_2')
    rpv_2_cell.region = +cyl_rpv_1 & -cyl_rpv_2 & +plane_fuel_bottom & -plane_fuel_top
    rpv_2_cell.fill = mat_dict['rpv_steel']

    # Build list of cells for root universe
    root_cells = [
        bottom_reflector_cell,
        fuel_region_cell,
        outer_tank_cell,
        rpv_1_cell,
        rpv_2_cell
    ]

    # Add moderator region and wall divider if enabled
    if inputs['enable_moderator_region']:
        # Moderator region
        moderator_mat_name = inputs['moderator_material']
        if moderator_mat_name not in mat_dict:
            raise ValueError(f"Invalid moderator_material: {moderator_mat_name}. Must be one of {list(mat_dict.keys())}")

        moderator_cell = openmc.Cell(name='moderator_region')
        moderator_cell.region = +cyl_rpv_2 & -cyl_moderator & +plane_fuel_bottom & -plane_fuel_top
        moderator_cell.fill = mat_dict[moderator_mat_name]
        root_cells.append(moderator_cell)

        # Wall divider (between moderator and lithium)
        wall_divider_cell = openmc.Cell(name='wall_divider')
        wall_divider_cell.region = +cyl_moderator & -cyl_wall_divider & +plane_fuel_bottom & -plane_fuel_top
        wall_divider_cell.fill = mat_dict['rpv_steel']
        root_cells.append(wall_divider_cell)

        # Breeder blanket (starts at wall_divider when moderator is enabled)
        breeder_cell = openmc.Cell(name='breeder_blanket')
        breeder_cell.region = +cyl_wall_divider & -cyl_lithium & +plane_fuel_bottom & -plane_fuel_top
        breeder_cell.fill = breeder_mat
    else:
        # Breeder blanket (starts at RPV_2 when moderator is disabled)
        breeder_cell = openmc.Cell(name='breeder_blanket')
        breeder_cell.region = +cyl_rpv_2 & -cyl_lithium & +plane_fuel_bottom & -plane_fuel_top
        breeder_cell.fill = breeder_mat

    root_cells.append(breeder_cell)

    # Breeder containment wall
    breeder_wall_cell = openmc.Cell(name='breeder_wall')
    breeder_wall_cell.region = +cyl_lithium & -cyl_lithium_wall & +plane_fuel_bottom & -plane_fuel_top
    breeder_wall_cell.fill = mat_dict['rpv_steel']
    root_cells.append(breeder_wall_cell)

    # Top reflector
    top_reflector_cell = openmc.Cell(name='top_reflector')
    top_reflector_cell.region = -cyl_lithium_wall & +plane_fuel_top & -plane_top
    top_reflector_cell.fill = mat_dict['rpv_steel']
    root_cells.append(top_reflector_cell)

    # Create root universe and geometry
    root_universe = openmc.Universe(cells=root_cells)

    geometry = openmc.Geometry(root_universe)

    # Add tritium breeder assembly info to surfaces_dict
    surfaces_dict['tritium_info'] = tritium_info

    return geometry, surfaces_dict


if __name__ == '__main__':
    from materials import make_materials

    mat_dict = make_materials()
    mat_colors = {mat_dict[name]: color for name, color in MATERIAL_COLORS.items()}

    print("Creating geometry components...")
    if inputs['assembly_type'] == 'candu':
        # For CANDU, create a pin from the first ring
        pin_universe = CANDU_pin(mat_dict, 'candu_fuel_central')
        assembly_universe = CANDU_assembly(mat_dict)
    elif inputs['assembly_type'] == 'sodium':
        # For SFR, create MOX fuel pin and assembly
        pin_universe = SFR_pin(mat_dict, 'mox_inner')
        assembly_universe = SFR_assembly(mat_dict, 'mox_inner')
    else:
        pin_universe = BEAVRS_pin(mat_dict)
        assembly_universe = BEAVRS_assembly(mat_dict)

    # Check if we have tritium assemblies in lattice
    if inputs['assembly_type'] == 'sodium':
        # SFR always has tritium breeder in center
        has_tritium = True
        tritium_universe, _, _ = SFR_tritium_breeder_assembly(mat_dict)
    else:
        core_lattice = inputs['candu_lattice'] if inputs['assembly_type'] == 'candu' else inputs['ap1000_lattice']
        has_tritium = any('T' in str(symbol) for row in core_lattice for symbol in row)
        # Create tritium breeder assembly only if present in lattice
        if has_tritium:
            tritium_universe, _, _ = tritium_breeder_assembly(mat_dict)

    core_geometry, surfaces_dict = create_core(mat_dict)

    print("\nGeometry created successfully!")
    print(f"Returned surfaces: {list(surfaces_dict.keys())}")

    # Create figures directory if it doesn't exist
    figures_dir = Path('geometry_figures')
    figures_dir.mkdir(exist_ok=True)
    print(f"\nSaving plots to '{figures_dir}' directory...")

    derived = get_derived_dimensions()

    # Plot pin
    print("\nPlotting fuel pin...")
    plot_params = {
        'basis': 'xy',
        'width': (2.0, 2.0),
        'pixels': inputs['plot_pixels'],
        'color_by': 'material',
        'colors': mat_colors
    }
    pin_plot = pin_universe.plot(**plot_params)
    pin_plot.figure.set_size_inches(6, 6)
    pin_plot.figure.savefig(figures_dir / 'pin_xy.png', dpi=inputs['plot_dpi'], bbox_inches='tight')
    plt.close()

    # Plot assembly
    print("Plotting assembly...")
    # Use appropriate width based on assembly type
    if inputs['assembly_type'] == 'candu':
        assembly_width = 2 * inputs['candu_moderator_or']  # ~28.6 cm
        plot_width = (assembly_width * 1.2, assembly_width * 1.2)  # Add some margin
    elif inputs['assembly_type'] == 'sodium':
        assembly_width = 2 * inputs['sfr_assembly_edge']  # Hexagonal diameter
        plot_width = (assembly_width * 1.2, assembly_width * 1.2)
    else:
        plot_width = (25.0, 25.0)

    assembly_params = {
        'basis': 'xy',
        'width': plot_width,
        'pixels': inputs['plot_pixels'],
        'color_by': 'material',
        'colors': mat_colors
    }
    assembly_plot = assembly_universe.plot(**assembly_params)
    assembly_plot.figure.set_size_inches(8, 8)
    assembly_plot.figure.savefig(figures_dir / 'assembly_xy.png', dpi=inputs['plot_dpi'], bbox_inches='tight')
    plt.close()

    # Plot tritium breeder assembly (if present)
    if has_tritium:
        print("Plotting tritium breeder assembly...")
        tritium_params = {
            'basis': 'xy',
            'width': plot_width,
            'pixels': inputs['plot_pixels'],
            'color_by': 'material',
            'colors': mat_colors
        }
        tritium_plot = tritium_universe.plot(**tritium_params)
        tritium_plot.figure.set_size_inches(8, 8)
        tritium_plot.figure.savefig(figures_dir / 'tritium_assembly_xy.png', dpi=inputs['plot_dpi'], bbox_inches='tight')
        plt.close()

    # Plot core XY
    print("Plotting core XY...")
    # Auto-adjust width based on assembly type
    if inputs['assembly_type'] == 'sodium':
        # For SFR, use hex core edge
        core_xy_width = 2 * inputs['sfr_core_edge'] * 1.2  # Add 20% margin for hex
    else:
        # For cylindrical cores, use lithium wall radius
        r_max = derived['r_lithium_wall']
        core_xy_width = 2 * r_max * 1.1  # Add 10% margin

    # Calculate origin z-coordinate based on assembly type
    if inputs['assembly_type'] == 'sodium':
        origin_z = 0.0  # SFR is centered at z=0
    else:
        origin_z = derived['z_fuel_bottom'] + inputs['fuel_height']/2

    core_xy_params = {
        'basis': 'xy',
        'origin': (0, 0, origin_z),
        'width': (core_xy_width, core_xy_width),
        'pixels': inputs['plot_pixels'],
        'color_by': 'material',
        'colors': mat_colors
    }
    core_xy_plot = core_geometry.root_universe.plot(**core_xy_params)
    core_xy_plot.figure.set_size_inches(10, 10)
    core_xy_plot.figure.savefig(figures_dir / 'core_xy.png', dpi=inputs['plot_dpi'], bbox_inches='tight')
    plt.close()

    # Plot core XZ
    print("Plotting core XZ...")

    if inputs['assembly_type'] == 'sodium':
        # For SFR, slice through center (tritium breeder)
        x_slice = 0.0
        y_slice = 0.0
        core_xz_width = 2 * inputs['sfr_core_edge'] * 1.2
        z_center = 0.0
        z_height = inputs['sfr_axial_height'] * 2 * 1.1
    else:
        # Calculate x and y coordinates to slice through middle of a fuel assembly
        # Select appropriate lattice based on assembly type
        if inputs['assembly_type'] == 'candu':
            core_lattice = inputs['candu_lattice']
        else:
            core_lattice = inputs['ap1000_lattice']

        n_rows = len(core_lattice)
        n_cols = len(core_lattice[0])

        # Find center of a fuel assembly in the lattice
        # Look for a fuel assembly closest to the center (0, 0)
        fuel_assembly_pos = None
        min_distance = float('inf')

        for row_idx, row in enumerate(core_lattice):
            for col_idx, symbol in enumerate(row):
                if symbol == 'F':
                    # Calculate center position of this assembly
                    x_center = (col_idx - n_cols/2 + 0.5) * derived['assembly_width']
                    y_center = (row_idx - n_rows/2 + 0.5) * derived['assembly_width']
                    # Calculate distance from center
                    distance = (x_center**2 + y_center**2)**0.5
                    # Keep the assembly closest to center
                    if distance < min_distance:
                        min_distance = distance
                        fuel_assembly_pos = (x_center, y_center)

        # Default to (0, 0) if no fuel assembly found or for CANDU (which is centered)
        if fuel_assembly_pos is None or inputs['assembly_type'] == 'candu':
            x_slice = 0.0  # Center of CANDU assembly
            y_slice = 0.0  # Center of CANDU assembly
        else:
            x_slice = fuel_assembly_pos[0]  # X-coordinate of assembly center
            y_slice = fuel_assembly_pos[1]  # Y-coordinate to slice through assembly center

        # Auto-adjust width based on lithium wall radius
        core_xz_width = 2 * r_max * 1.1  # Add 10% margin
        z_center = derived['z_top']/2
        z_height = derived['z_top']*1.1

    core_xz_params = {
        'basis': 'xz',
        'origin': (x_slice, y_slice, z_center),
        'width': (core_xz_width, z_height),
        'pixels': inputs['plot_pixels'],
        'color_by': 'material',
        'colors': mat_colors
    }
    core_xz_plot = core_geometry.root_universe.plot(**core_xz_params)
    core_xz_plot.figure.set_size_inches(10, 12)
    core_xz_plot.figure.savefig(figures_dir / 'core_xz.png', dpi=inputs['plot_dpi'], bbox_inches='tight')
    plt.close()
