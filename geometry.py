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

    # Create fuel pin surfaces
    surf_fuel = openmc.ZCylinder(r=r_fuel)
    clad_fuel = openmc.ZCylinder(r=clad_thickness)

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

            # Create fuel cell
            fuel_cell = openmc.Cell(fill=mat_dict[fuel_mat_name], region=-surf_fuel)
            fuel_cell.id = (i + 1) * 1000 + j
            fuel_cells.append(fuel_cell)

            # Create gap and cladding cells
            gap_cell = openmc.Cell(fill=mat_dict['candu_fuel_gap'], region=+surf_fuel & -clad_fuel)
            clad_cell = openmc.Cell(fill=mat_dict['candu_cladding'], region=+clad_fuel)
            pin_universe = openmc.Universe(cells=(fuel_cell, gap_cell, clad_cell))

            # Create pin cell
            pin = openmc.Cell(fill=pin_universe, region=-pin_boundary)
            pin.translation = (x, y, 0)
            pin.id = (i + 1)*100 + j
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
    moderator_or = inputs['candu_moderator_or']

    # Create surfaces
    pt_inner = openmc.ZCylinder(r=pressure_tube_ir)
    pt_outer = openmc.ZCylinder(r=pressure_tube_or)
    calandria_inner = openmc.ZCylinder(r=calandria_ir)
    calandria_outer = openmc.ZCylinder(r=calandria_or)
    moderator_outer = openmc.ZCylinder(r=moderator_or)

    # Create boundary planes (square boundary for moderator)
    half_width = moderator_or
    fuel_x0 = openmc.XPlane(x0=-half_width, boundary_type='reflective')
    fuel_x1 = openmc.XPlane(x0=half_width, boundary_type='reflective')
    fuel_y0 = openmc.YPlane(y0=-half_width, boundary_type='reflective')
    fuel_y1 = openmc.YPlane(y0=half_width, boundary_type='reflective')

    # Create cells
    bundle = openmc.Cell(fill=bundle_universe, region=-pt_inner)
    pressure_tube = openmc.Cell(fill=mat_dict['candu_pressure_tube'], region=+pt_inner & -pt_outer)
    gap_cell = openmc.Cell(fill=mat_dict['candu_gap'], region=+pt_outer & -calandria_inner)
    calandria = openmc.Cell(fill=mat_dict['candu_calandria_tube'], region=+calandria_inner & -calandria_outer)
    moder = openmc.Cell(fill=mat_dict['candu_moderator'], region=+calandria_outer & +fuel_x0 & -fuel_x1 & +fuel_y0 & -fuel_y1)

    root_universe = openmc.Universe(cells=[bundle, pressure_tube, gap_cell, calandria, moder])

    return root_universe


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


def build_core_lattice(mat_dict, cyl_core, plane_bottom, plane_top):
    """Build a lattice of assemblies based on core_lattice from inputs."""
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
            else:
                raise ValueError(f"Unknown core lattice symbol: {symbol}. Use 'C' or 'F'.")
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

    return lattice_cell


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

    # Build fuel region
    fuel_region_cell = build_core_lattice(mat_dict, cyl_core, plane_fuel_bottom, plane_fuel_top)

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
    else:
        pin_universe = BEAVRS_pin(mat_dict)
        assembly_universe = BEAVRS_assembly(mat_dict)
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
    # Use larger width for CANDU assemblies
    if inputs['assembly_type'] == 'candu':
        assembly_width = 2 * inputs['candu_moderator_or']  # ~28.6 cm
        plot_width = (assembly_width * 1.2, assembly_width * 1.2)  # Add some margin
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

    # Plot core XY
    print("Plotting core XY...")
    # Auto-adjust width based on lithium wall radius (add some margin)
    r_max = derived['r_lithium_wall']
    core_xy_width = 2 * r_max * 1.1  # Add 10% margin

    core_xy_params = {
        'basis': 'xy',
        'origin': (0, 0, derived['z_fuel_bottom'] + inputs['fuel_height']/2),
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

    core_xz_params = {
        'basis': 'xz',
        'origin': (x_slice, y_slice, derived['z_top']/2),
        'width': (core_xz_width, derived['z_top']*1.1),
        'pixels': inputs['plot_pixels'],
        'color_by': 'material',
        'colors': mat_colors
    }
    core_xz_plot = core_geometry.root_universe.plot(**core_xz_params)
    core_xz_plot.figure.set_size_inches(10, 12)
    core_xz_plot.figure.savefig(figures_dir / 'core_xz.png', dpi=inputs['plot_dpi'], bbox_inches='tight')
    plt.close()
