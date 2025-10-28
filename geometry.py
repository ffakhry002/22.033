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
    'double_enriched_pbli': 'slategray'
}

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

    fuel_assembly = BEAVRS_assembly(mat_dict)

    # Outer coolant universe (cold coolant)
    coolant_outer_universe = openmc.Universe(name='coolant_outer')
    coolant_cell = openmc.Cell(fill=mat_dict['ap_1000_coolant_outer'])
    coolant_outer_universe.add_cell(coolant_cell)

    # Build lattice universes array
    core_lattice = inputs['core_lattice']
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
    outer_tank_cell.fill = mat_dict['ap_1000_coolant_outer']

    # RPV Layer 1
    rpv_1_cell = openmc.Cell(name='rpv_layer_1')
    rpv_1_cell.region = +cyl_outer_tank & -cyl_rpv_1 & +plane_fuel_bottom & -plane_fuel_top
    rpv_1_cell.fill = mat_dict['rpv_steel']

    # RPV Layer 2
    rpv_2_cell = openmc.Cell(name='rpv_layer_2')
    rpv_2_cell.region = +cyl_rpv_1 & -cyl_rpv_2 & +plane_fuel_bottom & -plane_fuel_top
    rpv_2_cell.fill = mat_dict['rpv_steel']

    # Breeder blanket
    breeder_cell = openmc.Cell(name='breeder_blanket')
    breeder_cell.region = +cyl_rpv_2 & -cyl_lithium & +plane_fuel_bottom & -plane_fuel_top
    breeder_cell.fill = breeder_mat

    # Breeder containment wall
    breeder_wall_cell = openmc.Cell(name='breeder_wall')
    breeder_wall_cell.region = +cyl_lithium & -cyl_lithium_wall & +plane_fuel_bottom & -plane_fuel_top
    breeder_wall_cell.fill = mat_dict['rpv_steel']

    # Top reflector
    top_reflector_cell = openmc.Cell(name='top_reflector')
    top_reflector_cell.region = -cyl_lithium_wall & +plane_fuel_top & -plane_top
    top_reflector_cell.fill = mat_dict['rpv_steel']

    # Create root universe and geometry
    root_universe = openmc.Universe(cells=[
        bottom_reflector_cell,
        fuel_region_cell,
        outer_tank_cell,
        rpv_1_cell,
        rpv_2_cell,
        breeder_cell,
        breeder_wall_cell,
        top_reflector_cell
    ])

    geometry = openmc.Geometry(root_universe)

    return geometry, surfaces_dict


if __name__ == '__main__':
    from materials import make_materials

    mat_dict = make_materials()
    mat_colors = {mat_dict[name]: color for name, color in MATERIAL_COLORS.items()}

    print("Creating geometry components...")
    pin_universe = BEAVRS_pin(mat_dict)
    assembly_universe = BEAVRS_assembly(mat_dict)
    core_geometry, surfaces_dict = create_core(mat_dict)

    print("\nGeometry created successfully!")
    print(f"Returned surfaces: {list(surfaces_dict.keys())}")

    # Create figures directory if it doesn't exist
    figures_dir = Path('figures')
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
    assembly_params = {
        'basis': 'xy',
        'width': (25.0, 25.0),
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
    core_xy_params = {
        'basis': 'xy',
        'origin': (0, 0, derived['z_fuel_bottom'] + inputs['fuel_height']/2),
        'width': (400.0, 400.0),
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
    core_xz_params = {
        'basis': 'xz',
        'origin': (0, 1, derived['z_top']/2),
        'width': (400.0, derived['z_top']*1.1),
        'pixels': inputs['plot_pixels'],
        'color_by': 'material',
        'colors': mat_colors
    }
    core_xz_plot = core_geometry.root_universe.plot(**core_xz_params)
    core_xz_plot.figure.set_size_inches(10, 12)
    core_xz_plot.figure.savefig(figures_dir / 'core_xz.png', dpi=inputs['plot_dpi'], bbox_inches='tight')
    plt.close()
