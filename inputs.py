"""
Input parameters for PWR Fusion Breeder Reactor
"""
import numpy as np

# Core configuration dictionary
inputs = {
    ###########################################
    # Core Configuration
    ###########################################
    'core_power': 800.0,  # MW

    # Core Layout - 'C' = coolant, 'F' = fuel assembly
    'core_lattice': [
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'C', 'F', 'F', 'F', 'F', 'C', 'C'],
    ],

    ###########################################
    # Fuel Assembly Configuration
    ###########################################
    'n_pins': 17,              # 17x17 assembly
    'pin_pitch': 1.26,         # cm

    # Pin dimensions (cm)
    'fuel_or': 0.4096,         # Fuel outer radius
    'clad_ir': 0.418,          # Cladding inner radius
    'clad_or': 0.475,          # Cladding outer radius

    # Guide tube dimensions (cm)
    'gt_ir': 0.561,            # Guide tube inner radius
    'gt_or': 0.602,            # Guide tube outer radius

    # Guide tube positions in 17x17 assembly
    'guide_tube_positions': [
        (2, 5), (2, 8), (2, 11),
        (3, 3), (3, 13),
        (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
        (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
        (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
        (13, 3), (13, 13),
        (14, 5), (14, 8), (14, 11)
    ],

    ###########################################
    # Radial Geometry (all in cm)
    ###########################################
    'r_core': 100.0,            # Core radius (fuel + outer coolant)
    'outer_tank_thickness': 0.0,  # Outer tank region (cold coolant)
    'rpv_thickness_1': 25.0,     # Inner RPV layer (liner)
    'rpv_thickness_2': 5.0,    # Outer RPV layer (main vessel)
    'lithium_thickness': 30.0,  # Lithium breeding blanket
    'lithium_wall_thickness': 10.0,  # Lithium containment wall

    ###########################################
    # Axial Geometry (all in cm)
    ###########################################
    'bottom_reflector_thickness': 40.0,  # Bottom reflector
    'fuel_height': 400.0,                # Active fuel height
    'top_reflector_thickness': 40.0,     # Top reflector

    ###########################################
    # Material Configuration
    ###########################################
    'lithium_type': 'natural',  # 'natural' or 'enriched'

    # Coolant temperatures and pressure
    'T_hot_celsius': 315.0,     # Hot coolant temp (inside assemblies)
    'T_cold_celsius': 290.0,    # Cold coolant temp (outer core)
    'coolant_pressure_mpa': 15.5,  # MPa

    ###########################################
    # Tally Configuration
    ###########################################
    # Full reactor flux mesh discretization
    'n_radial_bins': 200,        # Number of radial bins
    'n_axial_bins': 200,         # Number of axial bins

    # Energy group definitions (eV)
    'thermal_cutoff': 0.625,    # Thermal/epithermal boundary
    'epithermal_cutoff': 100e3, # Epithermal/fast boundary (100 keV)
    'fast_cutoff': 10e6,        # Fast upper limit (10 MeV)

    # LOG_1001 energy bins
    'log_1001_bins': np.logspace(np.log10(1e-5), np.log10(20.0e6), 1001),

    ###########################################
    # Simulation Settings
    ###########################################
    'batches': 100,             # Total batches
    'inactive': 20,             # Inactive batches
    'particles': 10000,         # Particles per batch

    # Entropy mesh for source convergence
    'entropy_mesh_dimension': [20, 20, 20],

    ###########################################
    # Plotting Configuration
    ###########################################
    'plot_pixels': (3000, 2500),  # Resolution for plots
    'plot_dpi': 300,               # DPI for saved images
}


def get_derived_dimensions():
    derived = {}

    # Assembly width (cm)
    derived['assembly_width'] = inputs['n_pins'] * inputs['pin_pitch']

    # Total radial dimensions
    derived['r_outer_tank'] = inputs['r_core'] + inputs['outer_tank_thickness']
    derived['r_rpv_1'] = derived['r_outer_tank'] + inputs['rpv_thickness_1']
    derived['r_rpv_2'] = derived['r_rpv_1'] + inputs['rpv_thickness_2']
    derived['r_lithium'] = derived['r_rpv_2'] + inputs['lithium_thickness']
    derived['r_lithium_wall'] = derived['r_lithium'] + inputs['lithium_wall_thickness']

    # Total axial dimensions
    derived['z_bottom'] = 0.0
    derived['z_fuel_bottom'] = derived['z_bottom'] + inputs['bottom_reflector_thickness']
    derived['z_fuel_top'] = derived['z_fuel_bottom'] + inputs['fuel_height']
    derived['z_top'] = derived['z_fuel_top'] + inputs['top_reflector_thickness']

    # Three-group energy bins [thermal, epithermal, fast]
    derived['three_group_bins'] = [
        0.0,
        inputs['thermal_cutoff'],
        inputs['epithermal_cutoff'],
        inputs['fast_cutoff']
    ]

    # --- Power density calculation in kW/L (= MW/m^3) ---
    n_fuel_assemblies = sum(row.count('F') for row in inputs['core_lattice'])
    assembly_width_m = derived['assembly_width'] / 100.0 # (cm to m)
    assembly_height_m = inputs['fuel_height'] / 100.0
    assembly_volume_m3 = assembly_width_m * assembly_width_m * assembly_height_m
    total_fuelvol_m3 = n_fuel_assemblies * assembly_volume_m3
    derived['power_density_kW_per_L'] = inputs['core_power'] / total_fuelvol_m3 if total_fuelvol_m3 > 0 else float('nan')

    derived['n_fuel_assemblies'] = n_fuel_assemblies
    derived['fuel_assembly_volume_m3'] = assembly_volume_m3
    derived['total_fuel_assembly_volume_m3'] = total_fuelvol_m3

    return derived
