"""
Input parameters for PWR Fusion Breeder Reactor
"""
import numpy as np

# Core configuration dictionary
inputs = {
    ###########################################
    # Core Configuration
    ###########################################
    # Assembly type: 'candu', 'ap1000', or 'sodium'
    'assembly_type': 'sodium',  # Toggle between CANDU, AP1000, and Sodium Fast Reactor (SFR)

    'core_power': 800.0,  # MW

    # CANDU Core Layout - 'C' = coolant, 'F' = fuel assembly, 'T_1' = tritium breeder
    # 24x24 grid with buffer row/column of 'C' on all sides
    'candu_lattice': [
        ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C'],
        ['C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'T_1', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C'],
        ['C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C'],
        ['C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
        ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
    ],

    # AP1000 Core Layout - 'C' = coolant, 'F' = fuel assembly
    'ap1000_lattice': [
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
    # CANDU Assembly Configuration
    ###########################################
    # CANDU temperatures (K)
    'candu_T_fuel': 973.15,      # Fuel temperature
    'candu_T_clad': 563.16,      # Cladding temperature
    'candu_T_mod': 343.15,       # Moderator temperature
    'candu_T_cool': 563.15,      # Coolant temperature
    'candu_T_box': 544.16,       # Calandria/pressure tube temperature

    # CANDU fuel pin dimensions (cm)
    'candu_r_fuel': 0.59825,     # Fuel outer radius
    'candu_clad_thickness': 0.605,  # Cladding thickness (from fuel surface)
    'candu_r_clad': 0.655,      # Cladding outer radius

    # CANDU bundle geometry (cm)
    'candu_pressure_tube_ir': 5.1689,    # Pressure tube inner radius
    'candu_pressure_tube_or': 5.621,     # Pressure tube outer radius
    'candu_calandria_ir': 6.6002,        # Calandria tube inner radius
    'candu_calandria_or': 6.7526,       # Calandria tube outer radius
    'candu_moderator_or': 14.2875,      # Moderator outer radius

    # CANDU pin layout - ring radii from center (cm)
    'candu_ring_radii': [0.0, 1.49, 2.875, 4.333],
    'candu_num_pins': [1, 6, 12, 18],   # Number of pins per ring
    'candu_ring_angles': [0, 30, 15, 30],  # Starting angles for each ring (degrees)

    # CANDU assembly pitch (spacing between assembly centers in core lattice)
    # If not specified, defaults to 2 * candu_moderator_or
    'candu_assembly_pitch': 28.575,  # cm

    # CANDU fuel compositions (different rings use different fuel materials)
    'candu_fuel_density': 10.65,  # g/cm³
    'candu_U235_enrichment': 0.00711,  # Atomic fraction (0.711%)
    'candu_U238_fraction': 0.99289,     # Atomic fraction

    ###########################################
    # Sodium Fast Reactor (SFR) Configuration
    ###########################################
    # SFR temperatures (K)
    'sfr_T_fuel': 900.0,         # Fuel temperature
    'sfr_T_clad': 700.0,         # Cladding temperature
    'sfr_T_sodium': 700.0,       # Sodium coolant temperature

    # SFR fuel pin dimensions (cm)
    'sfr_fuel_or': 0.943/2,      # Fuel outer radius (0.4715 cm)
    'sfr_gap_ir': 0.943/2,       # Gap inner radius
    'sfr_gap_or': 0.973/2,       # Gap outer radius (0.4865 cm)
    'sfr_clad_ir': 0.973/2,      # Cladding inner radius
    'sfr_clad_or': 1.073/2,      # Cladding outer radius (0.5365 cm)

    # SFR pin assembly geometry (hexagonal)
    'sfr_pin_pitch': 21.08/17,   # Pin pitch in cm (~1.24 cm)
    'sfr_pins_per_ring': [48, 42, 36, 30, 24, 18, 12, 6, 1],  # Number of pins per ring
    'sfr_assembly_edge': 12.1705,  # Assembly hexagonal edge length (cm)

    # SFR core lattice (hexagonal rings)
    # Number of assemblies per ring: [96, 90, 84, ...] for reflector, outer fuel, inner fuel
    'sfr_reflector_rings': 3,     # Number of reflector rings (outermost)
    'sfr_outer_fuel_rings': 3,    # Number of outer fuel rings
    'sfr_inner_fuel_rings': 9,    # Number of inner fuel rings

    # SFR assembly pitch (distance between assembly centers)
    'sfr_assembly_pitch': 21.08,  # cm

    # SFR core outer boundary
    'sfr_core_edge': 347.82,      # Hexagonal edge length of entire core (cm)
    'sfr_ss316_wall_thickness': 20.0,  # SS316 wall thickness outside core (cm)

    # SFR axial dimensions (cm)
    'sfr_axial_height': 100.0,    # Active fuel height (half-core from z=0)
    'sfr_axial_reflector_thickness': 30.0,  # Axial reflector thickness (top and bottom, SS316)

    # SFR tritium breeder (scaled to fit in hexagonal assembly)
    'sfr_tritium_breeder_edge': 12.1705,  # Hexagonal edge for tritium assembly (same as fuel assembly to fill hexagon)

    ###########################################
    # Radial Geometry (all in cm)
    ###########################################
    'r_core': 340,            # Core radius (fuel + outer coolant)
    'outer_tank_thickness': 60,  # Outer tank region (cold coolant)
    'rpv_thickness_1': 15,     # Inner RPV layer (liner)
    'rpv_thickness_2': 15,    # Outer RPV layer (main vessel)
    'lithium_thickness': 0.1,  # Lithium breeding blanket
    'lithium_wall_thickness': 10.0,  # Lithium containment wall

    # Moderator region configuration (after RPV, before lithium)
    'enable_moderator_region': False,  # Toggle to enable/disable moderator region
    'moderator_thickness': 7.5,  # Thickness of moderator region (cm)
    'moderator_material': 'helium_moderator',  # Material for moderator region (options: 'heavy_water', 'ap_1000_coolant_outer', etc.)
    'wall_divider_thickness': 2.0,  # Thickness of wall divider between moderator and lithium (cm)

    ###########################################
    # Axial Geometry (all in cm)
    ###########################################
    'bottom_reflector_thickness': 40.0,  # Bottom reflector
    'fuel_height': 594.4,                # Active fuel height
    'top_reflector_thickness': 40.0,     # Top reflector

    ###########################################
    # Material Configuration
    ###########################################
    # Breeder material selection:
    # Options: 'natural_lithium', 'enriched_lithium',
    #          'natural_flibe', 'enriched_flibe',
    #          'natural_pbli', 'enriched_pbli', 'double_enriched_pbli'
    'breeder_material': 'enriched_lithium',

    # Coolant temperatures and pressure
    'T_hot_celsius': 315.0,     # Hot coolant temp (inside assemblies)
    'T_cold_celsius': 290.0,    # Cold coolant temp (outer core)
    'coolant_pressure_mpa': 15.5,  # MPa

    ###########################################
    # Tally Configuration
    ###########################################
    # Full reactor flux mesh discretization
    'n_radial_bins': 1000,        # Number of radial bins
    'n_axial_bins': 200,         # Number of axial bins

    # Energy group definitions (eV)
    'thermal_cutoff': 0.625,    # Thermal/epithermal boundary
    'epithermal_cutoff': 100e3, # Epithermal/fast boundary (100 keV)
    'fast_cutoff': 3e6,         # Fast upper limit (3 MeV)
    'very_fast_cutoff': 20e6,   # Very-fast upper limit (20 MeV)

    # LOG_1001 energy bins
    'log_1001_bins': np.logspace(np.log10(1e-5), np.log10(20.0e6), 1001),

    ###########################################
    # Simulation Settings
    ###########################################
    'batches': int(250),             # Total batches
    'inactive': int(50),             # Inactive batches
    'particles': int(10000),         # Particles per batch
    # Maximum number of particle events (OpenMC default: 1,000,000)
    'max_particle_events': 1000000000,

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

    # Assembly width/pitch (cm) - depends on assembly type
    if inputs['assembly_type'] == 'candu':
        # For CANDU, use specified pitch or default to 2 * moderator_or
        if inputs.get('candu_assembly_pitch') is not None:
            derived['assembly_width'] = inputs['candu_assembly_pitch']
        else:
            # Default: use the moderator outer radius as the assembly size
            derived['assembly_width'] = 2 * inputs['candu_moderator_or']
    elif inputs['assembly_type'] == 'sodium':
        # SFR uses hexagonal assembly pitch
        derived['assembly_width'] = inputs['sfr_assembly_pitch']
    else:
        # AP1000 uses lattice-based assembly
        derived['assembly_width'] = inputs['n_pins'] * inputs['pin_pitch']

    # Total radial dimensions
    derived['r_outer_tank'] = inputs['r_core'] + inputs['outer_tank_thickness']
    derived['r_rpv_1'] = derived['r_outer_tank'] + inputs['rpv_thickness_1']
    derived['r_rpv_2'] = derived['r_rpv_1'] + inputs['rpv_thickness_2']

    # Handle moderator region (if enabled)
    if inputs['enable_moderator_region']:
        derived['r_moderator'] = derived['r_rpv_2'] + inputs['moderator_thickness']
        derived['r_wall_divider'] = derived['r_moderator'] + inputs['wall_divider_thickness']
        derived['r_lithium'] = derived['r_wall_divider'] + inputs['lithium_thickness']
    else:
        derived['r_moderator'] = derived['r_rpv_2']  # Not used when disabled
        derived['r_wall_divider'] = derived['r_rpv_2']  # Not used when disabled
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

    # Four-group energy bins [thermal, epithermal, fast, very-fast]
    derived['four_group_bins'] = [
        0.0,
        inputs['thermal_cutoff'],       # 0.625 eV
        inputs['epithermal_cutoff'],    # 100 keV
        inputs['fast_cutoff'],          # 3 MeV
        inputs['very_fast_cutoff']      # 20 MeV (upper limit)
    ]

    # --- Power density calculation in kW/L (= MW/m^3) ---
    # Select appropriate lattice based on assembly type
    if inputs['assembly_type'] == 'candu':
        core_lattice = inputs['candu_lattice']
        n_fuel_assemblies = sum(row.count('F') for row in core_lattice)
        assembly_width_m = derived['assembly_width'] / 100.0 # (cm to m)
        assembly_height_m = inputs['fuel_height'] / 100.0
        assembly_volume_m3 = assembly_width_m * assembly_width_m * assembly_height_m
        total_fuelvol_m3 = n_fuel_assemblies * assembly_volume_m3
    elif inputs['assembly_type'] == 'sodium':
        # SFR: count fuel assemblies from ring configuration
        # Total assemblies = sum of inner and outer fuel rings
        # Each ring i has 6*i assemblies (hexagonal geometry)
        n_fuel_assemblies = 0
        total_rings = inputs['sfr_inner_fuel_rings'] + inputs['sfr_outer_fuel_rings']
        for i in range(1, total_rings + 1):
            n_fuel_assemblies += 6 * i
        # Center assembly
        n_fuel_assemblies += 1

        # For hexagonal assembly, use pitch and height
        assembly_width_m = derived['assembly_width'] / 100.0 # (cm to m)
        assembly_height_m = inputs['sfr_axial_height'] * 2 / 100.0  # Full height (2x half-height)
        # Hexagonal volume approximation
        assembly_volume_m3 = assembly_width_m * assembly_width_m * assembly_height_m * 0.866  # hex factor
        total_fuelvol_m3 = n_fuel_assemblies * assembly_volume_m3
    else:
        # AP1000
        core_lattice = inputs['ap1000_lattice']
        n_fuel_assemblies = sum(row.count('F') for row in core_lattice)
        assembly_width_m = derived['assembly_width'] / 100.0 # (cm to m)
        assembly_height_m = inputs['fuel_height'] / 100.0
        assembly_volume_m3 = assembly_width_m * assembly_width_m * assembly_height_m
        total_fuelvol_m3 = n_fuel_assemblies * assembly_volume_m3

    derived['power_density_kW_per_L'] = inputs['core_power'] / total_fuelvol_m3 if total_fuelvol_m3 > 0 else float('nan')

    derived['n_fuel_assemblies'] = n_fuel_assemblies
    derived['fuel_assembly_volume_m3'] = assembly_volume_m3
    derived['total_fuel_assembly_volume_m3'] = total_fuelvol_m3

    return derived
