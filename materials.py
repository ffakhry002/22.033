"""
Materials for PWR Fusion Breeder Reactor
"""
import openmc
from CoolProp.CoolProp import PropsSI
from inputs import inputs


def make_materials():
    materials_list = []

    # ====== UO2 FUEL ======
    uo2 = openmc.Material(name='ap_1000_fuel')
    uo2.add_element('U', 1.0, enrichment=4.5)  # 4.5% enriched
    uo2.add_element('O', 2.0)
    uo2.set_density('g/cm3', 10.3)
    uo2.temperature = 900.0  # K (typical fuel temperature)
    materials_list.append(uo2)

    # ====== HELIUM GAP ======
    he_gap = openmc.Material(name='ap_1000_gap')
    he_gap.add_element('He', 1.0)
    he_gap.set_density('g/cm3', 0.0018)
    he_gap.temperature = 600.0  # K
    materials_list.append(he_gap)

    # ====== ZIRCALOY CLADDING ======
    zircaloy = openmc.Material(name='ap_1000_cladding')
    zircaloy.add_element('Zr', 0.9821, percent_type='wo')
    zircaloy.add_element('Sn', 0.0145, percent_type='wo')
    zircaloy.add_element('Fe', 0.0021, percent_type='wo')
    zircaloy.add_element('Cr', 0.0010, percent_type='wo')
    zircaloy.add_element('O', 0.0003, percent_type='wo')
    zircaloy.set_density('g/cm3', 6.56)
    zircaloy.temperature = 600.0  # K
    materials_list.append(zircaloy)

    # ====== COOLANT: Light Water (Hot - Inside Assemblies) ======
    T_hot = inputs['T_hot_celsius'] + 273.15  # Convert to Kelvin
    P_coolant = inputs['coolant_pressure_mpa'] * 1e6  # Convert to Pa
    rho_hot = PropsSI('D', 'T', T_hot, 'P', P_coolant, 'Water')  # kg/m³

    coolant_hot = openmc.Material(name='ap_1000_coolant')
    coolant_hot.add_nuclide('H1', 2.0)
    coolant_hot.add_nuclide('O16', 1.0)
    coolant_hot.set_density('g/cm3', rho_hot / 1000)
    coolant_hot.add_s_alpha_beta('c_H_in_H2O')
    coolant_hot.temperature = T_hot
    materials_list.append(coolant_hot)

    # ====== COOLANT: Light Water (Cold - Outer Core) ======
    T_cold = inputs['T_cold_celsius'] + 273.15  # Convert to Kelvin
    rho_cold = PropsSI('D', 'T', T_cold, 'P', P_coolant, 'Water')  # kg/m³

    coolant_cold = openmc.Material(name='ap_1000_coolant_outer')
    coolant_cold.add_nuclide('H1', 2.0)
    coolant_cold.add_nuclide('O16', 1.0)
    coolant_cold.set_density('g/cm3', rho_cold / 1000)
    coolant_cold.add_s_alpha_beta('c_H_in_H2O')
    coolant_cold.temperature = T_cold
    materials_list.append(coolant_cold)

    # ====== RPV STEEL ======
    rpv_steel = openmc.Material(name='rpv_steel')
    rpv_steel.set_density('g/cm3', 7.9)
    rpv_steel.add_nuclide('Fe54', 0.05437098, 'wo')
    rpv_steel.add_nuclide('Fe56', 0.88500663, 'wo')
    rpv_steel.add_nuclide('Fe57', 0.0208008, 'wo')
    rpv_steel.add_nuclide('Fe58', 0.00282159, 'wo')
    rpv_steel.add_nuclide('Ni58', 0.0067198, 'wo')
    rpv_steel.add_nuclide('Ni60', 0.0026776, 'wo')
    rpv_steel.add_nuclide('Mn55', 0.01, 'wo')
    rpv_steel.add_nuclide('Cr52', 0.002092475, 'wo')
    rpv_steel.add_nuclide('C12', 0.0025, 'wo')  # Changed from C0
    rpv_steel.add_nuclide('Cu63', 0.0013696, 'wo')
    rpv_steel.temperature = 580.0  # K
    materials_list.append(rpv_steel)

    # ====== NATURAL LITHIUM BREEDER ======
    natural_lithium = openmc.Material(name='natural_lithium')
    natural_lithium.set_density('g/cm3', 0.512)
    natural_lithium.add_nuclide('Li6', 7.59, percent_type='ao')
    natural_lithium.add_nuclide('Li7', 92.41, percent_type='ao')
    natural_lithium.temperature = 773  # 500°C
    materials_list.append(natural_lithium)

    # ====== ENRICHED LITHIUM BREEDER (90% Li-6) ======
    enriched_lithium = openmc.Material(name='enriched_lithium')
    enriched_lithium.set_density('g/cm3', 0.512)
    enriched_lithium.add_nuclide('Li6', 90.0, percent_type='ao')
    enriched_lithium.add_nuclide('Li7', 10.0, percent_type='ao')
    enriched_lithium.temperature = 773  # 500°C
    materials_list.append(enriched_lithium)

    # Create materials collection
    materials = openmc.Materials(materials_list)

    # Create dictionary for easy access
    mat_dict = {mat.name: mat for mat in materials_list}

    return mat_dict


if __name__ == "__main__":
    """Test materials creation."""
    mat_dict = make_materials()

    print("Created materials:")
    for name in mat_dict:
        print(f"  - {name}")

    print(f"\nTotal: {len(mat_dict)} materials")
