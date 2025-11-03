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

    # ====== FLiBe MOLTEN SALT BREEDER (Natural Lithium) ======
    natural_flibe = openmc.Material(name='natural_flibe')
    natural_flibe.set_density('g/cm3', 1.94)  # At 600°C
    natural_flibe.add_nuclide('Li6', 0.0759 * 2, percent_type='ao')  # Natural Li-6 in 2 Li atoms
    natural_flibe.add_nuclide('Li7', 0.9241 * 2, percent_type='ao')  # Natural Li-7 in 2 Li atoms
    natural_flibe.add_element('Be', 1.0, percent_type='ao')          # 1 Be atom
    natural_flibe.add_element('F', 4.0, percent_type='ao')           # 4 F atoms
    natural_flibe.temperature = 873  # 600°C operating temperature
    materials_list.append(natural_flibe)

    # ====== FLiBe MOLTEN SALT BREEDER (Enriched Lithium - 90% Li-6) ======
    enriched_flibe = openmc.Material(name='enriched_flibe')
    enriched_flibe.set_density('g/cm3', 1.94)  # At 600°C
    enriched_flibe.add_nuclide('Li6', 0.90 * 2, percent_type='ao')  # 90% enriched Li-6 in 2 Li atoms
    enriched_flibe.add_nuclide('Li7', 0.10 * 2, percent_type='ao')  # 10% Li-7 in 2 Li atoms
    enriched_flibe.add_element('Be', 1.0, percent_type='ao')        # 1 Be atom
    enriched_flibe.add_element('F', 4.0, percent_type='ao')         # 4 F atoms
    enriched_flibe.temperature = 873  # 600°C operating temperature
    materials_list.append(enriched_flibe)

    # ====== Pb-Li LIQUID BREEDER (Natural Lithium) ======
    natural_pbli = openmc.Material(name='natural_pbli')
    natural_pbli.set_density('g/cm3', 10.52)  # 10.520 kg/m³ = 10.52 g/cm³ from user
    natural_pbli.add_element('Pb', 83.0, percent_type='ao')          # 83 at% Lead (natural isotopes)
    natural_pbli.add_nuclide('Li6', 17.0 * 0.0759, percent_type='ao')  # Natural Li-6 in 17 at% Li
    natural_pbli.add_nuclide('Li7', 17.0 * 0.9241, percent_type='ao')  # Natural Li-7 in 17 at% Li
    natural_pbli.temperature = 673  # 400°C operating temperature
    materials_list.append(natural_pbli)

    # ====== Pb-Li LIQUID BREEDER (Enriched Lithium - 90% Li-6) ======
    enriched_pbli = openmc.Material(name='enriched_pbli')
    enriched_pbli.set_density('g/cm3', 10.52)
    enriched_pbli.add_element('Pb', 83.0, percent_type='ao')          # 83 at% Lead (natural isotopes)
    enriched_pbli.add_nuclide('Li6', 17.0 * 0.90, percent_type='ao')  # 90% enriched Li-6 in 17 at% Li
    enriched_pbli.add_nuclide('Li7', 17.0 * 0.10, percent_type='ao')  # 10% Li-7 in 17 at% Li
    enriched_pbli.temperature = 673  # 400°C operating temperature
    materials_list.append(enriched_pbli)

    # ====== Pb-Li LIQUID BREEDER (Double Enriched: Li-6 + Pb-208) ======
    double_enriched_pbli = openmc.Material(name='double_enriched_pbli')
    double_enriched_pbli.set_density('g/cm3', 10.52)
    double_enriched_pbli.add_nuclide('Pb208', 83.0, percent_type='ao')  # 83 at% Pb-208 (enriched)
    double_enriched_pbli.add_nuclide('Li6', 17.0 * 0.90, percent_type='ao')  # 90% enriched Li-6 in 17 at% Li
    double_enriched_pbli.add_nuclide('Li7', 17.0 * 0.10, percent_type='ao')  # 10% Li-7 in 17 at% Li
    double_enriched_pbli.temperature = 673  # 400°C operating temperature
    materials_list.append(double_enriched_pbli)

    # ====== HEAVY WATER (D2O) ======
    # Room temperature and atmospheric pressure
    T_d2o = 298.15  # 25°C in Kelvin
    P_d2o = 101325.0  # Atmospheric pressure in Pa
    rho_d2o = PropsSI('D', 'T', T_d2o, 'P', P_d2o, 'HeavyWater')  # kg/m³

    d2o = openmc.Material(name='heavy_water')
    d2o.add_nuclide('H2', 2.0)
    d2o.add_nuclide('O16', 1.0)
    d2o.set_density('g/cm3', rho_d2o / 1000)
    d2o.add_s_alpha_beta('c_D_in_D2O')
    d2o.temperature = T_d2o
    materials_list.append(d2o)


    # ====== LIGHT WATER (Room Temperature, Atmospheric Pressure) ======
    T_room = 298.15  # 25°C in Kelvin
    P_atm = 101325.0  # Atmospheric pressure in Pa
    rho_room = PropsSI('D', 'T', T_room, 'P', P_atm, 'Water')  # kg/m³

    room_temp_lightwater = openmc.Material(name='room_temp_lightwater')
    room_temp_lightwater.add_nuclide('H1', 2.0)
    room_temp_lightwater.add_nuclide('O16', 1.0)
    room_temp_lightwater.set_density('g/cm3', rho_room / 1000)
    room_temp_lightwater.add_s_alpha_beta('c_H_in_H2O')
    room_temp_lightwater.temperature = T_room
    materials_list.append(room_temp_lightwater)

    # ====== CANDU MATERIALS ======
    T_fuel = inputs['candu_T_fuel']
    T_clad = inputs['candu_T_clad']
    T_mod = inputs['candu_T_mod']
    T_cool = inputs['candu_T_cool']
    T_box = inputs['candu_T_box']

    # CANDU Fuel Materials (4 different rings)
    candu_fuel_central = openmc.Material(name='candu_fuel_central', temperature=T_fuel)
    candu_fuel_central.add_nuclide('U235', inputs['candu_U235_enrichment'], percent_type='ao')
    candu_fuel_central.add_nuclide('U238', inputs['candu_U238_fraction'], percent_type='ao')
    candu_fuel_central.add_nuclide('O16', 2.0, percent_type='ao')
    candu_fuel_central.set_density('g/cm3', inputs['candu_fuel_density'])
    materials_list.append(candu_fuel_central)

    candu_fuel_inner = openmc.Material(name='candu_fuel_inner', temperature=T_fuel)
    candu_fuel_inner.add_nuclide('U235', inputs['candu_U235_enrichment'], percent_type='ao')
    candu_fuel_inner.add_nuclide('U238', inputs['candu_U238_fraction'], percent_type='ao')
    candu_fuel_inner.add_nuclide('O16', 2.0, percent_type='ao')
    candu_fuel_inner.set_density('g/cm3', inputs['candu_fuel_density'])
    materials_list.append(candu_fuel_inner)

    candu_fuel_intermediate = openmc.Material(name='candu_fuel_intermediate', temperature=T_fuel)
    candu_fuel_intermediate.add_nuclide('U235', inputs['candu_U235_enrichment'], percent_type='ao')
    candu_fuel_intermediate.add_nuclide('U238', inputs['candu_U238_fraction'], percent_type='ao')
    candu_fuel_intermediate.add_nuclide('O16', 2.0, percent_type='ao')
    candu_fuel_intermediate.set_density('g/cm3', inputs['candu_fuel_density'])
    materials_list.append(candu_fuel_intermediate)

    candu_fuel_outer = openmc.Material(name='candu_fuel_outer', temperature=T_fuel)
    candu_fuel_outer.add_nuclide('U235', inputs['candu_U235_enrichment'], percent_type='ao')
    candu_fuel_outer.add_nuclide('U238', inputs['candu_U238_fraction'], percent_type='ao')
    candu_fuel_outer.add_nuclide('O16', 2.0, percent_type='ao')
    candu_fuel_outer.set_density('g/cm3', inputs['candu_fuel_density'])
    materials_list.append(candu_fuel_outer)

    # CANDU Zircaloy Cladding
    candu_cladding = openmc.Material(name='candu_cladding', temperature=T_clad)
    candu_cladding.add_nuclide('Fe56', 0.002, 'wo')
    candu_cladding.add_nuclide('Nb93', 0.01, 'wo')
    candu_cladding.add_nuclide('Zr90', 0.503181, 'wo')
    candu_cladding.add_nuclide('Zr91', 0.1097316, 'wo')
    candu_cladding.add_nuclide('Zr92', 0.167727, 'wo')
    candu_cladding.add_nuclide('Zr94', 0.1699764, 'wo')
    candu_cladding.add_nuclide('Zr96', 0.027384, 'wo')
    candu_cladding.add_nuclide('Sn112', 0.000097, 'wo')
    candu_cladding.add_nuclide('Sn114', 0.000066, 'wo')
    candu_cladding.add_nuclide('Sn115', 0.000034, 'wo')
    candu_cladding.add_nuclide('Sn116', 0.001454, 'wo')
    candu_cladding.add_nuclide('Sn117', 0.000768, 'wo')
    candu_cladding.add_nuclide('Sn118', 0.002422, 'wo')
    candu_cladding.add_nuclide('Sn119', 0.000859, 'wo')
    candu_cladding.add_nuclide('Sn120', 0.003258, 'wo')
    candu_cladding.add_nuclide('Sn122', 0.000463, 'wo')
    candu_cladding.add_nuclide('Sn124', 0.000579, 'wo')
    candu_cladding.set_density('g/cm3', 6.44)
    materials_list.append(candu_cladding)

    # CANDU Moderator (99.911% wt D2O)
    candu_moderator = openmc.Material(name='candu_moderator', temperature=T_mod)
    candu_moderator.add_nuclide('H2', 0.200954, 'wo')
    candu_moderator.add_nuclide('H1', 0.00009959, 'wo')
    candu_moderator.add_nuclide('O16', 0.7989464, 'wo')
    candu_moderator.add_s_alpha_beta('c_D_in_D2O')
    candu_moderator.add_s_alpha_beta('c_H_in_H2O')
    candu_moderator.set_density('g/cm3', 1.08524)
    materials_list.append(candu_moderator)

    # CANDU Coolant (99.222% wt D2O)
    candu_coolant = openmc.Material(name='candu_coolant', temperature=T_cool)
    candu_coolant.add_nuclide('H2', 0.199568, 'wo')
    candu_coolant.add_nuclide('H1', 0.0008705, 'wo')
    candu_coolant.add_nuclide('O16', 0.7995615, 'wo')
    candu_coolant.add_s_alpha_beta('c_D_in_D2O')
    candu_coolant.add_s_alpha_beta('c_H_in_H2O')
    candu_coolant.set_density('g/cm3', 0.811862)
    materials_list.append(candu_coolant)

    # CANDU Calandria Tube
    candu_calandria_tube = openmc.Material(name='candu_calandria_tube', temperature=T_box)
    candu_calandria_tube.add_nuclide('Sn112', 0.00016, 'wo')
    candu_calandria_tube.add_nuclide('Sn114', 0.00011, 'wo')
    candu_calandria_tube.add_nuclide('Sn115', 0.00006, 'wo')
    candu_calandria_tube.add_nuclide('Sn116', 0.00247, 'wo')
    candu_calandria_tube.add_nuclide('Sn117', 0.00131, 'wo')
    candu_calandria_tube.add_nuclide('Sn118', 0.00412, 'wo')
    candu_calandria_tube.add_nuclide('Sn119', 0.00146, 'wo')
    candu_calandria_tube.add_nuclide('Sn120', 0.00554, 'wo')
    candu_calandria_tube.add_nuclide('Sn122', 0.00079, 'wo')
    candu_calandria_tube.add_nuclide('Sn124', 0.00098, 'wo')
    candu_calandria_tube.add_nuclide('Fe56', 0.0024, 'wo')
    candu_calandria_tube.add_nuclide('Cr50', 0.00006, 'wo')
    candu_calandria_tube.add_nuclide('Cr52', 0.00109, 'wo')
    candu_calandria_tube.add_nuclide('Cr53', 0.00012, 'wo')
    candu_calandria_tube.add_nuclide('Cr54', 0.00003, 'wo')
    candu_calandria_tube.add_nuclide('Zr90', 0.5038, 'wo')
    candu_calandria_tube.add_nuclide('Zr91', 0.1099, 'wo')
    candu_calandria_tube.add_nuclide('Zr92', 0.1679, 'wo')
    candu_calandria_tube.add_nuclide('Zr94', 0.1702, 'wo')
    candu_calandria_tube.add_nuclide('Zr96', 0.0274, 'wo')
    candu_calandria_tube.set_density('g/cm3', 6.55)
    materials_list.append(candu_calandria_tube)

    # CANDU Pressure Tube
    candu_pressure_tube = openmc.Material(name='candu_pressure_tube', temperature=T_box)
    candu_pressure_tube.add_nuclide('Nb93', 0.025, 'wo')
    candu_pressure_tube.add_nuclide('C', 0.00027, 'wo')
    candu_pressure_tube.add_nuclide('Cr52', 0.000085, 'wo')
    candu_pressure_tube.add_nuclide('Cr53', 0.000009, 'wo')
    candu_pressure_tube.add_nuclide('Cr54', 0.000002, 'wo')
    candu_pressure_tube.add_nuclide('Fe56', 0.0015, 'wo')
    candu_pressure_tube.add_nuclide('Si29', 0.0001, 'wo')
    candu_pressure_tube.add_nuclide('Zr90', 0.50063, 'wo')
    candu_pressure_tube.add_nuclide('Zr91', 0.10917, 'wo')
    candu_pressure_tube.add_nuclide('Zr92', 0.16688, 'wo')
    candu_pressure_tube.add_nuclide('Zr94', 0.16911, 'wo')
    candu_pressure_tube.add_nuclide('Zr96', 0.027245, 'wo')
    candu_pressure_tube.set_density('g/cm3', 6.51)
    materials_list.append(candu_pressure_tube)

    # CANDU Gap (between pressure tube and calandria)
    candu_gap = openmc.Material(name='candu_gap')
    candu_gap.add_nuclide('He4', 1.0, 'wo')
    candu_gap.set_density('g/cm3', 0.0014)
    materials_list.append(candu_gap)

    # CANDU Fuel Gap (between fuel and cladding)
    candu_fuel_gap = openmc.Material(name='candu_fuel_gap')
    candu_fuel_gap.add_nuclide('He4', 1.0, 'wo')
    candu_fuel_gap.set_density('g/cm3', 0.00014)
    materials_list.append(candu_fuel_gap)

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
