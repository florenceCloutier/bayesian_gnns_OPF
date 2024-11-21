import torch

"""
HeteroData Mapping to Raw OPF Data:
- Nodes:
  * 'bus': Represents buses in the grid, derived from `grid['nodes']`.
    - x: Node features (e.g., voltage magnitude, type).
    - y: Target values (e.g., optimal voltage magnitude and angle).
  * 'generator', 'load', 'shunt': Represent generators, loads, and shunts, respectively.
    - x: Features (e.g., generator limits, load demands).
    - y: Target values (e.g., active/reactive power outputs for generators).

- Edges:
  * (bus, ac_line, bus): Transmission lines between buses, derived from `grid['edges']`.
    - edge_index: Connectivity of buses (source, target).
    - edge_attr: Attributes (e.g., admittance, line limits).
    - edge_label: Target branch power flows (active/reactive power).

  * (bus, transformer, bus): Transformers between buses, similar structure to ac_line.

- Batch Attributes:
  * batch: Specifies which graph each node/edge belongs to in a batched dataset.
  * ptr: Defines graph boundaries in a batched dataset.
"""

BRANCH_FEATURE_INDICES = {
    'ac_line': {
        'b_fr': 2,
        'b_to': 3,
        'R_ij': 4,
        'X_ij': 5,
    },
    'transformer': {
        'b_fr': 9,
        'b_to': 10,
        'R_ij': 2,
        'X_ij': 3,
        'tap': 7,
        'shift': 8,
    }
}


def compute_branch_powers(out, data, type):
    voltage_magnitude = out['bus'][:, 1] # |V|
    voltage_angle = out['bus'][:, 0] # theta in radians

    # Compute complex bus voltages
    V = voltage_magnitude * torch.exp(1j * voltage_angle)

    # extract edge attributes
    edge_index = data['bus', type, 'bus'].edge_index
    edge_attr = data['bus', type, 'bus'].edge_attr



    # Extract branch features (correct indices derived from canos paper)
    indices = BRANCH_FEATURE_INDICES[type]
    b_fr = edge_attr[:, indices['b_fr']]
    b_to = edge_attr[:, indices['b_to']]
    R_ij = edge_attr[:, indices['R_ij']]
    X_ij = edge_attr[:, indices['X_ij']]

    # unified tap ratio and shift initialization
    T_ij = torch.ones(edge_attr.shape[0], dtype=torch.complex64)
    shift_rad = torch.zeros(edge_attr.shape[0])

    if type == 'transformer':
        tap = edge_attr[:, indices['tap']]
        shift = edge_attr[:, indices['shift']]
        shift_rad = shift * (torch.pi / 180)
        T_ij = tap * torch.exp(1j * shift_rad)

    # Series admittance
    Z_ij = R_ij + 1j * X_ij
    Y_ij = 1 / Z_ij

    # Shunt admittance (not 100% about this part, need to double check averaging is right)
    B_c = (b_fr + b_to) / 2
    Y_c = 1j * B_c

    # From and to Buses
    from_bus = edge_index[0]
    to_bus = edge_index[1]

    # Complex voltages
    V_i = V[from_bus]
    V_j = V[to_bus]

    # Voltage magnitudes squared
    V_i_abs_squared = torch.abs(V_i) ** 2
    V_j_abs_squared = torch.abs(V_j) ** 2

    # Voltage products
    Vi_Vj_conj = V_i * V_j.conj()
    Vi_conj_Vj = V_i.conj() * V_j

    # Equation 9 (not sure but i think for ac_line, we set Tij to 1)
    S_ij = (Y_ij + Y_c).conj() * (V_i_abs_squared / torch.abs(T_ij) ** 2) - Y_ij.conj() * (Vi_Vj_conj / T_ij)

    # Equation 10
    S_ji = (Y_ij + Y_c).conj() * V_j_abs_squared - Y_ij.conj() * (Vi_conj_Vj / T_ij.conj())

    pf = S_ij.real
    qf = S_ij.imag
    pt = S_ji.real
    qt = S_ji.imag

    return pf, qf, pt, qt

