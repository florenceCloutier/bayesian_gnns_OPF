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

# These indices are based on OPF dataset paper: https://arxiv.org/pdf/2406.07234
BRANCH_FEATURE_INDICES = {
    "ac_line": {
        "angmin": 0,
        "angmax": 1,
        "b_fr": 2,
        "b_to": 3,
        "R_ij": 4,
        "X_ij": 5,
        "rate_a": 6,
    },
    "transformer": {
        "angmin": 0,
        "angmax": 1,
        "b_fr": 9,
        "b_to": 10,
        "R_ij": 2,
        "X_ij": 3,
        "rate_a": 4,
        "tap": 7,
        "shift": 8,
    },
}


def enforce_bound_constraints(out, data):
    """
    Bound constraints (6)-(7) from CANOS
    y = sigmoid(y) * (y_upper - y_lower) + y_lower
    """
    vmin = data["bus"].x[:, 2]
    vmax = data["bus"].x[:, 3]
    pmin = data["generator"].x[:, 2]
    pmax = data["generator"].x[:, 3]
    qmin = data["generator"].x[:, 5]
    qmax = data["generator"].x[:, 6]

    out["bus"][:, 1] = torch.sigmoid(out["bus"][:, 1]) * (vmax - vmin) + vmin
    out["generator"][:, 0] = torch.sigmoid(out["generator"][:, 0]) * (pmax - pmin) + pmin
    out["generator"][:, 1] = torch.sigmoid(out["generator"][:, 1]) * (qmax - qmin) + qmin


def compute_branch_powers(out, data, type, device):
    voltage_magnitude = out["bus"][:, 1]  # |V|
    voltage_angle = out["bus"][:, 0]  # theta in radians

    # Compute complex bus voltages
    V = voltage_magnitude * torch.exp(1j * voltage_angle)

    # extract edge attributes
    edge_index = data["bus", type, "bus"].edge_index
    edge_attr = data["bus", type, "bus"].edge_attr

    # Extract branch features
    indices = BRANCH_FEATURE_INDICES[type]
    b_fr = edge_attr[:, indices["b_fr"]]
    b_to = edge_attr[:, indices["b_to"]]
    R_ij = edge_attr[:, indices["R_ij"]]
    X_ij = edge_attr[:, indices["X_ij"]]

    # unified tap ratio and shift initialization
    T_ij = torch.ones(edge_attr.shape[0], dtype=torch.complex64, device=device)
    shift_rad = torch.zeros(edge_attr.shape[0], device=device)

    if type == "transformer":
        tap = edge_attr[:, indices["tap"]]
        shift_rad = edge_attr[:, indices["shift"]]
        T_ij = tap * torch.exp(1j * shift_rad)
    else:
        tap = torch.ones(edge_attr.shape[0], device=device)

    # Series admittance
    Z_ij = R_ij + 1j * X_ij
    Y_ij = 1 / Z_ij

    # Shunt admittance (not 100% about this part, need to double check averaging is right)
    Y_c = 1j * b_fr

    # From and to Buses
    from_bus = edge_index[0]
    to_bus = edge_index[1]

    # Complex voltages
    V_i = V[from_bus]
    V_j = V[to_bus]

    # Voltage products
    Vi_Vj_conj = V_i * V_j.conj()
    Vi_conj_Vj = V_i.conj() * V_j

    # Equation 9 (not sure but i think for ac_line, we set Tij to 1)
    S_ij = (Y_ij + Y_c).conj() * ((voltage_magnitude[from_bus]) ** 2 / tap**2) - Y_ij.conj() * (
        Vi_Vj_conj / T_ij
    )

    # Equation 10
    S_ji = (Y_ij + Y_c).conj() * (voltage_magnitude[to_bus]) ** 2 - Y_ij.conj() * (
        Vi_conj_Vj / T_ij
    ) 

    pf = S_ij.real
    qf = S_ij.imag
    pt = S_ji.real
    qt = S_ji.imag

    return pf, qf, pt, qt


def power_balance_loss(out, data, branch_powers_ac_line, branch_powers_transformer, device):
    """
    Check power balance constraints at each bus (Eq. 8)
    The power in should equal the power demand on each bus.
    Implements violation constraints v_5a, v_5b, v_6a, v_6b.
    """

    def loss(pg, qg, pf, qf, pd, qd, p_shunt, q_shunt):
        # Net injection at from bus
        p_violation = torch.abs(pg - pd - pf - p_shunt)
        q_violation = torch.abs(qg - qd - qf - q_shunt)
        return p_violation.mean(dim=0) + q_violation.mean(dim=0)  # TODO: maybe we get individual means and stack?

    # Generator power injection
    gen_edge_index = data["generator", "generator_link", "bus"].edge_index
    from_gen = gen_edge_index[0]
    to_bus = gen_edge_index[1]
    pg = torch.zeros(out["bus"].shape[0])
    qg = torch.zeros(out["bus"].shape[0])
    pg_sum = torch.bincount(to_bus, weights=out["generator"][from_gen, 0])
    qg_sum = torch.bincount(to_bus, weights=out["generator"][from_gen, 1])
    pg[: pg_sum.shape[0]] = pg_sum
    qg[: qg_sum.shape[0]] = qg_sum

    # Load power demands
    load_edge_index = data["load", "load_link", "bus"].edge_index
    from_load = load_edge_index[0]
    to_bus = load_edge_index[1]
    pd = torch.zeros(out["bus"].shape[0])
    qd = torch.zeros(out["bus"].shape[0])
    pd_sum = torch.bincount(to_bus, weights=data["load"].x[from_load, 0])
    qd_sum = torch.bincount(to_bus, weights=data["load"].x[from_load, 1])
    pd[: pd_sum.shape[0]] = pd_sum
    qd[: qd_sum.shape[0]] = qd_sum

    # Net power injection from branches
    ac_edge_index = data["bus", "ac_line", "bus"].edge_index
    from_bus = ac_edge_index[0]
    pf = torch.zeros(out["bus"].shape[0])
    qf = torch.zeros(out["bus"].shape[0])
    pf_pred_ac_line, qf_pred_ac_line, _, _ = branch_powers_ac_line
    pf_sum = torch.bincount(from_bus, weights=pf_pred_ac_line)
    qf_sum = torch.bincount(from_bus, weights=qf_pred_ac_line)
    pf[: pf_sum.shape[0]] = pf_sum
    qf[: qf_sum.shape[0]] = qf_sum

    transformer_edge_index = data["bus", "transformer", "bus"].edge_index
    from_bus = transformer_edge_index[0]
    pf_pred_transformer, qf_pred_transformer, _, _ = branch_powers_transformer
    pf_sum = torch.bincount(from_bus, weights=pf_pred_transformer, device)
    qf_sum = torch.bincount(from_bus, weights=qf_pred_transformer)
    pf[: pf_sum.shape[0]] += pf_sum
    qf[: qf_sum.shape[0]] += qf_sum

    # Shunt power injection
    shunt_edge_index = data["shunt", "shunt_link", "bus"].edge_index
    from_shunt = shunt_edge_index[0]
    to_bus = shunt_edge_index[1]

    p_shunt = torch.zeros(out["bus"].shape[0])
    q_shunt = torch.zeros(out["bus"].shape[0])
    bs_sum = torch.bincount(to_bus, weights=data["shunt"].x[from_shunt, 0])
    gs_sum = torch.bincount(to_bus, weights=data["shunt"].x[from_shunt, 1])

    p_shunt[: gs_sum.shape[0]] = gs_sum * out["bus"][: bs_sum.shape[0], 1] ** 2
    q_shunt[: bs_sum.shape[0]] = -bs_sum * out["bus"][: bs_sum.shape[0], 0] ** 2

    return loss(pg, qg, pf, qf, pd, qd, p_shunt, q_shunt)


def flow_loss(data, branch_powers_ac_line, branch_powers_transformer):
    """
    Check flow constraints at each branch (Eq. 11)
    Use relu to only penalize if we go over the max.
    Implements violation constraint v_4
    """
    pf_pred_ac_line, qf_pred_ac_line, _, _ = branch_powers_ac_line
    edge_attr = data["bus", "ac_line", "bus"].edge_attr
    rate_a = edge_attr[:, BRANCH_FEATURE_INDICES["ac_line"]["rate_a"]]
    flow_loss_ac = torch.relu(torch.square(pf_pred_ac_line) + torch.square(qf_pred_ac_line) - rate_a)

    pf_pred_transformer, qf_pred_transformer, _, _ = branch_powers_transformer
    edge_attr = data["bus", "transformer", "bus"].edge_attr
    rate_a = edge_attr[:, BRANCH_FEATURE_INDICES["transformer"]["rate_a"]]
    flow_loss_transformer = torch.relu(
        torch.square(pf_pred_transformer) + torch.square(qf_pred_transformer) - torch.square(rate_a)
    )

    return flow_loss_ac.mean(dim=0) + flow_loss_transformer.mean(dim=0)


def voltage_magnitude_loss(out, data):
    """
    Compute violation degrees for voltage magnitude bounds at each bus.
    Returns the violation degrees for buses with out-of-bound voltages.

    Implements violation constraint v_2a
    """

    vmin = data["bus"].x[:, 2]
    vmax = data["bus"].x[:, 3]
    v_hat = out["bus"][:, 1]  # voltage magnitude predictions

    # Satisfiability degrees -> satisfied if sigma(x) <= 0 and smaller means more satisfiability
    sigma_2a_L = vmin - v_hat  # sigma(x) <= 0 => v_hat is bigger than min voltage
    sigma_2a_R = v_hat - vmax  # sigma(x) <= 0 => v_hat is smaller than max voltage

    # using the satisfiability degrees we can now define the violation degrees
    # for ineq constraints => v(x) = max(0, sigma(x))

    violation_degrees = torch.mean(torch.relu(sigma_2a_L) + torch.relu(sigma_2a_R))

    return violation_degrees


def voltage_angle_loss(out, data):
    """
    Check voltage angle constraints at each branch (Eq. 12)
    Use relu to only penalize if we break the constraints.
    Implements violation constraint v_2b
    """
    edge_attr = data["bus", "ac_line", "bus"].edge_attr
    edge_index = data["bus", "ac_line", "bus"].edge_index

    from_bus = edge_index[0]
    to_bus = edge_index[1]

    va_min = edge_attr[:, BRANCH_FEATURE_INDICES["ac_line"]["angmin"]]
    va_max = edge_attr[:, BRANCH_FEATURE_INDICES["ac_line"]["angmax"]]

    # Compute voltage angle difference θ_ij = θ_i - θ_j
    va_diff_ac = out["bus"][from_bus, 0] - out["bus"][to_bus, 0]

    max_loss_ac = torch.relu(va_diff_ac - va_max)
    min_loss_ac = torch.relu(va_min - va_diff_ac)
    va_loss_ac = max_loss_ac + min_loss_ac

    edge_attr = data["bus", "transformer", "bus"].edge_attr
    edge_index = data["bus", "transformer", "bus"].edge_index

    from_bus = edge_index[0]
    to_bus = edge_index[1]

    va_min = edge_attr[:, BRANCH_FEATURE_INDICES["transformer"]["angmin"]]
    va_max = edge_attr[:, BRANCH_FEATURE_INDICES["transformer"]["angmax"]]

    va_diff_tf = out["bus"][from_bus, 0] - out["bus"][to_bus, 0]

    max_loss_transformer = torch.relu(va_diff_tf - va_max)
    min_loss_transformer = torch.relu(va_min - va_diff_tf)
    va_loss_transformer = max_loss_transformer + min_loss_transformer

    return va_loss_ac.mean(dim=0) + va_loss_transformer.mean(dim=0)


def real_power_loss(out, data):
    """
    Compute violation degrees for active power generation limits at each generator.
    Implements violation constraint v_3a.
    """
    p_hat = out["generator"][:, 0]
    pmin = data["generator"].x[:, 2]
    pmax = data["generator"].x[:, 3]

    # Compute satisfiability degrees
    sigma_L = pmin - p_hat  # sigma^L_3a: Violation of lower bound
    sigma_R = p_hat - pmax  # sigma^R_3a: Violation of upper bound

    # Compute violation degrees using ReLU (v_c(sigma) = max(0, sigma))
    violation_L = torch.relu(sigma_L)  # Left violation degree
    violation_R = torch.relu(sigma_R)  # Right violation degree

    # Total violation degree for each generator
    total_violation = violation_L + violation_R

    # Average violation across all generators
    avg_violation = total_violation.mean(dim=0)

    return avg_violation


def reactive_power_loss(out, data):
    """
    Compute violation degrees for reactive power generation limits at each generator.
    Implements violation constraint v_3b
    """
    q_hat = out["generator"][:, 1]
    qmin = data["generator"].x[:, 5]
    qmax = data["generator"].x[:, 6]

    # Compute satisfiability degrees
    sigma_L = qmin - q_hat  # sigma^L_3a: Violation of lower bound
    sigma_R = q_hat - qmax  # sigma^R_3a: Violation of upper bound

    # Compute violation degrees using ReLU (v_c(sigma) = max(0, sigma))
    violation_L = torch.relu(sigma_L)  # Left violation degree
    violation_R = torch.relu(sigma_R)  # Right violation degree

    # Total violation degree for each generator
    total_violation = violation_L + violation_R

    # Average violation across all generators
    avg_violation = total_violation.mean(dim=0)

    return avg_violation


def cost(out, data):
    """
    Function to compute total generator costs.
    """
    pg = out["generator"][:, 0]
    cost_squared = data["generator"].x[:, 8]
    cost_linear = data["generator"].x[:, 9]
    cost_offset = data["generator"].x[:, 10]

    return cost_squared * torch.square(pg) + cost_linear * pg + cost_offset