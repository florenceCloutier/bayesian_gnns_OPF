import torch

"""Compute Total Relative Mean Absolute Error (TRMAE)."""
def compute_trmae(y_true, y_pred, max_threshold=0.001):
    relative_errors = torch.abs(y_true - y_pred) / torch.abs(y_true)
    
    # Threshold to avoid explosions in this metric due to small values in targets
    if max_threshold is not None:
        relative_errors = torch.clamp(relative_errors, max=max_threshold)

    return torch.mean(relative_errors).item()

import torch

def power_balance_violation(
    pf_true_ac_line, pt_true_ac_line, 
    pf_pred_ac_line, pt_pred_ac_line, 
    pf_true_transformer, pt_true_transformer, 
    pf_pred_transformer, pt_pred_transformer):
    """
    Quantifies constraint violation for power balance using equation (8) from CANOS
    
    Parameters:
    - pf_true_ac_line, qf_true_ac_line, pt_true_ac_line, qt_true_ac_line: True branch power flows (AC lines).
    - pf_pred_ac_line, qf_pred_ac_line, pt_pred_ac_line, qt_pred_ac_line: Predicted branch power flows (AC lines).
    - pf_true_transformer, qf_true_transformer, pt_true_transformer, qt_true_transformer: True transformer flows.
    - pf_pred_transformer, qf_pred_transformer, pt_pred_transformer, qt_pred_transformer: Predicted transformer flows.
    - voltage_angles: Voltage angles at each bus.
    - voltage_angle_limits: Tuple of (lower_bound, upper_bound) for angle differences.
    - thermal_limits: Thermal limit for each branch.

    Returns:
    - Power balance violation
    """
    
    # Power Balance Violation
    # Sum of branch powers should equal net power injection at each bus
    power_balance_violation_ac_lines = torch.abs((pf_pred_ac_line + pt_pred_ac_line) - (pf_true_ac_line + pt_true_ac_line)).mean()
    power_balance_violation_trans = torch.abs((pf_pred_transformer + pt_pred_transformer) - (pf_true_transformer + pt_true_transformer)).mean()
    ac = power_balance_violation_ac_lines.item()
    trans = power_balance_violation_trans.item()
    return (ac, trans)

def thermal_limit_violation(pf_pred_ac_line, qf_pred_ac_line, pf_pred_transformer, qf_pred_transformer, thermal_limits_ac_line, thermal_limits_transformer):
    """
    Quantifies constraint violation for thermal limit using equation (11) from CANOS
    
    Parameters:
    - pf_pred_ac_line, qf_pred_ac_line, pt_pred_ac_line, qt_pred_ac_line: Predicted branch power flows (AC lines).
    - pf_pred_transformer, qf_pred_transformer, pt_pred_transformer, qt_pred_transformer: Predicted transformer flows.
    - thermal_limits: Thermal limit for each branch.

    Returns:
    - Thermal limit violation
    """
    
    # Thermal Limit Violation
    # Check if predicted power flow exceeds the thermal limits
    S_pred_ac_line = torch.sqrt(pf_pred_ac_line**2 + qf_pred_ac_line**2)
    S_pred_transformer = torch.sqrt(pf_pred_transformer**2 + qf_pred_transformer**2)
    thermal_limit_violation_ac_line = torch.sum(S_pred_ac_line > thermal_limits_ac_line)
    thermal_limit_violation_transformer = torch.sum(S_pred_transformer > thermal_limits_transformer)
    return (thermal_limit_violation_ac_line + thermal_limit_violation_transformer).item()

def voltage_angle_difference(voltage_angle, voltage_angle_limits):
    """
    Quantifies constraint violation for voltage angle difference using equation (12) from CANOS
    
    Parameters:
    - voltage_angles: Voltage angles at each bus.
    - voltage_angle_limits: Tuple of (lower_bound, upper_bound) for angle differences.

    Returns:
    - Voltage angle difference violation
    """

    # Voltage Angle Difference Violation
    # Check if the angle differences exceed the lower or upper bounds
    angle_lower_bound, angle_upper_bound = voltage_angle_limits
    angle_violations = torch.sum((voltage_angle < angle_lower_bound[0]) | (voltage_angle > angle_upper_bound[0]))
    return angle_violations.item()

def compute_metrics(data, out, branch_powers_ac_line, branch_powers_transformer, optimizer, BRANCH_FEATURE_INDICES, data_type):
    pf_pred_ac_line, qf_pred_ac_line, pt_pred_ac_line, qt_pred_ac_line = branch_powers_ac_line
    pf_pred_transformer, qf_pred_transformer, pt_pred_transformer, qt_pred_transformer = branch_powers_transformer
    
    edge_label_ac_line = data['bus', 'ac_line', 'bus'].edge_label
    pf_true_ac_line = edge_label_ac_line[:, 0]
    qf_true_ac_line = edge_label_ac_line[:, 1]
    pt_true_ac_line = edge_label_ac_line[:, 2]
    qt_true_ac_line = edge_label_ac_line[:, 3]
    
    edge_label_transformer = data['bus', 'transformer', 'bus'].edge_label
    pf_true_transformer = edge_label_transformer[:, 0]
    qf_true_transformer = edge_label_transformer[:, 1]
    pt_true_transformer = edge_label_transformer[:, 2]
    qt_true_transformer = edge_label_transformer[:, 3]
    
    # LR
    current_lr = optimizer.param_groups[0]['lr']

    # TRMAE
    trmae_to_bus_real_batch = compute_trmae(pt_true_ac_line, pt_pred_ac_line)
    trmae_to_bus_reactive_batch = compute_trmae(qt_true_ac_line, qt_pred_ac_line)
    trmae_from_bus_real_batch = compute_trmae(pf_true_ac_line, pf_pred_ac_line)
    trmae_from_bus_reactive_batch = compute_trmae(qf_true_ac_line, qf_pred_ac_line)
    
    trmae_generator_withdrawn_to_bus_real_batch = compute_trmae(pt_true_transformer, pt_pred_transformer)
    trmae_generator_withdrawn_to_bus_reactive_batch = compute_trmae(qt_true_transformer, qt_pred_transformer)
    trmae_generator_withdrawn_from_bus_real_batch = compute_trmae(pf_true_transformer, pf_pred_transformer)
    trmae_generator_withdrawn_from_bus_reactive_batch = compute_trmae(qf_true_transformer, qf_pred_transformer)
                    
    # Voltage angles
    voltage_angle_bus = out['bus'][:, 0] # theta in radians
    indices_bus_angmin = BRANCH_FEATURE_INDICES['ac_line']['angmin']
    indices_bus_angmax = BRANCH_FEATURE_INDICES['ac_line']['angmax']
    edge_attr_ac_line = data['bus', 'ac_line', 'bus'].edge_attr
    bus_angmin = edge_attr_ac_line[:, indices_bus_angmin]
    bus_angmax = edge_attr_ac_line[:, indices_bus_angmax]

    voltage_angle_node = out['generator'][:, 0] # theta in radians
    indices_node_angmin = BRANCH_FEATURE_INDICES['transformer']['angmin']
    indices_node_angmax = BRANCH_FEATURE_INDICES['transformer']['angmax']
    edge_attr_transformer = data['bus', 'transformer', 'bus'].edge_attr
    node_angmin = edge_attr_transformer[:, indices_node_angmin]
    node_angmax = edge_attr_transformer[:, indices_node_angmax]
                    
    # Thermal limits
    indices_ac_line = BRANCH_FEATURE_INDICES['ac_line']
    rate_a_ac_line = edge_attr_ac_line[:, indices_ac_line['rate_a']]
    indices_transformer = BRANCH_FEATURE_INDICES['transformer']
    rate_a_transformer = edge_attr_transformer[:, indices_transformer['rate_a']]
    
    # Power balance
    power_balance = power_balance_violation(pf_true_ac_line, pt_true_ac_line, pf_pred_ac_line, pt_pred_ac_line, pf_true_transformer, pt_true_transformer, pf_pred_transformer, pt_pred_transformer)
    
    metrics = {
        data_type + '_lr': current_lr,
        data_type + '_trmae_to_bus_real_batch': trmae_to_bus_real_batch,
        data_type + '_trmae_to_bus_reactive_batch': trmae_to_bus_reactive_batch,
        data_type + '_trmae_from_bus_real_batch': trmae_from_bus_real_batch,
        data_type + '_trmae_from_bus_reactive_batch': trmae_from_bus_reactive_batch,
        data_type + '_trmae_bus_total': trmae_to_bus_real_batch + trmae_to_bus_reactive_batch + trmae_from_bus_real_batch + trmae_from_bus_reactive_batch,
        data_type + '_trmae_generator_withdrawn_to_bus_real_batch': trmae_generator_withdrawn_to_bus_real_batch,
        data_type + '_trmae_generator_withdrawn_to_bus_reactive_batch': trmae_generator_withdrawn_to_bus_reactive_batch,
        data_type + '_trmae_generator_withdrawn_from_bus_real_batch': trmae_generator_withdrawn_from_bus_real_batch,
        data_type + '_trmae_generator_withdrawn_from_bus_reactive_batch': trmae_generator_withdrawn_from_bus_reactive_batch,
        data_type + '_trma_generator_total': trmae_generator_withdrawn_to_bus_real_batch + trmae_generator_withdrawn_to_bus_reactive_batch + trmae_generator_withdrawn_from_bus_real_batch + trmae_generator_withdrawn_from_bus_reactive_batch,
        data_type + '_power_balance_violation_ac_line': power_balance[0],
        data_type + '_power_balance_violation_trans': power_balance[1],
        data_type + '_voltage_angle_difference_node': voltage_angle_difference(voltage_angle_node, [node_angmin, node_angmax]),
        data_type + '_voltage_angle_difference_bus': voltage_angle_difference(voltage_angle_bus, [bus_angmin, bus_angmax]),
        data_type + '_thermal_limits': thermal_limit_violation(pf_pred_ac_line, qf_pred_ac_line, pf_pred_transformer, qf_pred_transformer, rate_a_ac_line, rate_a_transformer),
    }
    
    return metrics
                    