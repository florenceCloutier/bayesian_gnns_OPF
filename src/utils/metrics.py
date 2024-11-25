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