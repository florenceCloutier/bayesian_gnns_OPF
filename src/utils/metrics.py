import torch
import numpy as np

"""Compute Total Relative Mean Absolute Error (TRMAE)."""
def compute_trmae(y_true, y_pred, max_threshold=0.001):
    relative_errors = torch.abs(y_true - y_pred) / torch.abs(y_true)
    
    # Threshold to avoid explosions in this metric due to small values in targets
    if max_threshold is not None:
        relative_errors = torch.clamp(relative_errors, max=max_threshold)

    return torch.mean(relative_errors).item()

import torch

def compute_metrics(data, branch_powers_ac_line, branch_powers_transformer, violations, data_type):
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
    # current_lr = optimizer.param_groups[0]['lr']

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
    va_violations = violations['voltage_angle']

    # Thermal limits
    flow_violation = violations['flow']
    
    # Power balance
    power_balance_violation = violations['power_balance']
                    
    metrics = {
        # data_type + '_lr': current_lr,
        data_type + '_trmae_to_bus_real_batch': np.mean(trmae_to_bus_real_batch),
        data_type + '_trmae_to_bus_reactive_batch': np.mean(trmae_to_bus_reactive_batch),
        data_type + '_trmae_from_bus_real_batch': np.mean(trmae_from_bus_real_batch),
        data_type + '_trmae_from_bus_reactive_batch': np.mean(trmae_from_bus_reactive_batch),
        data_type + '_trmae_bus_total': np.mean(trmae_to_bus_real_batch) + np.mean(trmae_to_bus_reactive_batch) + np.mean(trmae_from_bus_real_batch) + np.mean(trmae_from_bus_reactive_batch),
        data_type + '_trmae_generator_withdrawn_to_bus_real_batch': np.mean(trmae_generator_withdrawn_to_bus_real_batch),
        data_type + '_trmae_generator_withdrawn_to_bus_reactive_batch': np.mean(trmae_generator_withdrawn_to_bus_reactive_batch),
        data_type + '_trmae_generator_withdrawn_from_bus_real_batch': np.mean(trmae_generator_withdrawn_from_bus_real_batch),
        data_type + '_trmae_generator_withdrawn_from_bus_reactive_batch': np.mean(trmae_generator_withdrawn_from_bus_reactive_batch),
        data_type + '_trma_generator_total': np.mean(trmae_generator_withdrawn_to_bus_real_batch) + np.mean(trmae_generator_withdrawn_to_bus_reactive_batch) + np.mean(trmae_generator_withdrawn_from_bus_real_batch) + np.mean(trmae_generator_withdrawn_from_bus_reactive_batch),
        data_type + '_power_balance_violation': va_violations,
        data_type + '_voltage_angle_violation': flow_violation,
        data_type + '_thermal_flow_violation': power_balance_violation,
    }
    
    return metrics
                    