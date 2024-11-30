def test_va_values(out, data):
    """ 
    This is to verify if what is stored in out['bus'][:, 0] is the node
    level angles (theta_i) or the edge level angle differences (theta_ij).
    This will help determine if we need to actually compute angle differences
    or not for the voltage angle constraints.
    """
    # Number of nodes (buses)
    num_nodes = data['bus'].x.shape[0] / 4
    print(f"Number of nodes (buses): {num_nodes}")

    # Number of edges (branches) for AC lines
    num_edges_ac_line = data['bus', 'ac_line', 'bus'].edge_index.shape[1] / 4
    print(f"Number of edges (AC lines): {num_edges_ac_line}")

    # Number of edges (branches) for transformers
    num_edges_transformer = data['bus', 'transformer', 'bus'].edge_index.shape[1] / 4
    print(f"Number of edges (transformers): {num_edges_transformer}")

    # Total number of edges
    total_edges = num_edges_ac_line + num_edges_transformer 
    print(f"Total number of edges (branches): {total_edges}")

    # Shape of predicted va
    predicted_va = out['bus'][:, 0] / 4 
    print(f"Predicted va shape: {predicted_va.shape[0]}")

    if predicted_va.shape[0] == 14:
        print("va corresponds to node voltage angles (θ_i).")
    elif predicted_va.shape[0] == 20:
        print("va corresponds to edge voltage angle differences (θ_ij).")
    else:
        print("The shape of va does not directly match nodes or edges.")