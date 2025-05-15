import numpy as np
from scipy.ndimage import convolve

def get_neighbors_coords(r, c, shape):
    """ Get 8-connectivity neighbors within image bounds """
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                neighbors.append((nr, nc))
    return neighbors

def image_to_graph(binary_skeleton, pixel_interval):
    if not np.any(binary_skeleton):
        print("Warning: No skeleton pixels found in the image after binarization.")
        return np.array([]), np.array([])

    # Kernel for counting 8-connectivity neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_counts = convolve(binary_skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    neighbor_counts_on_skeleton = neighbor_counts * binary_skeleton
    
    junction_pixels_rc = list(zip(*np.where(neighbor_counts_on_skeleton > 2)))
    endpoint_pixels_rc = list(zip(*np.where(neighbor_counts_on_skeleton == 1)))

    critical_pixels_rc_set = set(junction_pixels_rc + endpoint_pixels_rc)

    nodes_list_rc = [] 
    coord_to_node_id = {} 
    edges_list = [] 

    def add_node(r, c):
        if (r, c) not in coord_to_node_id:
            node_id = len(nodes_list_rc)
            nodes_list_rc.append((r, c))
            coord_to_node_id[(r, c)] = node_id
            return node_id
        return coord_to_node_id[(r, c)]

    for r_crit, c_crit in critical_pixels_rc_set:
        add_node(r_crit, c_crit)
    
    explored_pixel_segments = set() # Stores tuples of ((r1,c1), (r2,c2)) sorted, to mark physical segments

    initial_node_coords_for_iteration = list(critical_pixels_rc_set) 

    for r_start_node, c_start_node in initial_node_coords_for_iteration:
        # This is an initial critical node. Its ID is already in coord_to_node_id.
        current_path_origin_node_id = coord_to_node_id[(r_start_node, c_start_node)]

        for r_neighbor, c_neighbor in get_neighbors_coords(r_start_node, c_start_node, binary_skeleton.shape):
            if not binary_skeleton[r_neighbor, c_neighbor]:
                continue

            segment_canonical_repr = tuple(sorted(((r_start_node, c_start_node), (r_neighbor, c_neighbor))))
            if segment_canonical_repr in explored_pixel_segments:
                continue

            path_current_rc = (r_neighbor, c_neighbor)
            path_prev_rc = (r_start_node, c_start_node)
            
            last_established_node_id = current_path_origin_node_id
            pixels_traveled_from_last_node = 0

            while True:
                current_segment_canonical_repr = tuple(sorted((path_prev_rc, path_current_rc)))
                explored_pixel_segments.add(current_segment_canonical_repr)
                
                pixels_traveled_from_last_node += 1

                is_critical_node_ahead = path_current_rc in critical_pixels_rc_set
                is_intermediate_node_candidate = pixels_traveled_from_last_node == pixel_interval

                if is_critical_node_ahead:
                    # Reached an original junction or endpoint
                    target_node_id = coord_to_node_id[path_current_rc]
                    if last_established_node_id != target_node_id:
                        edges_list.append((last_established_node_id, target_node_id))
                    break 
                
                if is_intermediate_node_candidate:
                    # Add an intermediate node at path_current_rc
                    intermediate_node_id = add_node(path_current_rc[0], path_current_rc[1])
                    if last_established_node_id != intermediate_node_id:
                         edges_list.append((last_established_node_id, intermediate_node_id))
                    last_established_node_id = intermediate_node_id
                    pixels_traveled_from_last_node = 0 

                next_skeleton_pixels = []
                for r_next_cand, c_next_cand in get_neighbors_coords(path_current_rc[0], path_current_rc[1], binary_skeleton.shape):
                    if binary_skeleton[r_next_cand, c_next_cand] and (r_next_cand, c_next_cand) != path_prev_rc:
                        next_skeleton_pixels.append((r_next_cand, c_next_cand))
                
                if not next_skeleton_pixels: 
                    target_node_id = add_node(path_current_rc[0], path_current_rc[1])
                    if last_established_node_id != target_node_id:
                        edges_list.append((last_established_node_id, target_node_id))
                    break 

                if len(next_skeleton_pixels) == 1:
                    path_prev_rc = path_current_rc
                    path_current_rc = next_skeleton_pixels[0]
                else: # len(next_skeleton_pixels) > 1, meaning path_current_rc is a junction
                      # This implies path_current_rc should have been in critical_pixels_rc_set
                      # or it's a point where a path crosses itself without being a pre-identified junction.
                      # Treat it as a terminal node for this path segment.
                    target_node_id = add_node(path_current_rc[0], path_current_rc[1]) # ensure it's a node
                    if last_established_node_id != target_node_id:
                        edges_list.append((last_established_node_id, target_node_id))
                    break 
    
    nodes_array_xy = np.array([(c, r) for r, c in nodes_list_rc], dtype=int)
    
    unique_edges_set = set()
    final_edges_list = []
    for u, v in edges_list:
        if u == v: continue 
        # Store edges as (u,v) assuming tracing direction matters.
        # If graph is undirected, canonicalize: edge_tuple = tuple(sorted((u, v)))
        # For now, assume directed edges are fine, just remove exact duplicates.
        if (u,v) not in unique_edges_set: # Check for exact (u,v) duplicate
             final_edges_list.append((u,v))
             unique_edges_set.add((u,v))
            
    edges_array = np.array(final_edges_list, dtype=int)

    if nodes_array_xy.shape[0] == 0 and edges_array.shape[0] > 0:
        print("Warning: Edges found but no nodes. This indicates an issue in graph construction.")
        return np.array([]), np.array([])
        
    return nodes_array_xy, edges_array