import cv2
import numpy as np
import networkx as nx
from typing import *
import matplotlib.pyplot as plt
import math

def generate_tree(segments: np.ndarray, dist_lookup: np.ndarray) -> nx.Graph:   # Maya
    """
    Generate a tree of line segments to detect quadrilaterals from

    The root of the graph points to every possible line segment. From there,
    each child node points to every line segment whose start is "close enough"
    to the parent node's end. Close enough is defined as 2*parent length plus
    five pixels. Repeat for four levels deep.

    Args:
        segments: a [n x (x1, y1, x2, y2)] numpy array giving the start and end
            points of each line segment, so that traveling from point 1 to
            point 2 has the light side on the right.
        dist_lookup: a [n x n] numpy array giving the distance (in pixels)
            between the end line A and the start of line B for all A and B. A
            is indexed by row, B by column. Points along the diagonal give
            the length of the line.
    
    Returns:
        a NetworkX graph giving the tree of line s
    """
    graph = nx.DiGraph()

    # Add the root node to the graph
    root = "Root"
    graph.add_node(root)

    # level 1
    for i, segment in enumerate(segments):
        child = (i, 1)
        graph.add_edge(root, child)
        # level 2
        for j in range(len(segments)):
            if j != i:
                child2 = (j, i, 2)
                graph.add_edge(child, child2)
                # level 3
                for k in range(len(segments)):
                    if k != i and k != j:
                        child3 = (k, j, i, 3)
                        graph.add_edge(child2, child3)
                        # level 4
                        for l in range(len(segment)):
                            if l != i and l != k and l != j:
                                child4 = (l, k, j , i, 4)
                                graph.add_edge(child3, child4)
    return graph
  


def get_quads_from_tree(segment_tree: nx.Graph, dist_lookup: np.ndarray, segments) -> np.ndarray:  # Maya
    """
    Takes tree and conducts a depth-first search for possible quadrilaterals
        based on nearby line segments.

    Follows possible quads in a consistent direction clockwise or 
    counter-clockwise based on first two line segments. Rejects quad if third
    line segment is over pi radians away. Also rejects quad if last segment
    does not end "close enough" to start of first  segment.

    Args:
        tree:
        
    Return:
        a [q x 4 x (x1, y1, x2, y2)] numpy array where 4 x (x1, y1, x2, y2)
            determines the four line segments making up the quad
    """
    quads = []
    for root in segment_tree.nodes():
        quads = dfs(root, [], segment_tree, quads, dist_lookup, segments)
    return quads
    

def dfs(node, path, segment_tree, quads, dist_lookup, segments):
    if len(path) == 4:
        path.sort()
        if path not in quads:
            quads.append(path)
        return
    for child in list(segment_tree.successors(node)):
        if node == "Root":
            dfs(child, path + [child[0]], segment_tree, quads,  dist_lookup, segments)
        elif dist_lookup[node[0]][child[0]]:
            dfs(child, path + [child[0]], segment_tree, quads,  dist_lookup, segments)
    return np.array(quads)

# def check_distance(parent_index, child_index, dist_lookup, segments):
#     x1, y1, x2, y2 = segments[parent_index]
#     length = math.sqrt((x1 - x2)**2 + (y1-y2)**2)
#     thresh_hold = length * 2 + 5
#     if thresh_hold > dist_lookup[parent_index][child_index]:
#         return True
#     return False


def show_tree(tree):
    # Define the layout with root at the top center
    layout = nx.drawing.nx_agraph.graphviz_layout(tree, prog="dot")
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set the y-coordinates to be negative to reverse the tree orientation
    for node, coords in layout.items():
        layout[node] = (coords[0], -coords[1])
    
    # Draw the tree with customized layout
    nx.draw(tree, layout, with_labels=True, node_size=500, node_color="skyblue", font_size=10, ax=ax)
    
    # Show the plot
    plt.title("Segment Tree")
    plt.show()

def plot_segment(segment):
    x1, y1, x2, y2 = segment
    x = [x1,x2]
    y = [y1,y2]
    plt.plot(x, y, label='Line', marker='o', markersize=5)  # 'o' specifies markers at the data points
    


def main():
    # Number of line segments
    num_segments = 5

    # Create a random set of line segments as (x1, y1, x2, y2)
    segments = np.random.rand(num_segments, 4)

    # Ensure that x1 < x2, y1 < y2 for each segment
    segments[:, [0, 2]] = np.sort(segments[:, [0, 2]], axis=1)
    segments[:, [1, 3]] = np.sort(segments[:, [1, 3]], axis=1)

    print("Random Line Segments:")


    # Calculate distances between segments
    dist_lookup = np.zeros((num_segments, num_segments))
    print(dist_lookup)


    for i in range(num_segments):
        for j in range(num_segments):
            # Calculate distance between the end of segment i and the start of segment j
            x1, y1, x2, y2 = segments[i]
            x3, y3, x4, y4 = segments[j]
            distance = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
            dist_lookup[i, j] = distance

    # print("Distance Lookup Table:")
    # print(dist_lookup)

   
    tree = generate_tree(segments, dist_lookup)
    print(tree)

    # =show_tree(tree)

    quads = get_quads_from_tree(tree, dist_lookup, segments)

    print(quads)
    print(len(quads))

    l1, l2, l3, l4 = quads[0]
    x1, y1, x2, y2 = segments[l1]
    x = [x1,x2]
    y = [y1,y2]
    for i in range(4):
        plot_segment(quads[0][i])
    plt.show()


   

if __name__ == '__main__':
    main()
