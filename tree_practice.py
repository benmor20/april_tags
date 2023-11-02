import cv2
import numpy as np
import networkx as nx
from typing import *
import matplotlib.pyplot as plt
import math
from main import line_segment_distances


def generate_tree(segments: np.ndarray, dist_lookup: np.ndarray, cross_lookup: np.ndarray) -> nx.Graph:   # Maya
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
    direction = ""

    # Add the root node to the graph
    root = "Root"
    graph.add_node(root)

    # level 1
    for i, segment in enumerate(segments):
        child = (i, 1)
        graph.add_edge(root, child)
        # level 2
        for j in range(len(segments)):
            if j != i and dist_lookup[i][j] and cross_lookup[i][j] < 0:
                child2 = (j, i, 2)
                graph.add_edge(child, child2)
                # level 3
                for k in range(len(segments)):
                    if k != i and k != j and dist_lookup[j][k] and cross_lookup[j][k] < 0:
                        child3 = (k, j, i, 3)
                        graph.add_edge(child2, child3)
                        # level 4
                        for l in range(len(segment)):
                            if l != i and l != k and l != j and dist_lookup[k][l] and cross_lookup[k][l] < 0:
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
        quads = dfs(root, [], segment_tree, quads, segments)
    return quads
    

def dfs(node, path, segment_tree, quads, segments):
    """
    Doc string here
    """
    if len(path) == 4:
        path.sort()
        if path not in quads:
            quads.append(path)
        return
    for child in list(segment_tree.successors(node)):
        dfs(child, path + [child[0]], segment_tree, quads, segments)
    return np.array(quads)


def cross_table(segments):
    """
    Doc string here
    """
    num_segments = len(segments)
    cross_table = np.zeros((num_segments, num_segments))  # Initialize cross_table with zeros

    for i in range(len(segments)):
        for j in range(len(segments)):            
            parent = np.array([segments[i][2]-segments[i][0], segments[i][3]-segments[i][1]])
            child = np.array([segments[j][2]-segments[j][0], segments[j][3]-segments[j][1]])
            cross_table[i][j] = np.cross(parent, child)
    return cross_table


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
  
    
def plot_quad(quad):
    for segment in quad:
        plot_segment(segment)



def main():
    segments = np.array([[1, 0, 437, 0],
                         [381, 105, 59, 105],
                         [58, 112, 58, 221],
                         [381, 163, 58, 164],
                         [382, 221, 382, 107],
                         [383, 221, 383, 109],
                         [384, 221, 384, 108],
                         [59, 222, 381, 222],
                         [0, 326, 0, 1],
                         [0, 163, 437, 163],
                         [438, 1, 438, 326],
                         [437, 327, 1, 327]])

    dist_lookup = line_segment_distances(segments)
    cross_lookup = cross_table(segments)
    print(cross_lookup)

   
    tree = generate_tree(segments, dist_lookup, cross_lookup)
    print(tree)

    # =show_tree(tree)

    quads = get_quads_from_tree(tree, dist_lookup, segments)

    print(quads)
    print(len(quads))


    # for quad in quads:
    #     for i in quad:
    #         plot_segment(segments[i])

    for i in quads[37]:
        plot_segment(segments[i])

    
    # for quad in quads:
    #     plot_quad(quad)
    plt.show()


   

if __name__ == '__main__':
    main()
