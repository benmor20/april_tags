"""
Module to detect April Tags
"""
import itertools

import cv2
import numpy as np
import networkx as nx
from scipy import signal
from typing import *
from cluster import Cluster
from angle_utils import angle_diff
from matplotlib import pyplot as plt


def camera_capture(show_image: bool = False) -> np.ndarray:
    """
    Capture an image from the laptop's webcam using OpenCV

    Args:
        show_image: a bool, whether to display the image. Defaults to false.

    Returns:
        a [w x h x 3] numpy array giving the image - axes are x by y by BRG
    """
    cam = cv2.VideoCapture(0)
    result, image = cam.read()

    if show_image:
        if result:
            cv2.imshow('Camera', image)
            cv2.waitKey(0)
        else:
            print('Result is False')
    return image


def april_tag_detector(image_matrix: np.ndarray):
    """
    Detect an april tag within an image
    
    Args:
        image_matrix: a [w x h x 3] numpy array giving the image to capture
            an april tag from - axes are x by y by BRG
    """
    bw_image = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
    segments = segmentation(bw_image)
    quad_detector(segments)


def segmentation(black_white_matrix: np.ndarray) -> np.ndarray:
    """
    Find line segments that border black and white shapes
    
    Args:
        black_white_matrix: a [w x h] numpy array giving black and white
            image

    Returns:
        a [n x (x1, y1, x2, y2)] numpy array giving a list of line segments
            identified by endpoints. Direction of line segments is such that
            light shapes are on the right from a segment's perspective
    """
    # Compute the gradient of a black_white_matrix
    gradient = compute_gradient(black_white_matrix)
    
    # Generate the clusters based on the gradient
    cluster_list = generate_clusters(gradient)
    
    # Turn clusters into segments
    segment_array = np.zeros(shape=(len(cluster_list), 4))
    for i in range(len(cluster_list)):
        segment_array[i,:] = cluster_list[i].to_segment()
    return segment_array


def compute_gradient(black_white_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the gradient of each pixel in the image.

    The gradient is calculated using a 3x3 Scharr operator as a convolution
    kernel. The Scharr kernel is similar to a Sobel kernel, but more accurate
    as a 3x3 matrix.

    Args:
        black_white_matrix: a [w x h] numpy array giving the black and white
            image to compute the gradient of
    
    Returns:
        a [w x h x 2] numpy array giving the gradient at each pixel - axes are
            x by y by (magnitude, direction). Direction is in degrees.
    """
    # For each pixel, find gradient x, gradient y, magnitude, direction
    scharr_x_kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    scharr_y_kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
    gradient_array = np.zeros(shape = (np.shape(black_white_matrix)[0], np.shape(black_white_matrix)[1], 2))
    
    # Expects a float for magnitude, None type radian for direction
    x_grads_array = signal.convolve2d(black_white_matrix, scharr_x_kernel, 'same')
    
    # Spits out an array of the same dimensions as the first entered array
    y_grads_array = signal.convolve2d(black_white_matrix, scharr_y_kernel, 'same')
    for y in range(int(np.shape(gradient_array)[0])): # goes column by column
        for x in range(int(np.shape(gradient_array[y])[0])): # goes row by row
            mag = np.sqrt(np.square(x_grads_array[y][x]) + np.square(y_grads_array[y][x]))
            dir = np.arctan2(y_grads_array[y][x], x_grads_array[y][x]) # gives radians
            gradient_array[y][x][0] = mag # dir stored as a float
            gradient_array[y][x][1] = dir 
    return gradient_array


def generate_clusters(mag_dir_matrix: np.ndarray) -> List[Cluster]:
    """
    Finds clusters of pixels.

    Creates a graph where each node is a pixel and the edges are the difference
    in the gradient direction. Combine clusters according to the clustering step
    
    Args:
        mag_dir_matrix: a [w x h x (m,d)] numpy array containing the location
            of each pixel as well as its magnitude and direction.
    
    Returns:
        a list of every Cluster found in the image
    """
    # Create graph
    row_len = mag_dir_matrix.shape[1]
    graph = create_cluster_graph(mag_dir_matrix)
    print('Made cluster graph')

    # Create initial clusters (one per pixel)
    clusters = {i: Cluster(np.array([[i % row_len, i // row_len]]),
                           mag_dir_matrix) for i in graph.nodes}
    next_cluster_num = mag_dir_matrix.shape[0] * mag_dir_matrix.shape[1]

    # Keep track of which cluster each pixel is in
    pixel_to_cluster = np.array(range(len(graph.nodes)))

    # Edges are tested in order of their weight
    sorted_edges = sorted(graph.edges, key=lambda e: graph[e[0]][e[1]]['weight'])
    for edge in sorted_edges:
        if pixel_to_cluster[edge[0]] == pixel_to_cluster[edge[1]]:
            # If the two pixels are in the same cluster, skip them
            continue
        # Get the two clusters and combine them
        cluster1 = clusters[pixel_to_cluster[edge[0]]]
        cluster2 = clusters[pixel_to_cluster[edge[1]]]
        combined = cluster1.combine_clusters(cluster2)

        if combined is None:
            # If combined is None, we should not combine - skip
            continue

        # Delete the old clusters
        del clusters[pixel_to_cluster[edge[0]]]
        del clusters[pixel_to_cluster[edge[1]]]

        # Add the combined cluster to the lookup dictionary
        clusters[next_cluster_num] = combined

        # Find all of the pixels in the combined cluster
        pixel_nums = combined.points[:, 0] + row_len * combined.points[:, 1]

        # Update which cluster each pixel is in
        pixel_to_cluster[pixel_nums] = next_cluster_num

        # Update the number for the next cluster
        next_cluster_num += 1
        
    final_clusters = [c for c in clusters.values() if c.size > 5]
        
    return final_clusters


def create_cluster_graph(mag_dir_matrix: np.ndarray) -> nx.Graph:
    """
    Create the graph to use in clustering the image

    Each node is an integer representing a pixel, with 0 being top left and
    continuing left-to-right, top-to-bottom. There is an edge between each
    adjacent pixel, with the edge weight being equal to the difference in
    directions, in radians.

    Args:
        mag_dir_matrix: a [w x h x (m,d)] numpy array containing the location
            of each pixel as well as its magnitude and direction.

    Returns:
        a networkx Graph of the image to use to create clusters
    """
    # Create a graph that contains one node per pixel
    graph = nx.Graph()
    graph.add_nodes_from(range(mag_dir_matrix.shape[0] * mag_dir_matrix.shape[1]))

    # Loop through all the nodes
    row_len = mag_dir_matrix.shape[1]
    for pixel_num in graph.nodes:
        # Define neighboring pixels
        pixel = pixel_num % row_len, pixel_num // row_len
        neighbors = [
            (pixel[0], pixel[1] + 1),
            (pixel[0], pixel[1] - 1),
            (pixel[0] + 1, pixel[1]),
            (pixel[0] - 1, pixel[1])
        ]
        # Get the direction (angle) of the current pixel
        pixel_angle = mag_dir_matrix[pixel[1], pixel[0], 1]

        # Loop through neighboring pixel nodes
        for neighbor in neighbors:
            if neighbor[0] >= row_len or neighbor[1] >= mag_dir_matrix.shape[0]\
                    or any(n < 0 for n in neighbor):
                continue # Skip neighbors that are outside the image boundaries
            neighbor_num = neighbor[1] * row_len + neighbor[0]

            # Check if an edge already exists between the current pixel and the neighbor
            if graph.has_edge(pixel_num, neighbor_num):
                continue

            # Check if neighbor node exists in the graph
            if neighbor_num in graph.nodes:
                # Calculate the difference in angles between the current pixel and the neighbor
                neighbor_angle = mag_dir_matrix[neighbor[1], neighbor[0], 1]
                weight = angle_diff(pixel_angle, neighbor_angle)
                graph.add_edge(pixel_num, neighbor_num, weight=weight)
    return graph
    
    
def quad_detector(segments: np.ndarray) -> np.ndarray:
    """
    Find possible quadrilaterals in the image given the list of line segments
    
    Args:
        segments: a [n x (x1, y1, x2, y2)] numpy array giving the start and end
            points of each line segment

    Returns:
        a [q x 4 x (x, y)] numpy array containing q quads determined by corners
            listed in (x, y)
    """
    # Build lookup tables
    dist_lookup = line_segment_distances(segments)
    cross_lookup = cross_table(segments)

    # Build and search tree for quads
    tree_graph = generate_tree(segments, dist_lookup, cross_lookup)
    quads = get_quads_from_tree(tree_graph)
    show_tree(tree_graph)
    return quads


def line_segment_distances(segments: np.ndarray) -> np.ndarray:
    """
    Creates a boolean lookup table representing if two segments can be neighboring segments.
    
    Calculates the distance between end point of the first line segment and the
        start point of the next segment. Compares that distance to the threshold
        distance. The threshold is calculates from the length of the first segment
        multiplied by 2 and adding 5.
    
    Args:
        segments: a [n x (x1, y1, x2, y2)] numpy array giving the start and end
            points of each line segment
    
    Returns:
        a [n x n] numpy array dist_lookup that returns True if the start of
        line B is within 2 times with length of line A plus five pixels. A is
        indexed by row, B by column.
    """
    # Find the number of segments
    nsegments = segments.shape[0]

    # Create an [n x n] grid
    a, b = np.meshgrid(np.arange(nsegments), np.arange(nsegments))

    # Calculate the distances between each segment in the grid
    dist_sq = (segments[a, 3] - segments[b, 1]) ** 2 + (segments[a, 2] - segments[b, 0]) ** 2
    dists = np.sqrt(dist_sq)
    lengths = np.diag(dists)

    # Check if the distance is less than the threshold
    dist_lookup = dists < 2 * lengths + 5
    return dist_lookup


def generate_tree(segments: np.ndarray, dist_lookup: np.ndarray, cross_lookup: np.ndarray) -> nx.Graph:
    """
    Generate a tree of line segments to detect quadrilaterals

    The root of the graph points to every possible line segment. From there,
    each child node points to every line segment whose start is "close enough"
    to the parent node's end. Close enough is defined as 2*parent length plus
    five pixels. Additionally each child node must obey the winding order of
    the previous branches. Winding order means that each segment points in the
    same direction. Repeat for four levels deep.

    Args:
        segments: a [n x (x1, y1, x2, y2)] numpy array giving the start and end
            points of each line segment, so that traveling from point 1 to
            point 2 has the light side on the right.
        dist_lookup: a [n x n] numpy array dist_lookup that returns True if the start of
            line B is within 2 times with length of line A plus five pixels. A is
            indexed by row, B by column.
        cross_lookup: a [n x n] numpy array that returns the direction of two vectors as
            a -1, 0, or 1. 
    
    Returns:
        a NetworkX graph giving the tree of line s
    """
    graph = nx.DiGraph()

    # Add the root node to the graph
    root = "Root"
    graph.add_node(root)

    # Level 1: add all segments as node to root
    for i in range(segments.shape[0]):
        child = (i, 1)
        graph.add_edge(root, child)
        # Level 2: add only new segments, check distances and winding order
        for j in range(segments.shape[0]):
            if j != i and dist_lookup[i][j] and cross_lookup[i][j] != 0:
                child2 = (j, i, 2)
                graph.add_edge(child, child2)
                # Level 3: add only new segments, check distances and winding order
                for k in range(segments.shape[0]):
                    if k != i and k != j and dist_lookup[j][k] and cross_lookup[j][k] == cross_lookup[i][j]:
                        child3 = (k, j, i, 3)
                        graph.add_edge(child2, child3)
                        # Level 4: add only new segments, check distances and winding order
                        for l in range(segments.shape[0]):
                            if l != i and l != k and l != j and dist_lookup[k][l] and cross_lookup[k][l] == cross_lookup[i][j]:
                                child4 = (l, k, j , i, 4)
                                graph.add_edge(child3, child4)
    return graph 


def get_quads_from_tree(segment_tree: nx.Graph) -> np.ndarray:
    """
    Takes tree and conducts a depth-first search for possible quadrilaterals
        based on number of line segments.

    Finds all branches with exactly four nodes and appends them to quads. Each value
    in the quad is the index of the segment from the segment list.

    Args:
        tree: a NetworkX graph that represents all potential combinations of segments
        
    Return:
        a [q x 4 x (x1, y1, x2, y2)] numpy array where 4 x (x1, y1, x2, y2)
            determines the four line segments making up the quad
    """
    quads = []
    for root in segment_tree.nodes():
        quads = dfs(root, [], segment_tree, quads)
    return quads
    

def dfs(node, path, segment_tree, quads) -> np.ndarray:
    """
    Depth first searches a tree for all paths containing four nodes.

    Finds all branches with exactly four nodes and appends them to quads. Each value
    in the quad is the index of the segment from the segment list. Removes any repeat
    quads before appending.

    Args:
        node: the next node that will be added to a path
        path: a list that contains the indexes for a potential quad
        segment_tree: a NetworkX graph that represents all potential combinations of segments
        quads: a [q x 4 x (x1, y1, x2, y2)] numpy array that each new quad
            gets added to

    Return:
        a [q x 4 x (x1, y1, x2, y2)] numpy array where 4 x (x1, y1, x2, y2)
            determines the four line segments making up the quad
    """
    # Check if all four segments have been found
    if len(path) == 4:
        if path[0] == min(path):    # Prevents quads with the same segment from existing
            quads.append(path)
        return
    # Checks all children of a node
    for child in list(segment_tree.successors(node)):
        dfs(child, path + [child[0]], segment_tree, quads)
    return np.array(quads)


def cross_table(segments: np.ndarray)-> np.ndarray:
    """
    Creates a lookup table of the signs of the cross product between two line segments.

    The sign of the cross product between segments represents the direction of the lines.
        If all of the signs match between line segments than it could be a valid shape.
    
    Args:
        segments: a [n x (x1, y1, x2, y2)] numpy array giving the start and end
            points of each line segment
    
    Returns:
        a [n x n] numpy array cross_table that returns -1, 0, or 1. 
    """
    # Initialize cross_table with zeros
    num_segments = len(segments)
    cross_table = np.zeros((num_segments, num_segments))

    # Loop through every value
    for i in range(len(segments)):
        for j in range(len(segments)):
            parent = np.array([segments[i][2]-segments[i][0], segments[i][3]-segments[i][1]])
            child = np.array([segments[j][2]-segments[j][0], segments[j][3]-segments[j][1]])
            # Calculate the cross product between segments and store its sign
            cross_table[i][j] = np.sign(np.cross(parent, child))
    return cross_table


def show_tree(tree):
    """
    Display a graph with a graphical interface

    tree: a NetworkX graph that represents all potential combinations of segments 
    """
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


def main():
    # Load in the picture
    color_matrix = cv2.imread('images/black_rectangle.png')

    # Call the april tag detector
    # quads = april_tag_detector(color_matrix)

    # segments = np.array([[1, 0, 437, 0],
    #                      [381, 105, 59, 105],
    #                      [58, 112, 58, 221],
    #                      [381, 163, 58, 164],
    #                      [382, 221, 382, 107],
    #                      [383, 221, 383, 109],
    #                      [384, 221, 384, 108],
    #                      [59, 222, 381, 222],
    #                      [0, 326, 0, 1],
    #                      [0, 163, 437, 163],
    #                      [438, 1, 438, 326],
    #                      [437, 327, 1, 327]])

    segments = np.array([[0, 0, 0, 1],
                         [0, 1, 1, 1],
                         [1, 1, 1, 0],
                         [1, 0, 0, 0],
                         [-5, -5, -5, 5],
                         [-5, 5, 5, 5],
                         [5, 5, 5, -5],
                         [5, -5, -5, -5]])
    

    print(segments.shape)
    quads_by_idx = quad_detector(segments)
    print(quads_by_idx)
    print(quads_by_idx.shape)

    for quad in quads_by_idx:
        points = np.array([[segments[quad[0], 2], segments[quad[0], 3]],
                            [segments[quad[0], 0], segments[quad[0], 1]],
                            [segments[quad[1], 2], segments[quad[1], 3]],
                            [segments[quad[1], 0], segments[quad[1], 1]],
                            [segments[quad[2], 2], segments[quad[2], 3]],
                            [segments[quad[2], 0], segments[quad[2], 1]],
                            [segments[quad[3], 2], segments[quad[3], 3]],
                            [segments[quad[3], 0], segments[quad[3], 1]]])
        plt.plot(points[:, 0], points[:, 1])
    plt.show()

    

if __name__ == '__main__':
    main()

