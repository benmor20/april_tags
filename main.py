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
    print(quad_detector(segments))


def segmentation(black_white_matrix: np.ndarray) -> np.ndarray: # Alana
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
    gradient = compute_gradient(black_white_matrix)
    print('Found gradient')
    cluster_list = generate_clusters(gradient)
    print('Generated clusters')
    segment_array = np.zeros(shape=(len(cluster_list), 4))
    for i in range(len(cluster_list)):
        segment_array[i,:] = cluster_list[i].to_segment()
    return segment_array

def compute_gradient(black_white_matrix: np.ndarray) -> np.ndarray: # Alana
    """
    Calculate the gradient of each pixel in the image.

    The gradient is calculated using a 3x3 Scharr operator as a convolution
    kernel.

    Args:
        black_white_matrix: a [w x h] numpy array giving the black and white
            image to compute the gradient of
    
    Returns:
        a [w x h x 2] numpy array giving the gradient at each pixel - axes are
            x by y by (magnitude, direction). Direction is in degrees.
    """
    # for each pixel, find gradient x, gradient y, magnitude, direction
    # there is a sobel operator built into cv2 which might be easier to use
    # Scharr kernel is similar to Sobel kernel, but more accurate as a 3x3 matrix
    
    scharr_x_kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    scharr_y_kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
    gradient_array = np.zeros(shape = (np.shape(black_white_matrix)[0], np.shape(black_white_matrix)[1], 2))
        # expects a float for magnitude, None type radian for direction
    x_grads_array = signal.convolve2d(black_white_matrix, scharr_x_kernel, 'same')
        # spits out an array of the same dimensions as the first entered array
    y_grads_array = signal.convolve2d(black_white_matrix, scharr_y_kernel, 'same')
    for y in range(int(np.shape(gradient_array)[0])): # goes column by column
        for x in range(int(np.shape(gradient_array[y])[0])): # goes row by row
            mag = np.sqrt(np.square(x_grads_array[y][x]) + np.square(y_grads_array[y][x]))
            dir = np.arctan2(y_grads_array[y][x], x_grads_array[y][x]) # gives radians
            gradient_array[y][x][0] = mag # dir stored as a float
            gradient_array[y][x][1] = dir 
    return gradient_array


def generate_clusters(mag_dir_matrix: np.ndarray) -> List[Cluster]:  # Ben
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
    # print('Plotting clusters')
    # for cluster in final_clusters:
    #     plt.scatter(cluster.points[:, 0], cluster.points[:, 1])
    #     plt.xlim([-10, mag_dir_matrix.shape[1] + 10])
    #     plt.ylim([-10, mag_dir_matrix.shape[0] + 10])
    #     plt.show()
        
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
    graph = nx.Graph()
    graph.add_nodes_from(range(mag_dir_matrix.shape[0] * mag_dir_matrix.shape[1]))
    row_len = mag_dir_matrix.shape[1]
    for pixel_num in graph.nodes:
        pixel = pixel_num % row_len, pixel_num // row_len
        neighbors = [
            (pixel[0], pixel[1] + 1),
            (pixel[0], pixel[1] - 1),
            (pixel[0] + 1, pixel[1]),
            (pixel[0] - 1, pixel[1])
        ]
        pixel_angle = mag_dir_matrix[pixel[1], pixel[0], 1]
        for neighbor in neighbors:
            if neighbor[0] >= row_len or neighbor[1] >= mag_dir_matrix.shape[0]\
                    or any(n < 0 for n in neighbor):
                continue
            neighbor_num = neighbor[1] * row_len + neighbor[0]
            if graph.has_edge(pixel_num, neighbor_num):
                continue
            if neighbor_num in graph.nodes:
                neighbor_angle = mag_dir_matrix[neighbor[1], neighbor[0], 1]
                weight = angle_diff(pixel_angle, neighbor_angle)
                graph.add_edge(pixel_num, neighbor_num, weight=weight)
    return graph
    
    
def quad_detector(segments: np.ndarray) -> np.ndarray:  # Anusha
    """
    Find possible quadrilaterals in the image given the list of line segments
    
    Args:
        segments: a [n x (x1, y1, x2, y2)] numpy array giving the start and end
            points of each line segment

    Returns:
        a [q x 4 x (x, y)] numpy array containing q quads determined by corners
            listed in (x, y)
    """
    dist_lookup = line_segment_distances(segments)
    print(dist_lookup)
    cross_lookup = cross_table(segments)
    tree_graph = generate_tree(segments, dist_lookup, cross_lookup)
    quads = get_quads_from_tree(tree_graph)
    show_tree(tree_graph)
    return quads


def line_segment_distances(segments: np.ndarray) -> np.ndarray: # Anusha
    """
    Creates a lookup table of distances between the end of one line segment to
        the start of all others. Does this for all line segments found via
        segmentation function.
    
    Args:
        segments: a [n x (x1, y1, x2, y2)] numpy array giving the start and end
            points of each line segment
    
    Returns:
        a [n x n] numpy array dist_lookup that returns True if the start of
        line B is within 2 times with length of line A plus five pixels. A is
        indexed by row, B by column. Points along the diagonal return True.
    """
    nsegments = segments.shape[0]
    a, b = np.meshgrid(np.arange(nsegments), np.arange(nsegments))
    dist_sq = (segments[a, 3] - segments[b, 1]) ** 2 + (segments[a, 2] - segments[b, 0]) ** 2
    dists = np.sqrt(dist_sq)
    lengths = np.diag(dists)
    dist_lookup = dists < 2 * lengths + 5
    return dist_lookup


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

    # Add the root node to the graph
    root = "Root"
    graph.add_node(root)

    # level 1
    for i in range(segments.shape[0]):
        child = (i, 1)
        graph.add_edge(root, child)
        # level 2
        for j in range(segments.shape[0]):
            if j != i and dist_lookup[i][j] and cross_lookup[i][j] != 0:
                child2 = (j, i, 2)
                graph.add_edge(child, child2)
                # level 3
                for k in range(segments.shape[0]):
                    if k != i and k != j and dist_lookup[j][k] and cross_lookup[j][k] == cross_lookup[i][j]:
                        child3 = (k, j, i, 3)
                        graph.add_edge(child2, child3)
                        # level 4
                        for l in range(segments.shape[0]):
                            if l != i and l != k and l != j and dist_lookup[k][l] and cross_lookup[k][l] == cross_lookup[i][j]:
                                child4 = (l, k, j , i, 4)
                                graph.add_edge(child3, child4)
    return graph 


def get_quads_from_tree(segment_tree: nx.Graph) -> np.ndarray:  # Maya
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
        quads = dfs(root, [], segment_tree, quads)
    return quads
    

# def dfs(node, path, segment_tree, quads):
#     """
#     Doc string here
#     """
#     if len(path) == 4:
#         path.sort()
#         if path not in quads:
#             quads.append(path)
#         return
#     for child in list(segment_tree.successors(node)):
#         dfs(child, path + [child[0]], segment_tree, quads)
#     return np.array(quads)

def dfs(node, path, segment_tree, quads):
    """
    Doc string here
    """
    if len(path) == 4:
        # path.sort()
        if path[0] == min(path):
            quads.append(path)
        return
    for child in list(segment_tree.successors(node)):
        dfs(child, path + [child[0]], segment_tree, quads)
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
            cross_table[i][j] = np.sign(np.cross(parent, child))
    return cross_table


def main():
    color_matrix = cv2.imread('images/black_rectangle.png')
    print(color_matrix.shape)
    bw_image = cv2.cvtColor(color_matrix, cv2.COLOR_BGR2GRAY)

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

    # april_tag_detector(color_matrix)
    

if __name__ == '__main__':
    main()

