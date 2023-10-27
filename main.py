"""
Module to detect April Tags
"""
import itertools

import cv2
import numpy as np
import networkx as nx
from typing import *
from cluster import Cluster
from angle_utils import angle_diff


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
    pass


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
    cluster_list = generate_clusters(compute_gradient(black_white_matrix))
    segment_array = np.zeros(shape=(len(cluster_list), 1) dtype = "i, i, i, i")
        #init blank array, will accept tuples of 4 ints
        #for normal array, change 1 to 4 and delete dtype bit
    for i in range(len(cluster_list)):
        segment_array[i,:] = cluster_to_segment(cluster_list[i])
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
    
    scharr_x_kernel = np.array([-3, 0, 3], [-10, 0, 10], [-3, 0, 3])
    scharr_y_kernel = np.array([-3, -10, -3], [0, 0, 0], [3, 10, 3])
    gradient_array = np.zeros(shape = (np.shape(black_white_matrix)[0], np.shape(black_white_matrix)[1], 1), dtype = "f, f")
        # expects a float for magnitude, None type radian for direction
    x_grads_array = np.convolve(scharr_x_kernel, black_white_matrix, 'same')
        # spits out an array of the same dimensions as the larger entered array
    y_grads_array = np.convolve(scharr_y_kernel, black_white_matrix, 'same')
    for y in range(int(np.shape(gradient_array)[0])): # goes column by column
        for x in range(int(np.shape(gradient_array[y])[0])): # goes row by row
            mag = np.sqrt(np.square(x_grads_array[y][x]) + np.square(y_grads_array[y][x]))
            dir = np.arctan2(y_grads_array[y][x], x_grads_array[y][x]) # gives radians
            gradient_array[y][x][0] = (mag, dir) # dir stored as a float
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
    row_len = mag_dir_matrix.shape[1]
    graph = create_cluster_graph(mag_dir_matrix)
    clusters = {i: Cluster(np.array([[i % row_len, i // row_len]]),
                           mag_dir_matrix) for i in graph.nodes}
    next_cluster_num = mag_dir_matrix.size // 2
    pixel_to_cluster = np.array(range(len(graph.nodes)))
    sorted_edges = sorted(graph.edges, key=lambda e: graph[e[0]][e[1]]['weight'])
    for edge in sorted_edges:
        if pixel_to_cluster[edge[0]] == pixel_to_cluster[edge[1]]:
            continue
        cluster1 = clusters[pixel_to_cluster[edge[0]]]
        cluster2 = clusters[pixel_to_cluster[edge[1]]]
        combined = cluster1.combine_clusters(cluster2)
        if combined is None:
            print(f'Clusters {pixel_to_cluster[0]} and {pixel_to_cluster[1]} will not be combined')
            continue
        print(f'Combining clusters {pixel_to_cluster[edge[0]]} and {pixel_to_cluster[edge[1]]} into {next_cluster_num}')
        del clusters[pixel_to_cluster[edge[0]]]
        del clusters[pixel_to_cluster[edge[1]]]
        clusters[next_cluster_num] = combined
        pixel_nums = combined.points[:, 0] + row_len * combined.points[:, 1]
        pixel_to_cluster[pixel_nums] = next_cluster_num
        next_cluster_num += 1
    return [c for c in clusters.values() if c.size > 1]


def create_cluster_graph(mag_dir_matrix: np.ndarray) -> nx.Graph:
    """
    Create the graph to use in clustering the image

    Each node represents a pixel, with 0 being top left and continuing
    left-to-right, top-to-bottom. There is an edge between each adjacent pixel,
    with the edge weight being equal to the difference in directions, in
    radians.

    Args:
        mag_dir_matrix: a [w x h x (m,d)] numpy array containing the location
            of each pixel as well as its magnitude and direction.

    Returns:
        a networkx Graph of the image to use to create clusters
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(mag_dir_matrix.size // 2))
    row_len = mag_dir_matrix.shape[1]
    for pixel_num in graph.nodes:
        pixel = divmod(pixel_num, row_len)[::-1]
        neighbors = [
            (pixel[0], pixel[1] + 1),
            (pixel[0], pixel[1] - 1),
            (pixel[0] + 1, pixel[1]),
            (pixel[0] - 1, pixel[1])
        ]
        pixel_angle = mag_dir_matrix[pixel[0], pixel[1], 1]
        for neighbor in neighbors:
            if neighbor[0] >= row_len or neighbor[1] >= mag_dir_matrix.shape[0]\
                    or any(n < 0 for n in neighbor):
                continue
            neighbor_num = neighbor[1] * row_len + neighbor[0]
            if graph.has_edge(pixel_num, neighbor_num):
                continue
            if neighbor_num in graph.nodes:
                neighbor_angle = mag_dir_matrix[neighbor[0], neighbor[1], 1]
                weight = angle_diff(pixel_angle, neighbor_angle)
                graph.add_edge(pixel_num, neighbor_num, weight=weight)
    return graph


def cluster_to_segment(cluster: Cluster) -> Tuple[int, int, int, int]:  # Ben
    """
    Fits a line segment to a given cluster via linear regression. Identifies
        endpoints from points at extremes of line.

    Args:
        cluster: the Cluster to make into a line segment

    Returns:
        a (x1, y1, x2, y2) tuple of endpoints based on the line of best fit for
            the input cluster. the two points are ordered so that when
            traveling from point 1 to point 2, the light side is on the right
    """
    return cluster.to_segment()
    
    
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
    line_segment_distances()
    generate_tree()
    get_quads_from_tree()


def line_segment_distances(segments: np.ndarray) -> np.ndarray: # Anusha
    """
    Creates a lookup table of distances between the end of one line segment to
        the start of all others. Does this for all line segments found via
        segmentation function.
    
    Args:
        segments: a [n x (x1, y1, x2, y2)] numpy array giving the start and end
            points of each line segment
    
    Returns:
        a [n x n] numpy array giving the distance (in pixels) between the end
        line A and the start of line B for all A and B. A is indexed by row, B
        by column. Points along the diagonal give the length of the line.
    """
    pass


def generate_tree(segments: np.ndarray, dist_lookup: np.ndarray) -> nx.Graph:   # Maya
    """
    Generate a tree of line segments to detect quadrilaterals from

    The root of the graph points to every possible line segment. From there,
    each child node points to every line segment whose start is "close enough"
    to the parent node's end. Close enough is defined as 2*parent length plus
    five pixels. Repeat for four levels deep.

    Args:
        segments: a [n x (x1, y1, x2, y2)] numpy array giving the start and end
            points oof each line segment, so that traveling from point 1 to
            point 2 has the light side on the right.
        dist_lookup: a [n x n] numpy array giving the distance (in pixels)
            between the end line A and the start of line B for all A and B. A
            is indexed by row, B by column. Points along the diagonal give
            the length of the line.
    
    Returns:
        a NetworkX graph giving the tree of line s
    """
    pass


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
    pass


def main():
    pass
    

if __name__ == '__main__':
    main()

