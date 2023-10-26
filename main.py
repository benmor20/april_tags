"""
Module to detect April Tags
"""

import cv2
import numpy as np
import networkx as nx
from typing import *
from cluster import Cluster


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
    pass


def compute_gradient(black_white_matrix: np.ndarray) -> np.ndarray: # Alana
    """
    Calculate the gradient of each pixel in the image.

    The gradient is calculated using a 3x3 Sobel operator as a convolution
    kernel.

    Args:
        black_white_matrix: a [w x h] numpy array giving the black and white
            image to compute the gradient of
    
    Returns:
        a [w x h x 2] numpy array giving the gradient at each pixel - axes are
            x by y by (magnitude, direction)
    """
    pass


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
    pass


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

