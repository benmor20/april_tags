"""
Module for working with angles and ranges of angles
"""
from typing import *
import numpy as np


TAU = 2 * np.pi


def range_from_angles(angles: Iterable[float]) -> Tuple[float, float]:
    """
    Find the smallest range of angles that includes all angles in the given list

    Specifically, sorts the angles list and finds two adjacent angles which are
    as far apart as possible. The other side of that range is the smallest
    possible range.

    :param angles: a list of angles from -pi to pi
    :return: a tuple defining the start and end of a range that contains every
        angle in the list; a CCW sweep from the first element to the second
        element of the tuple will cover every angle in the list
    """
    sorted_angles = sorted(angles)
    angle_pairs = zip(sorted_angles[1:] + [TAU + sorted_angles[0]], angles)
    hi, lo = max(angle_pairs, key=lambda p: p[0] - p[1])
    if hi > np.pi:
        hi = sorted_angles[0]
    return hi, lo


def angle_in_range(angle: float, ang_range: Tuple[float, float]) -> bool:
    """
    Determine whether the given angle is within the given range

    The range is given as a tuple such that traveling CCW from the first element
    to the second traverses the range

    :param angle: a float from -pi to pi, what to determine is in range
    :param ang_range: an angle range, the range to check if angle is in
    :return: True if angle is in ang_range, False otherwise
    """
    lo, hi = ang_range
    if lo < hi:
        return lo <= angle <= hi
    # Includes the cut from pi to -pi; slightly different logic here
    return angle >= lo or angle <= hi


def range_size(ang_range: Tuple[float, float]) -> float:
    """
    Calculate the size of the given angle range

    :param ang_range: a tuple of two floats giving the start and end of a range
    :return: a float, the size of the given range
    """
    length = ang_range[1] - ang_range[0]
    if length < 0:
        return length + TAU
    return length


def combine_angle_ranges(arange: Tuple[float, float],
                         brange: Tuple[float, float]) -> Tuple[float, float]:
    """
    Given two ranges of angles, find the range of their union

    :param arange: a tuple defining the first range to combine
    :param brange: a tuple defining the second range to combine
    :return: a range giving the range of the arange union brange
    """
    # Only four possible options
    ranges = [
        arange,
        (arange[0], brange[1]),
        (brange[0], arange[1]),
        brange
    ]
    # Remove any ranges that do not contain all of the angles
    filt_func = lambda r: all(angle_in_range(a, r) for a in (*arange, *brange))
    filtered_ranges = filter(filt_func, ranges)
    # Of the remaining ranges, return the smallest
    return min(filtered_ranges, key=range_size)


def angle_diff(angle1: float, angle2: float):
    """
    Compute the shortest difference between two angles

    :param angle1: a float, the first angle
    :param angle2: a float, the second angle
    :return: a float, the smallest difference between the two angles
    """
    diff = abs(angle1 - angle2)
    if diff > np.pi:
        return TAU - diff
    return diff
