
from typing import *
import numpy as np
from angle_utils import *


DIR_RANGE_CONST = 100
MAG_RANGE_CONST = 1200


class Cluster:
    """
    A representation of a cluster of pixels whose gradients have similar
        magnitude and direction.

    Attributes (Read Only):
        size: an int, the number of pixels in this cluster
        points: an (n x 2) numpy array specifying the coordinates of the pixels
            in this cluster
        gradient: an (n x 2) numpy array giving the magnitude and direction of
            each point in this cluster
        mag_limits: a tuple giving the minimum and maximum gradient magnitude in
            this cluster
        mag_range: a float, the difference between the minimum and maximum
            gradient magnitude in this cluster
        dir_limits: a tuple giving the minimum and maximum gradient direction in
            this cluster
        dir_range: a float, the difference between the minimum and maximum
            gradient direction in this cluster
    """

    def __init__(self, points: Optional[np.ndarray] = None,
                 gradient: Optional[np.ndarray] = None):
        """
        Create a new cluster given the points in this cluster and the gradient
            of pixels.

        :param points: an (n x 2) numpy array giving the coordinates of pixels
            in this cluster, or None to create an empty cluster
        :param gradient: an (n x 2) numpy array giving the magnitude and
            direction of the gradient of each given pixel. Or, an (x x y x 2)
            numpy array giving the magnitude and direction of the gradient for
            the full image. Must be given if points is specified
        """
        if points is None:
            self._size = 0
            self._points = np.zeros((0, 2))
            self._gradient = np.zeros((0, 2))
            nan = float('NaN')
            self._mag_limits = (nan, nan)
            self._mag_range = nan
            self._dir_limits = (nan, nan)
            self._dir_range = nan
        else:
            self._size = points.shape[0]
            self._points = points
            if gradient.ndim == 3:
                gradient = gradient[points[:, 0], points[:, 1], :]
            self._gradient = gradient
            min_limits = np.min(gradient, axis=0)
            max_limits = np.max(gradient, axis=0)
            self._mag_limits = min_limits[0], max_limits[0]
            self._dir_limits = range_from_angles(gradient[:, 1])
            self._mag_range = self._mag_limits[1] - self._mag_limits[0]
            self._dir_range = range_size(self._dir_limits)

    def __len__(self) -> int:
        return self.size

    @property
    def size(self) -> int:
        return self._size

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def gradient(self) -> np.ndarray:
        return self._gradient

    @property
    def mag_limits(self) -> Tuple[float, float]:
        return self._mag_limits

    @property
    def mag_range(self) -> float:
        return self._mag_range

    @property
    def dir_limits(self) -> Tuple[float, float]:
        return self._dir_limits

    @property
    def dir_range(self) -> float:
        return self._dir_range

    def combine_clusters(self, other: 'Cluster') -> Optional['Cluster']:
        """
        Compute whether this cluster should be combined with another, and
            combine them if so

        It is assumed that the other cluster has at least one adjacent point to
        this cluster and no overlap. Two clusters should combine if the
        resultant range (for both magnitude and direction) is not bigger than
        the smaller cluster's range plus some constant over the size of the
        new cluster.

        :param other: the Cluster to possibly combine this cluster with
        :return: the combined cluster, or None if the clusters should not be
            combined
        """
        combination = self + other
        target_dir_range = min(self.dir_range, other.dir_range) \
                           + DIR_RANGE_CONST / combination.size
        target_mag_range = min(self.mag_range, other.mag_range) \
                           + MAG_RANGE_CONST / combination.size
        if combination.dir_range > target_dir_range \
                or combination.mag_range > target_mag_range:
            return None
        return combination

    def __add__(self, other: 'Cluster') -> 'Cluster':
        """
        Create a new cluster that would result from the combination of this
            with other

        Combines the points and gradients, and updates the remaining variables
        accordingly. It is assumed that the two clusters have at least one
        adjacent pixel between them and no overlap.

        :param other: the cluster to add to this one
        :return: the cluster that would result from combining this with other
        """
        res = Cluster()
        res._points = np.concatenate((self.points, other.points))
        res._gradient = np.concatenate((self.gradient, other.gradient))
        res._size = self.size + other.size
        max_mag = max(self.mag_limits[1], other.mag_limits[1])
        min_mag = min(self.mag_limits[0], other.mag_limits[0])
        res._mag_limits = min_mag, max_mag
        res._mag_range = max_mag - min_mag
        res._dir_limits = combine_angle_ranges(self.dir_limits, other.dir_limits)
        res._dir_range = range_size(res._dir_limits)
        return res

    def to_segment(self) -> Tuple[int, int, int, int]:
        """
        Find a line segment that best fits this cluster via linear regression

        Returns:
            a (x1, y1, x2, y2) tuple of endpoints based on the line of best fit for
                the input cluster. the two points are ordered so that when
                traveling from point 1 to point 2, the light side is on the right
        """
        # Compute line
        try:
            fit = np.polyfit(self.points[:, 0], self.points[:, 1], 1)
            slope = fit[0]
            yint = fit[1]
            yint_vec = np.array([0, yint])
            adj_pts = self.points - yint_vec
            unit_vec = np.array([1, slope]) / np.hypot(1, slope)
        except ValueError:
            unit_vec = np.array([0, 1])
            adj_pts = self.points
            yint_vec = np.array([0, 0])
        projected = adj_pts @ unit_vec
        min_pt = np.min(projected) * unit_vec + yint_vec
        max_pt = np.max(projected) * unit_vec + yint_vec
        segment = int(min_pt[0]), int(min_pt[1]), int(max_pt[0]), int(max_pt[1])
        seg_vec = segment[2] - segment[0], segment[3] - segment[1]

        # Flip line if needed
        mid_dir = (self.dir_limits[0] + self.dir_limits[1]) / 2
        if mid_dir > self.dir_limits[0] or mid_dir < self.dir_limits[1]:
            # if mid_dir is out of range, flip it
            # Should only happen if pi is in range
            mid_dir += np.pi
            if mid_dir > np.pi:
                mid_dir -= TAU
        seg_angle = np.arctan2(seg_vec[1], seg_vec[0])
        target_dir = seg_angle - np.pi / 2
        anti_target = target_dir + np.pi
        if target_dir < -np.pi:
            target_dir += TAU
        if anti_target > np.pi:
            anti_target -= TAU
        if abs(mid_dir - anti_target) < abs(mid_dir - target_dir):
            segment = segment[2], segment[3], segment[0], segment[1]
        return segment
