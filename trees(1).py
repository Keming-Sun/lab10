from typing import List
from collections import namedtuple
import time
import math


# Set of points
class Point(namedtuple("Point", "x y")):
    def __repr__(self) -> str:
        return f'Point{tuple(self)!r}'


# Rectangular region
class Rectangle(namedtuple("Rectangle", "lower upper")):
    def __repr__(self) -> str:
        return f'Rectangle{tuple(self)!r}'

    # Determines whether it is in a rectangular area
    def is_contains(self, p: Point) -> bool:
        return self.lower.x <= p.x <= self.upper.x and self.lower.y <= p.y <= self.upper.y


# Save the value and the left and right nodes separately
class Node(namedtuple("Node", "location left right")):
    """
    location: Point
    left: Node
    right: Node
    """

    def __repr__(self):
        return f'{tuple(self)!r}'


class KDTree:
    """k-d tree"""

    # Initialize, leaving the header node empty with size 0
    def __init__(self):
        self._root = None
        self._n = 0

    def insert(self, p: List[Point]):
        """insert a list of points"""

        # The comparison is regular according to the latitude change
        def __sort_x(element):
            return element[0]

        def __sort_y(element):
            return element[1]

        def insert_points(_points, depth=0, root=None):

            # No element represents an empty tree
            if len(_points) == 0:
                return None

            # Get the dimension of the secondary judgment
            dimension = depth % 2
            if dimension == 0:
                _points.sort(key=__sort_x)
            else:
                _points.sort(key=__sort_y)

            # The median is the node
            mid = len(_points) // 2
            mid_point = _points[mid]

            # Insert the point
            if root is None:
                root_node = Node(mid_point, None, None)
                # Determine whether to insert the right subtree or the left subtree based on the comparison of
                # dimensions
                left = insert_points(_points[0:mid], depth + 1)
                right = insert_points(_points[mid + 1:].copy(), depth + 1)

                return Node(root_node.location, left, right)
            else:
                if mid_point[dimension] < root.location[dimension]:
                    root.left = insert_points(_points[:mid], depth + 1)
                    root.right = insert_points(_points[mid:], depth + 1)
                else:
                    root.left = insert_points(_points[:mid + 1], depth + 1)
                    root.right = insert_points(_points[mid + 1:], depth + 1)
                    return root

        self._root = insert_points(p, 0, self._root)
        self._n += len(p)

    def range(self, rectangle: Rectangle) -> List[Point]:
        """range query"""

        def _range(root, depth=0):

            # Nonexistent element
            if root is None:
                return []

            result = list()
            # The point is within the given dimension in all dimensions
            location = root.location
            if rectangle.is_contains(location):
                result.append(root.location)

            # Determine which subtree to query next based on the current dimension
            dimension = depth % 2
            if rectangle.lower[dimension] <= root.location[dimension]:
                result.extend(_range(root.left, depth + 1))
            if rectangle.upper[dimension] >= root.location[dimension]:
                result.extend(_range(root.right, depth + 1))
            return result

        return _range(self._root)

    # Find the closest point to the given point
    def nearest_neighbor(self, point):
        # Get the distance between two points a and b
        def distance(a, b):
            return math.sqrt(
                (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            )

        def _nearest_neighbor(root, target, _nearest, depth=0):
            if root is None:
                return _nearest

            # Obtain the corresponding length from the point to the given point

            if _nearest is None:
                dis = distance(root.location, target)
                _nearest = (root.location, dis)
            dis = distance(root.location, target)
            if dis < _nearest[1]:
                _nearest = (root.location, dis)

            # Judgment corresponding dimension
            dimension = depth % 2

            if target[dimension] < root.location[dimension]:
                next_point_nearest = _nearest_neighbor(root.left, target, _nearest, depth + 1)
            else:
                next_point_nearest = _nearest_neighbor(root.right, target, _nearest, depth + 1)

            if next_point_nearest[1] < _nearest[1]:
                _nearest = next_point_nearest

            hyperplane_dist = abs(root.location[dimension] - target[dimension])

            # Check whether the distance between two points in this
            # dimension is less than the shortest distance obtained earlier
            if hyperplane_dist < _nearest[1]:

                # Less than indicates that the subtree on the undetected side may have a smaller distance
                if target[dimension] < root.location[dimension]:
                    next_point_nearest = _nearest_neighbor(root.right, target, _nearest, depth + 1)
                else:
                    next_point_nearest = _nearest_neighbor(root.left, target, _nearest, depth + 1)

                if next_point_nearest[1] < _nearest[1]:
                    _nearest = next_point_nearest
            return _nearest

        nearest = None
        point, distance = _nearest_neighbor(self._root, point, nearest)
        return Point(point[0], point[1]), distance


def range_test():
    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    result = kd.range(Rectangle(Point(0, 0), Point(6, 6)))
    assert sorted(result) == sorted([Point(2, 3), Point(5, 4)])


def performance_test():
    points = [Point(x, y) for x in range(1000) for y in range(1000)]

    lower = Point(500, 500)
    upper = Point(504, 504)
    rectangle = Rectangle(lower, upper)
    #  naive method
    start = int(round(time.time() * 1000))
    result1 = [p for p in points if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000))
    print(f'Naive method: {end - start}ms')

    kd = KDTree()
    kd.insert(points)
    # k-d tree
    start = int(round(time.time() * 1000))
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000))

    print(f'K-D tree: {end - start}ms')

    assert sorted(result1) == sorted(result2)


def nearest_neighbor_test():
    """
    
    5. Implement the nearest neighbor query

        1. Start at the root node of the tree。

        2. Calculate the distance between the point at the current node and the target point.
           If this distance is less than the current nearest neighbor distance,
           the current nearest neighbor is set to the point at the current node.

        3. Search recursively down the left and right subtrees, looking for points that are shorter distances away

        4. Search for the nearest subtree in the corresponding dimension.
           If the distance between two points in this dimension is less than the shortest distance obtained previously,
           then another subtree may have the shortest distance.
    """

    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    nearser = Point(5, 1)
    result, distance = kd.nearest_neighbor(nearser)
    # The closest point to this point is
    print(f"The closest point {nearser} to this point is {result}.The distance is {distance}")
    assert result == Point(7, 2)


if __name__ == '__main__':
    range_test()
    performance_test()
    nearest_neighbor_test()

"""



2 

    insert
        There are two situations at the insertion point:
        
        1. The node does not have a value. After sorting the array, 
           the midpoint of the dimension is taken out, and then the midpoint is divided. 
           The array before the midpoint is inserted into the left subtree, and then into the right subtree. 
           Repeat until all inserts are complete.
           
        2. If the node exists, the data is sorted and divided by the midpoint. 
           The array before the midpoint is inserted into the left subtree, 
           and the array after the midpoint is inserted into the right subtree. 
           Repeat the above operations until the first case occurs, and then the first case.
        
        Note: The dimension of each sort comparison can be depth%k. For example, 
        the point created this time is a two-dimensional point, so k=2

    range
        You have to have a minimum point and a maximum point to get the interval.
        
        First, determine whether each dimension of the point of the node is within the given dimension range. 
        If so, add the point to the list; otherwise, do not add it to the list.

        To determine whether the point in this dimension is not less than the given minimum value, 
        if not, then search the left subtree.
        
        Then judge whether the point is not greater than the given maximum value in this dimension.
        If not, search the right subtree.
        
        Then judge whether the hyperplane distance of another subtree is less than the minimum distance at present.
        If it is less, the other subtree may have a smaller value, so it also needs to search.
        
        Repeat until the entire tree has been searched.
        
"""

"""

        

3. When doing range queries, the time complexity is affected by the balance of the tree.
    The best case
       The tree presents a perfectly balanced state, when the time complexity is O(log n)
        
    The worst case
        The tree presents a completely unbalanced state.
        At this time, it is necessary to check whether all points are located in the given interval,
        and the time complexity is O(n)
"""
