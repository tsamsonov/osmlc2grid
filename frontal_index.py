import math
import numpy as np


# borrowed from tryalgo library
# https://jilljenn.github.io/tryalgo/_modules/tryalgo/union_rectangles.html#union_rectangles
class CoverQuery:
    """Segment tree to maintain a set of integer intervals
    and permitting to query the size of their union.
    """

    def __init__(self, L):
        """creates a structure, where all possible intervals
        will be included in [0, L - 1].
        """
        assert L != []  # L is assumed sorted
        self.N = 1
        while self.N < len(L):
            self.N *= 2
        self.c = [0] * (2 * self.N)  # --- covered
        self.s = [0] * (2 * self.N)  # --- score
        self.w = [0] * (2 * self.N)  # --- length
        for i, _ in enumerate(L):
            self.w[self.N + i] = L[i]
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]

    def cover(self):
        """:returns: the size of the union of the stored intervals
        """
        return self.s[1]

    def change(self, i, k, offset):
        """when offset = +1, adds an interval [i, k],
        when offset = -1, removes it
        :complexity: O(log L)
        """
        self._change(1, 0, self.N, i, k, offset)

    def _change(self, p, start, span, i, k, offset):
        if start + span <= i or k <= start:  # --- disjoint
            return
        if i <= start and start + span <= k:  # --- included
            self.c[p] += offset
        else:
            self._change(2 * p, start, span // 2, i, k, offset)
            self._change(2 * p + 1, start + span // 2, span // 2,
                         i, k, offset)
        if self.c[p] == 0:
            if p >= self.N:  # --- leaf
                self.s[p] = 0
            else:
                self.s[p] = self.s[2 * p] + self.s[2 * p + 1]
        else:
            self.s[p] = self.w[p]


# borrowed from tryalgo library
# https://jilljenn.github.io/tryalgo/_modules/tryalgo/union_rectangles.html#union_rectangles
def union_rectangles_fastest(R):
    """Area of union of rectangles

    :param R: list of rectangles defined by (x1, y1, x2, y2)
       where (x1, y1) is top left corner and (x2, y2) bottom right corner
    :returns: area
    :complexity: :math:`O(n \\log n)`
    """
    OPENING = +1  # constants for events
    CLOSING = -1  # -1 has higher priority

    if R == []:  # segment tree would fail on an empty list
        return 0
    X = set()  # set of all x coordinates in the input
    events = []  # events for the sweep line
    for Rj in R:
        (x1, y1, x2, y2) = Rj
        assert x1 <= x2 and y1 <= y2
        X.add(x1)
        X.add(x2)
        events.append((y1, OPENING, x1, x2))
        events.append((y2, CLOSING, x1, x2))
    i_to_x = list(sorted(X))
    # inverse dictionary
    x_to_i = {i_to_x[i]: i for i in range(len(i_to_x))}
    L = [i_to_x[i + 1] - i_to_x[i] for i in range(len(i_to_x) - 1)]
    C = CoverQuery(L)
    area = 0
    previous_y = 0  # arbitrary initial value,
    #                 because C.cover() is 0 at first iteration
    for y, offset, x1, x2 in sorted(events):
        area += (y - previous_y) * C.cover()
        i1 = x_to_i[x1]
        i2 = x_to_i[x2]
        C.change(i1, i2, offset)
        previous_y = y
    return area


def grow_object(heights, objects, i, j, rows, cols, id):
    objects[i, j] = id
    for k in range(-1, 2):
        for m in range(-1, 2):
            ik = i + k
            jm = j + m
            if 0 <= ik < rows and 0 <= jm < cols:
                if heights[ik, jm] > 0 and objects[ik, jm] == 0:
                    grow_object(heights, objects, ik, jm, rows, cols, id)
    return


def mark_objects(heights):

    shape = heights.shape
    rows = shape[0]
    cols = shape[1]

    objects = np.zeros(shape, dtype=int)

    id = 0

    for i in range(rows):
        for j in range(cols):
            if heights[i, j] > 0 and objects[i, j] == 0:
                id += 1
                grow_object(heights, objects, i, j, rows, cols, id)

    return objects


def exp(heights, azimuth, cellsize=1.0):

    rad_dir = math.pi * (azimuth / 180.0 - 0.5)

    rcos = math.cos(rad_dir)
    rsin = math.sin(rad_dir)
    dir = np.array([rcos, rsin])

    grad_heights = np.nan_to_num(heights)

    fx = np.gradient(grad_heights, 0.5, axis=1)
    fy = np.gradient(grad_heights, 0.5, axis=0)

    norm = np.stack([fx, fy], axis=2)

    prj = np.dot(norm, dir)

    exp = cellsize * prj * (heights > 0)

    return fx * (heights > 0)


def frontal_index_surface(heights, azimuth, cellsize=1.0):
    """Frontal area index for all surfaces

    :param heights: 2D numpy array with building elevations and np.isnan where masked
    :param azimuth: geographic azimuth of the wind direction
    :param cellsize: cell size of heights parameter
    :returns: frontal area index
    """
    rad_dir = math.pi * (azimuth / 180.0 - 0.5)

    rcos = math.cos(rad_dir)
    rsin = math.sin(rad_dir)
    dir  = np.array([rcos, rsin])


    grad_heights = np.nan_to_num(heights)

    fx = np.gradient(grad_heights, 0.5, axis=1)
    fy = np.gradient(grad_heights, 0.5, axis=0)

    norm = np.stack([fx, fy], axis=2)
    prj = np.dot(norm, dir)
    exp = cellsize * prj * (heights > 0)

    volume = (cellsize ** 2) * np.count_nonzero(np.logical_not(np.isnan(heights))) * np.mean(heights[heights > 0])
    FAI = np.sum(exp[exp > 0]) / volume

    return FAI

def frontal_index(heights, azimuth, cellsize=1.0):
    """Frontal area index for individual buildings

    :param heights: 2D numpy array with building elevations and np.isnan where masked
    :param azimuth: geographic azimuth of the wind direction
    :param cellsize: cell size of heights parameter
    :returns: frontal area index
    """
    rad_dir = -math.pi * azimuth / 180.0
    rcos = math.cos(rad_dir)
    rsin = math.sin(rad_dir)

    rotate = lambda pnt: rsin * pnt[0] - rcos * pnt[1]

    ids = mark_objects(heights)
    n = ids.max()

    FAI = 0

    for id in range(1, n+1):
        pts = np.argwhere((heights > 0) & (ids == id))
        x = np.array(list(map(rotate, pts)))
        z = np.array(heights[pts[:, 0], pts[:, 1]])

        delta = 0.5 * math.sqrt(2) * math.fabs(math.cos(0.25 * math.pi - rad_dir % (0.5 * math.pi)))

        x1 = x - delta
        x2 = x + delta

        rects = []
        for i in range(len(x1)):
            rects.append((x1[i], 0, x2[i], z[i]))

        front = cellsize * union_rectangles_fastest(rects)

        if azimuth == 90:
            print(front)

        FAI += front

    volume = (cellsize ** 2) * np.count_nonzero(np.logical_not(np.isnan(heights))) * np.mean(heights[heights > 0])

    FAI /= volume

    return FAI

def frontal_index_blocking(heights, azimuth, cellsize=1.0):
    """Frontal area index with buildings blocking

    :param heights: 2D numpy array with building elevations and np.isnan where masked
    :param azimuth: geographic azimuth of the wind direction
    :param cellsize: cell size of heights parameter
    :returns: frontal area index
    """
    rad_dir = -math.pi * azimuth / 180.0
    rcos = math.cos(rad_dir)
    rsin = math.sin(rad_dir)

    rotate = lambda pnt: rsin * pnt[0] - rcos * pnt[1]

    pts = np.argwhere(heights > 0)
    x = np.array(list(map(rotate, pts)))
    z = np.array(heights[pts[:, 0], pts[:, 1]])

    # print(x)

    delta = 0.5 * math.sqrt(2) * math.fabs(math.cos(0.25 * math.pi - rad_dir % (0.5 * math.pi)))
    x1 = x - delta
    x2 = x + delta

    rects = []
    for i in range(len(x1)):
        rects.append((x1[i], 0, x2[i], z[i]))

    front = cellsize * union_rectangles_fastest(rects)
    volume = (cellsize ** 2) * np.count_nonzero(np.logical_not(np.isnan(heights))) * np.mean(z)

    return front / volume
