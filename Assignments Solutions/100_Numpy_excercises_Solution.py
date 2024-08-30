"""Problem 1: Import the numpy package under the name np"""

import numpy as np

##############################################################################################
"""Problem 2: Print the numpy version and the configuration"""

print(np.__version__)
print(np.show_config())
print("**********************************************************")

##############################################################################################
"""Problem 3: Create a null vector of size 10"""

null_vector = np.zeros(10, dtype=int)
print(null_vector)

# Alternate solution
arr = np.array([0] * 10)
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 4: How to find the memory size of any array"""

print(null_vector.size)
print(arr.size)
print("**********************************************************")

##############################################################################################
"""Problem 5: How to get the documentation of the numpy add function from the command line?"""

print(np.info(np.add))
print("**********************************************************")

##############################################################################################
"""Problem 6: Create a null vector of size 10 but the fifth value which is 1"""

null_vector = np.zeros(10, dtype=int)
null_vector[4] = 1
print(null_vector)

# Alternate solution
arr = np.array([0] * 10)
arr[4] = 1
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 7: Create a vector with values ranging from 10 to 49"""

arr = np.arange(10, 50)
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 8: Reverse a vector (first element becomes last)"""

arr = np.arange(10, 20)
print(arr[::-1])
print("**********************************************************")

##############################################################################################
"""Problem 9: Create a 3x3 matrix with values ranging from 0 to 8"""

arr = np.arange(9).reshape(3, 3)
print(arr)

# Alternate solution
arr1 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(arr1)
print("**********************************************************")

##############################################################################################
"""Problem 10: Find indices of non-zero elements from [1,2,0,0,4,0]"""

arr = np.array([1, 2, 0, 0, 4, 0])
print(np.nonzero(arr))

# Alternate solution
print(np.where(arr != 0))
print("**********************************************************")

##############################################################################################
"""Problem 11: Create a 3x3 identity matrix"""

arr = np.eye(3, dtype=int)
print(arr)

# Alternate solution
arr1 = np.identity(3, dtype=int)
print(arr1)
print("**********************************************************")

##############################################################################################
"""Problem 12: Create a 3x3x3 array with random values"""

arr = np.random.random((3, 3, 3))
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 13: Create a 10x10 array with random values and find the minimum and maximum values"""

arr = np.random.random((10, 10))
print(arr)
print(arr.min())
print(arr.max())
print("**********************************************************")

##############################################################################################
"""Problem 14: Create a random vector of size 30 and find the mean value"""

arr = np.random.random(30)
print(arr)
print(arr.mean())
print("**********************************************************")

##############################################################################################
"""Problem 15: Create a 2d array with 1 on the border and 0 inside"""

arr = np.ones((5, 5), dtype=int)
arr[1:-1, 1:-1] = 0
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 16: How to add a border (filled with 0's) around an existing array?"""

arr = np.ones((5, 5), dtype=int)
arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 17: What is the result of the following expression?
    0 * np.nan
    np.nan == np.nan
    np.inf > np.nan
    np.nan - np.nan
    np.nan in set([np.nan])
    0.3 == 3 * 0.1 """

print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)
print("**********************************************************")

##############################################################################################
"""Problem 18: Create a 5x5 matrix with values 1,2,3,4 just below the diagonal"""

arr = np.diag(np.arange(1, 5), k=-1)
print(arr)

# Alternate solution
arr1 = np.diag([1, 2, 3, 4], k=-1)
print(arr1)
print("**********************************************************")

##############################################################################################
"""Problem 19: Create a 8x8 matrix and fill it with a checkerboard pattern"""

arr = np.zeros((8, 8), dtype=int)
arr[1::2, ::2] = 1
arr[::2, 1::2] = 1
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 20: Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?"""

print(np.unravel_index(100, (6, 7, 8)))
print("**********************************************************")

##############################################################################################
"""Problem 21: Create a checkerboard 8x8 matrix using the tile function"""

arr = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 22: Normalize a 5x5 random matrix"""

arr = np.random.random((5, 5))
arr = (arr - arr.min()) / (arr.max() - arr.min())
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 23: Create a custom dtype that describes a color as four unsigned bytes (RGBA)"""

color_dtype = np.dtype([("R", np.ubyte, 1),
                        ("G", np.ubyte, 1),
                        ("B", np.ubyte, 1),
                        ("A", np.ubyte, 1)])
print(color_dtype)
print("**********************************************************")

##############################################################################################
"""Problem 24: Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)"""

arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
print(arr1)
arr2 = np.array([[16, 17], [18, 19], [20, 21]])
print(arr2)
print(np.dot(arr1, arr2))
print("**********************************************************")

##############################################################################################
"""Problem 25: Given a 1D array, negate all elements which are between 3 and 8, in place"""

arr = np.arange(11)
print(arr)
arr[(3 <= arr) & (arr <= 8)] *= -1
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 26: What is the output of the following script?
    # Author: Jake VanderPlas

    print(sum(range(5),-1))
    from numpy import *
    print(sum(range(5),-1)) """

print(sum(range(5), -1))
from numpy import *

print(sum(range(5), -1))
print("**********************************************************")

##############################################################################################
"""Problem 27: Consider an integer vector Z, which of these expressions are legal?
    Z**Z
    2 << Z >> 2
    Z <- Z
    1j*Z
    Z/1/1
    Z<Z>Z """

Z = np.arange(5)
print(Z ** Z)
print(2 << Z)
print(2 << Z >> 2)
print(Z < - Z)
print(1j * Z)
print(Z / 1 / 1)

# Z<Z>Z is illegal
print("**********************************************************")

##############################################################################################
"""Problem 28: What are the result of the following expressions?
    np.array(0) / np.array(0)
    np.array(0) // np.array(0)
    np.array([np.nan]).astype(int).astype(float) """

print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))
print("**********************************************************")

##############################################################################################
"""Problem 29: How to round away from zero a float array?"""

arr = np.random.uniform(-10, 10, 5)
print(arr)
print(np.copysign(np.ceil(np.abs(arr)), arr).astype(int))
print("**********************************************************")

##############################################################################################
"""Problem 30: How to find common values between two arrays?"""

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([3, 4, 5, 6, 7])
print(np.intersect1d(arr1, arr2))
print("**********************************************************")

##############################################################################################
"""Problem 31: How to ignore all numpy warnings (not recommended)?"""

import warnings

warnings.filterwarnings('ignore')
arr = np.ones(1) / 0
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 32: Is the following expressions true?
    np.sqrt(-1) == np.emath.sqrt(-1)"""

print(np.sqrt(-1) == np.emath.sqrt(-1))
print("**********************************************************")

##############################################################################################
"""Problem 33: How to get the dates of yesterday, today and tomorrow?"""

yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print(yesterday, today, tomorrow)
print("**********************************************************")

##############################################################################################
"""Problem 34: How to get all the dates corresponding to the month of July 2016?"""

arr = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(arr)

# Alternate solution
arr1 = np.arange(np.datetime64('2016-07-01'), np.datetime64('2016-08-01'))
print(arr1)
print("**********************************************************")

##############################################################################################
"""Problem 35: How to compute ((A+B)*(-A/2)) in place (without copy)?"""

A = np.ones(3) * 1
B = np.ones(3) * 2
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)
print(A)
print("**********************************************************")

##############################################################################################
"""Problem 36: Extract the integer part of a random array of positive numbers using 4 different methods"""

arr = np.random.uniform(0, 10, 5)
print(arr)
print(arr - arr % 1)
print(np.floor(arr))
print(np.ceil(arr) - 1)
print(arr.astype(int))
print("**********************************************************")

##############################################################################################
"""Problem 37: Create a 5x5 matrix with row values ranging from 0 to 4"""

arr = np.zeros((5, 5))
arr += np.arange(5)
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 38: Consider a generator function that generates 10 integers and use it to build an array"""


def generate():
    for x in range(10):
        yield x  # yield is used to return a generator (returns a sequence of values instead of a single value)


arr = np.fromiter(generate(), dtype=int, count=-1)
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 39: Create a vector of size 10 with values ranging from 0 to 1, both excluded"""

arr = np.linspace(0, 1, 12, endpoint=True)[1:-1]
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 40: Create a random vector of size 10 and sort it"""

arr = np.random.random(10)
arr.sort()
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 41: How to sum a small array faster than np.sum?"""

arr = np.arange(10)
print(np.add.reduce(arr))
print("**********************************************************")

##############################################################################################
"""Problem 42: Consider two random array A and B, check if they are equal"""

A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
print(np.array_equal(A, B))
print("**********************************************************")

##############################################################################################
"""Problem 43: Make an array immutable (read-only)"""

arr = np.ones(10)
arr.flags.writeable = False
print(arr)
# arr[0] = 2
# print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 44: Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates"""

arr = np.random.random((10, 2))
X, Y = arr[:, 0], arr[:, 1]
R = np.sqrt(X ** 2 + Y ** 2)
T = np.arctan2(Y, X)
print(R)
print(T)
print("**********************************************************")

##############################################################################################
"""Problem 45: Create random vector of size 10 and replace the maximum value by 0"""

arr = np.random.random(10)
print(arr)
arr[arr.argmax()] = 0
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 46: Create a structured array with x and y coordinates covering the [0,1]x[0,1] area"""
# ask
arr = np.zeros((5, 5), [('x', float), ('y', float)])
arr['x'], arr['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 47: Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))"""
# ask
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(C)
print("**********************************************************")

##############################################################################################
"""Problem 48: Print the minimum and maximum representable value for each numpy scalar type"""

for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)

for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)
print("**********************************************************")

##############################################################################################
"""Problem 49: How to print all the values of an array?"""

arr = np.array([1, 2, 3, 4, 5])
for value in arr:
    print(value)
print("**********************************************************")

##############################################################################################
"""Problem 50: How to find the closest value (to a given scalar) in a vector?"""

arr = np.arange(100)
value = np.random.uniform(0, 100)
index = (np.abs(arr - value)).argmin()
print(arr[index])
print("**********************************************************")

##############################################################################################
"""Problem 51: Create a structured array representing a position (x,y) and a color (r,g,b)"""

dtype = np.dtype([('position', [('x', np.float32), ('y', np.float32)]),
                  ('color', [('r', np.uint8), ('g', np.uint8), ('b', np.uint8)])])

data = np.array([((1.0, 2.0), (255, 0, 0)),  # Red at position (1.0, 2.0)
                 ((3.0, 4.0), (0, 255, 0)),  # Green at position (3.0, 4.0)
                 ((5.0, 6.0), (0, 0, 255))],  # Blue at position (5.0, 6.0)
                dtype=dtype)
print(data)
print("**********************************************************")

##############################################################################################
"""Problem 52: Consider a random vector with shape (100,2) representing coordinates, find point by point distances"""

arr = np.random.random((100, 2))
X, Y = np.atleast_2d(arr[:, 0]), np.atleast_2d(arr[:, 1])
D = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)
print(D)
print("**********************************************************")

##############################################################################################
"""Problem 53: How to convert a float (32 bits) array into an integer (32 bits) in place?"""

arr = np.array([1.5, 2.7, 3.9], dtype=np.float32)
arr.astype(np.int32, copy=False)
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 54: How to read the following file?
    1, 2, 3, 4, 5
    6,  ,  , 7, 8
    ,  , 9,10,11 """

from io import StringIO

data = """1, 2, 3, 4, 5
        6,  ,  , 7, 8
         ,  , 9,10,11"""
arr = np.genfromtxt(StringIO(data), delimiter=",", filling_values=0)
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 55: What is the equivalent of enumerate for numpy arrays?"""

arr = np.array([[1, 2], [3, 4]])
for index, value in np.ndenumerate(arr):
    print(index, value)
print("**********************************************************")

##############################################################################################
"""Problem 56: Generate a generic 2D Gaussian-like array"""

def generate_gaussian(size, sigma=1.0):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x*x + y*y)
    gaussian = np.exp(-(d**2 / (2.0 * sigma**2)))
    return gaussian

gaussian_array = generate_gaussian(5, sigma=1.0)
print(gaussian_array)
print("**********************************************************")

##############################################################################################
"""Problem 57: How to randomly place p elements in a 2D array?"""

arr = np.zeros((5, 5))
p = 5
indices = np.random.choice(arr.size, p, replace=False)
np.put(arr, indices, 1)
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 58: Subtract the mean of each row of a matrix"""

arr = np.random.rand(5, 5)
arr = arr - arr.mean(axis=1, keepdims=True)
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 59: How to sort an array by the nth column?"""

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
n = 1
sorted_arr = arr[arr[:, n].argsort()]
print(sorted_arr)
print("**********************************************************")

##############################################################################################
"""Problem 60: How to tell if a given 2D array has null columns?"""

arr = np.array([[1, 0, 3], [0, 0, 6], [0, 0, 9]])
null_columns = np.all(arr == 0, axis=0)
print(null_columns)
print("**********************************************************")

##############################################################################################
"""Problem 61: Find the nearest value from a given value in an array"""

arr = np.array([1, 3, 5, 7, 9])
value = 6
nearest_value = arr[np.abs(arr - value).argmin()]
print(nearest_value)
print("**********************************************************")

##############################################################################################
"""Problem 62: Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator?"""

a = np.array([[1, 2, 3]])
b = np.array([[4], [5], [6]])

with np.nditer([a, b, None]) as it:
    for x, y, z in it:
        z[...] = x + y
        print(z)
print("**********************************************************")

##############################################################################################
"""Problem 63: Create an array class that has a name attribute"""

class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'name', "no name")

arr = NamedArray(np.arange(10), "range_10")
print(arr)
print(arr.name)
print("**********************************************************")

##############################################################################################
"""Problem 64: Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)?"""

arr = np.zeros(10, dtype=int)
indices = np.array([1, 3, 5, 3, 1, 7])
np.add.at(arr, indices, 1)
print(arr)
print("**********************************************************")

##############################################################################################
"""Problem 65: How to accumulate elements of a vector (X) to an array (F) based on an index list (I)?"""

X = np.array([1, 2, 3, 4, 5, 6])
I = np.array([1, 3, 9, 3, 4, 1])
F = np.bincount(I, weights=X)
print(F)
print("**********************************************************")

##############################################################################################
"""Problem 66: Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors"""

image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
colors = image.reshape(-1, 3)
unique_colors = np.unique(colors, axis=0)
num_unique_colors = len(unique_colors)
print(num_unique_colors)
print("**********************************************************")

##############################################################################################
"""Problem 67: Considering a four dimensions array, how to get sum over the last two axis at once?"""

arr = np.random.randint(0, 10, (3, 4, 5, 6))
sum_last_two_axis = arr.sum(axis=(-2, -1))
print(sum_last_two_axis)
print("**********************************************************")

##############################################################################################
"""Problem 68: Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices?"""

D = np.random.rand(10)
S = np.array([0, 1, 0, 1, 2, 2, 0, 1, 2, 0])
sum_per_subset = np.bincount(S, weights=D)
count_per_subset = np.bincount(S)
mean_per_subset = sum_per_subset / count_per_subset
print(mean_per_subset)
print("**********************************************************")

##############################################################################################
"""Problem 69: How to get the diagonal of a dot product?"""

A = np.random.randint(0, 10, (3, 3))
B = np.random.randint(0, 10, (3, 3))
dot_product = np.dot(A, B)
diagonal = np.diag(dot_product)
print(diagonal)
print("**********************************************************")

##############################################################################################
"""Problem 70: Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value?"""

arr = np.array([1, 2, 3, 4, 5])
zeros = 3
interleaved_zeros = np.zeros(len(arr) + (len(arr) - 1) * zeros, dtype=int)
interleaved_zeros[::zeros + 1] = arr
print(interleaved_zeros)
print("**********************************************************")

##############################################################################################
"""Problem 71: Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)?"""

A = np.random.randint(0, 10, (5, 5, 3))
B = np.random.randint(0, 10, (5, 5))
C = A * B[:, :, None]
print(C)
print("**********************************************************")

##############################################################################################
"""Problem 72: How to swap two rows of an array?"""

arr = np.arange(25).reshape(5, 5)
print(arr)
arr[[0, 1]] = arr[[1, 0]]
swapped_arr = arr
print(swapped_arr)
print("**********************************************************")

##############################################################################################
"""Problem 73: Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles"""
## ask
faces = np.random.randint(0, 100, (10, 3))
segments = np.vstack([np.sort(faces[:, [i, j]], axis=1) for i, j in [(0, 1), (1, 2), (2, 0)]])
unique_segments = np.unique(segments, axis=0)

print(unique_segments)
print("**********************************************************")

##############################################################################################
"""Problem 74: Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C?"""

C = np.bincount([1, 1, 2, 3, 4, 4, 6])
A = np.repeat(np.arange(len(C)), C)
print(A)
print("**********************************************************")

##############################################################################################
"""Problem 75: How to compute averages using a sliding window over an array?"""

arr = np.arange(10)
window_size = 4
averages = np.convolve(arr, np.ones(window_size), mode='valid') / window_size
print(averages)
print("**********************************************************")

##############################################################################################
"""Problem 76: Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1])"""

arr = np.arange(1, 15)
R = np.array([Z[i:i+4] for i in range(len(Z)-3)])
print(R)
print("**********************************************************")

##############################################################################################
"""Problem 77: How to negate a boolean, or to change the sign of a float inplace?"""

# Boolean negation
Z_bool = np.array([True, False, True], dtype=bool)
np.logical_not(Z_bool, out=Z_bool)
print(Z_bool)

# Float sign change
Z_float = np.array([1.0, -2.0, 3.0])
np.negative(Z_float, out=Z_float)
print(Z_float)
print("**********************************************************")

##############################################################################################
"""Problem 78: Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])?"""

def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:, 0] - p[..., 0]) * T[:, 0] + (P0[:, 1] - p[..., 1]) * T[:, 1]) / L
    U = U.reshape(len(U), 1)
    D = P0 + U * T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(0, 10, (10, 2))
P1 = np.random.uniform(0, 10, (10, 2))
p = np.random.uniform(0, 10, (1, 2))
distances = distance(P0, P1, p)
print(distances)
print("**********************************************************")

##############################################################################################
"""Problem 79: Consider 3 points (p0,p1,p2) and a line L, how to compute distance from each point to line?"""

def distance(p0, p1, p2, L):
    return np.abs((L[1] - L[0]) * (p0 - p2) - (L[1] - L[0]) * (p1 - p0)) / np.sqrt((L[1] - L[0]) ** 2).sum()

p0 = np.random.uniform(0, 10, (10, 2))
p1 = np.random.uniform(0, 10, (10, 2))
p2 = np.random.uniform(0, 10, (10, 2))
L = np.random.uniform(0, 10, (2, 2))
distances = distance(p0, p1, p2, L)
print(distances)
print("**********************************************************")

##############################################################################################
"""Problem 80: Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary)"""

def extract_subarray(arr, center, shape, fill_value=0):
    padded_arr = np.pad(arr, [(s//2, s//2) for s in shape], constant_values=fill_value)
    slices = [slice(c + s//2, c + s//2 + s) for c, s in zip(center, shape)]
    return padded_arr[tuple(slices)]

arr = np.random.rand(5, 5)
center = (2, 2)
shape = (3, 3)
subpart = extract_subarray(arr, center, shape)
print(subpart)
print("**********************************************************")

##############################################################################################
"""Problem 81: Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]?"""

Z = np.arange(1, 15, dtype=int)
R = np.lib.stride_tricks.as_strided(Z, shape=(11, 4), strides=(Z.itemsize, Z.itemsize))
print(R)
print("**********************************************************")

##############################################################################################
"""Problem 82: Compute a matrix rank"""

arr = np.random.uniform(0, 1, (10, 5))
rank = np.linalg.matrix_rank(arr)
print(rank)
print("**********************************************************")

##############################################################################################
"""Problem 83: How to find the most frequent value in an array?"""

arr = np.random.randint(0, 10, 50)
most_frequent_value = np.bincount(arr).argmax()
print(most_frequent_value)
print("**********************************************************")

##############################################################################################
"""Problem 84: Extract all the contiguous 3x3 blocks from a random 10x10 matrix"""

arr = np.random.randint(0, 5, (10, 10))
shape = (3, 3)
strides = arr.strides * 2
blocks = np.lib.stride_tricks.as_strided(arr, shape=(8, 8, 3, 3), strides=strides)
print(blocks)
print("**********************************************************")

##############################################################################################
"""Problem 85: Create a 2D array subclass such that Z[i, j] == Z[j, i]"""

class SymmetricArray(np.ndarray):
    def __setitem__(self, index, value):
        i, j = index
        super(SymmetricArray, self).__setitem__((i, j), value)
        super(SymmetricArray, self).__setitem__((j, i), value)

Z = np.random.random((5, 5)).view(SymmetricArray)
Z[1, 2] = 5
print(Z)
print("**********************************************************")

##############################################################################################
"""Problem 86: Consider a set of p matrices with shape (n, n) and a set of p vectors with shape (n, 1). How to compute the sum of the p matrix products at once? (result has shape (n, 1))"""

p = 10
n = 5
matrices = np.random.rand(p, n, n)
vectors = np.random.rand(p, n, 1)
sum_matrix_products = np.tensordot(matrices, vectors, axes=[[0, 2], [0, 1]])
print(sum_matrix_products)
print("**********************************************************")

##############################################################################################
"""Problem 87: Consider a 16x16 array, how to get the block-sum (block size is 4x4)?"""

arr = np.random.rand(16, 16)
block_size = 4
block_sum = np.add.reduceat(np.add.reduceat(arr, np.arange(0, arr.shape[0], block_size), axis=0),
                            np.arange(0, arr.shape[1], block_size), axis=1)
print(block_sum)
print("**********************************************************")

##############################################################################################
"""Problem 88: How to implement the Game of Life using numpy arrays?"""

from scipy.signal import convolve2d

grid = np.random.randint(2, size=(10, 10))
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

def game_of_life_step(grid):
    neighbor_count = convolve2d(grid, kernel, mode='same', boundary='wrap')
    return (neighbor_count == 3) | ((grid == 1) & (neighbor_count == 2))

grid = game_of_life_step(grid)
print(grid)
print("**********************************************************")

##############################################################################################
"""Problem 89: How to get the n largest values of an array?"""

arr = np.arange(100)
n = 5
largest_values = arr[np.argsort(arr)[-n:]]
print(largest_values)
print("**********************************************************")

##############################################################################################
"""Problem 90: Given an arbitrary number of vectors, build the cartesian product (every combinations of every item)"""

import itertools

vectors = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
cartesian_product = np.array(list(itertools.product(*vectors)))
print(cartesian_product)
print("**********************************************************")

##############################################################################################
"""Problem 91: How to create a record array from a regular array?"""

Z = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
record_array = np.core.records.fromarrays(Z.T, names='x, y', formats='f8, f8')
print(record_array)
print("**********************************************************")

##############################################################################################
"""Problem 92: Consider a large vector Z, compute Z to the power of 3 using 3 different methods"""

Z = np.random.rand(1000000)

# Method 1: Using **
result_1 = Z ** 3

# Method 2: Using np.power
result_2 = np.power(Z, 3)

# Method 3: Using a loop (less efficient but possible)
result_3 = np.array([z ** 3 for z in Z])

print(result_1, result_2, result_3)
print("**********************************************************")

##############################################################################################
"""Problem 93: Consider two arrays A and B of shape (8, 3) and (2, 2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B?"""

A = np.random.randint(0, 5, (8, 3))
B = np.random.randint(0, 5, (2, 2))
C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.all((3, 1)).any(1))[0]
print(rows)
print("**********************************************************")

##############################################################################################
"""Problem 94: Considering a 10x3 matrix, extract rows with unequal values (e.g. [2, 2, 3])"""

arr = np.random.randint(0, 5, (10, 3))
unequal_rows = arr[np.any(arr[:, 1:] != arr[:, :-1], axis=1)]
print(unequal_rows)
print("**********************************************************")

##############################################################################################
"""Problem 95: Convert a vector of ints into a matrix binary representation"""

arr = np.array([0, 1, 2, 3, 15, 16, 32], dtype=np.uint8)
binary_matrix = ((arr[:, None] & (1 << np.arange(8))) > 0).astype(int)
print(binary_matrix)
print("**********************************************************")

##############################################################################################
"""Problem 96: Given a two-dimensional array, how to extract unique rows?"""

arr = np.random.randint(0, 2, (6, 3))
unique_rows = np.unique(arr, axis=0)
print(unique_rows)
print("**********************************************************")

##############################################################################################
"""Problem 97: Considering 2 vectors A and B, write the einsum equivalent of inner, outer, sum and mul function"""

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# Inner product
inner_product = np.einsum('i,i', A, B)

# Outer product
outer_product = np.einsum('i,j->ij', A, B)

# Sum
sum_result = np.einsum('i->', A)

# Element-wise multiplication
elementwise_mul = np.einsum('i,i->i', A, B)

print(f"Inner Product: {inner_product}")
print(f"Outer Product:\n{outer_product}")
print(f"Sum: {sum_result}")
print(f"Element-wise Multiplication: {elementwise_mul}")
print("**********************************************************")

##############################################################################################
"""Problem 98: Considering a path described by two vectors (X, Y), how to sample it using equidistant samples?"""

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# Inner product
inner_product = np.einsum('i,i', A, B)

# Outer product
outer_product = np.einsum('i,j->ij', A, B)

# Sum
sum_result = np.einsum('i->', A)

# Element-wise multiplication
elementwise_mul = np.einsum('i,i->i', A, B)

print(f"Inner Product: {inner_product}")
print(f"Outer Product:\n{outer_product}")
print(f"Sum: {sum_result}")
print(f"Element-wise Multiplication: {elementwise_mul}")
print("**********************************************************")

##############################################################################################
"""Problem 99: Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n"""

X = np.array([[1, 1, 3], [2, 2, 2], [4, 4, 4]])
n = 5
valid_rows = X[np.all(X.astype(int) == X, axis=1) & (np.sum(X, axis=1) == n)]
print(valid_rows)
print("**********************************************************")

##############################################################################################
"""Problem 100: Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means)"""

X = np.random.randn(100)
N = 1000
means = [np.mean(np.random.choice(X, len(X))) for _ in range(N)]
confidence_interval = np.percentile(means, [2.5, 97.5])
print(confidence_interval)

