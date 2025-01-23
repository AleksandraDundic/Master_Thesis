from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from scipy.signal import convolve2d

import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv


# 1) Skeletonization



def extract_skeleton(letter_image):
    """
    Converts a binary image of a letter into its skeleton representation.
    
    Parameters:
    - letter_image (PIL.Image): Binary image of the letter.
    
    Returns:
    - np.array: Skeletonized binary image of the letter.
    """
    image_array = np.array(letter_image)
    image_array //= 255
    inverted_array = 1 - image_array
    letter_skeleton = skeletonize(inverted_array)
    
    letter_skeleton_01 = letter_skeleton.astype(np.uint8)
    
    return letter_skeleton_01


def show_skeleton(skeleton):
    return Image.fromarray((1-skeleton)*255)


# 2) Keypoint Detection



# Detecting endpoints
def detect_endpoints(skeleton):
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    
    convolved = convolve2d(skeleton, kernel, mode = 'same', boundary='fill', fillvalue=0)
    m1 = convolved == 11
    m2 = convolved == 12
    m3 = convolved == 13
    
    endpoints = []
    nonzero_positions_1 = np.nonzero(m1)
    for x, y in zip(nonzero_positions_1[1], nonzero_positions_1[0]):
        endpoints.append(tuple((y, x)))
        
    nonzero_positions_2 = np.nonzero(m2)
    for x, y in zip(nonzero_positions_2[1], nonzero_positions_2[0]):
        if skeleton[y+1,x+1] == 1 and skeleton[y+1,x] == 1:
            endpoints.append(tuple((y, x)))
        elif skeleton[y+1,x-1] == 1 and skeleton[y+1,x] == 1:
            endpoints.append(tuple((y, x)))
        elif skeleton[y,x+1] == 1 and skeleton[y+1,x+1] == 1:
            endpoints.append(tuple((y, x)))
        elif skeleton[y,x+1] == 1 and skeleton[y-1,x+1] == 1:
            endpoints.append(tuple((y, x)))
        elif skeleton[y-1,x] == 1 and skeleton[y-1,x+1] == 1:
            endpoints.append(tuple((y, x)))
        elif skeleton[y-1,x] == 1 and skeleton[y-1,x-1] == 1:
            endpoints.append(tuple((y, x)))
        elif skeleton[y,x-1] == 1 and skeleton[y-1,x-1] == 1:
            endpoints.append(tuple((y, x)))
        elif skeleton[y,x-1] == 1 and skeleton[y+1,x-1] == 1:
            endpoints.append(tuple((y, x)))
            
    
    nonzero_positions_3 = np.nonzero(m3)
    for x, y in zip(nonzero_positions_3[1], nonzero_positions_3[0]):
        if skeleton[y+1,x-1] == 1 and skeleton[y+1,x] and skeleton[y+1,x+1] == 1:
            endpoints.append(tuple((y, x)))
        elif skeleton[y-1,x+1] == 1 and skeleton[y,x+1] and skeleton[y+1,x+1] == 1:
            endpoints.append(tuple((y, x)))
        elif skeleton[y+1,x-1] == 1 and skeleton[y+1,x] and skeleton[y+1,x+1] == 1:
            endpoints.append(tuple((y, x)))
        elif skeleton[y-1,x-1] == 1 and skeleton[y,x-1] and skeleton[y+1,x-1] == 1:
            endpoints.append(tuple((y, x)))
  
    return endpoints


def count_4_neighbors(point, skeleton):
    """
    Counts the number of 4-connected neighbors for a given point in the skeleton.
    """
    y, x = point
    neighbors = [
        (y-1, x),  # Up
        (y+1, x),  # Down
        (y, x-1),  # Left
        (y, x+1)   # Right
    ]
    
    # Count how many of these neighbors are part of the skeleton
    count = 0
    for ny, nx in neighbors:
        if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
            if skeleton[ny, nx] == 1:
                count += 1
    return count


def sort_intersections_by_neighbors(intersections, skeleton):
    """
    Sorts intersections by the number of 4-connected neighbors in descending order.
    """
    sorted_intersections = sorted(intersections, key=lambda point: count_4_neighbors(point, skeleton), reverse=True)
    return sorted_intersections


# Detecting intersections
def detect_intersections(skeleton):
    kernel1 = np.array([[1,0,1],[0,10,0],[0,1,0]])
    kernel2 = np.array([[0,1,0],[0,10,1],[1,0,0]])
    kernel3 = np.array([[0,0,1],[1,10,0],[0,0,1]])
    kernel4 = np.array([[1,0,0],[0,10,1],[0,1,0]])
    kernel5 = np.array([[0,1,0],[0,10,0],[1,0,1]])
    kernel6 = np.array([[0,0,1],[1,10,0],[0,1,0]])
    kernel7 = np.array([[1,0,0],[0,10,1],[1,0,0]])
    kernel8 = np.array([[0,1,0],[1,10,0],[0,0,1]])
    kernel9 = np.array([[1,0,0],[0,10,0],[1,0,1]])
    kernel10 = np.array([[1,0,1],[0,10,0],[1,0,0]])
    kernel11 = np.array([[1,0,1],[0,10,0],[0,0,1]])
    kernel12 = np.array([[0,0,1],[0,10,0],[1,0,1]])
    kernel13 = np.array([[0,1,0],[1,10,1],[0,1,0]])
    kernel14 = np.array([[1,0,1],[0,10,0],[1,0,1]])
    
    intersection_points = set()
    
    kernels = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8, kernel9, kernel10, kernel11, kernel12, kernel13, kernel14]
    comparison_values = [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14]
    adjustments = [
        (lambda y, x: (y+1, x) if skeleton[y+1, x] == 1 else (y, x)),
        (lambda y, x: (y, x)),
        (lambda y, x: (y, x-1) if skeleton[y, x-1] == 1 else (y, x)),
        (lambda y, x: (y, x)),
        (lambda y, x: (y-1, x) if skeleton[y-1, x] == 1 else (y, x)),
        (lambda y, x: (y, x)),
        (lambda y, x: (y, x+1) if skeleton[y, x+1] == 1 else (y, x)),
        (lambda y, x: (y, x)),
        (lambda y, x: (y-1, x) if skeleton[y-1, x] == 1 else (y, x+1) if skeleton[y, x+1] == 1 else (y, x)),
        (lambda y, x: (y+1, x) if skeleton[y+1, x] == 1 else (y, x+1) if skeleton[y, x+1] == 1 else (y, x)),
        (lambda y, x: (y, x-1) if skeleton[y, x-1] == 1 else (y+1, x) if skeleton[y+1, x] == 1 else (y, x)),
        (lambda y, x: (y-1, x) if skeleton[y-1, x] == 1 else (y, x-1) if skeleton[y, x-1] == 1 else (y, x)),
        (lambda y, x: (y, x)),
        (lambda y, x: (y, x))
    ]
    
    for kernel, comp_val, adjust in zip(kernels, comparison_values, adjustments):
        convolved = convolve2d(skeleton, kernel, mode='same', boundary='fill', fillvalue=0)
        mask = convolved == comp_val
        nonzero_positions = np.nonzero(mask)
        
        for x, y in zip(nonzero_positions[1], nonzero_positions[0]):
            intersection_points.add(adjust(y, x))
        
    return sort_intersections_by_neighbors(list(intersection_points), skeleton)



# 3) Curve Representation



def get_neighbors(pixel_coordinates, skeleton):
    """
    Finds all 8-connected neighbors of a given pixel in the skeleton.
    """
    y, x = pixel_coordinates
    skeleton_height, skeleton_length = skeleton.shape

    neighbors = [(y+dy, x+dx) for dx in range(-1, 2) for dy in range(-1, 2)
                 if (dx != 0 or dy != 0) and (0 <= x+dx < skeleton_length) and (0 <= y+dy < skeleton_height) and skeleton[y+dy, x+dx] == 1]
    return neighbors


def all_curve_points(skeleton):
    return {(y, x) for y, x in zip(*np.nonzero(skeleton))}


def get_next(current_point, previous_point, skeleton):
    y, x = current_point
    
    neighbors = get_neighbors(current_point, skeleton)
    if previous_point in neighbors:
        neighbors.remove(previous_point)
    
    neighbors_count = [(n, len(get_neighbors(n, skeleton))) for n in neighbors]
    sorted_neighbors_count = sorted(neighbors_count, key=lambda x: x[1], reverse=False)
    sorted_neighbors = [n[0] for n in sorted_neighbors_count]
    
    non_diagonal_neighbors = [p for p in sorted_neighbors if (p[0] == y or p[1] == x)]
    
    if not sorted_neighbors:
        return None
    
    if non_diagonal_neighbors:
        return non_diagonal_neighbors[0]
    
    return sorted_neighbors[0]


def find_left_top_most_point(skeleton):
    indices = np.argwhere(skeleton == 1)
    
    if indices.size == 0:
        return None
    
    return tuple(indices[0])


def trace_curve(start_point, second_point, skeleton, intersections, endpoints):
    """
    Traces a curve from a starting point along the skeleton until it reaches an endpoint or intersection.
    
    Parameters:
    - start_point (tuple): Starting coordinates of the curve.
    - second_point (tuple): The next point in the curve.
    - skeleton (np.ndarray): Skeletonized image.
    - intersections (list): List of intersection points.
    - endpoints (list): List of endpoints.
    
    Returns:
    - list: List of points forming the traced curve.
    """
    curve = [start_point, second_point]
    skeleton[second_point[0], second_point[1]] = 0
    current_point = second_point
    previous_point = start_point
    
    while True:
        if current_point in endpoints:
            skeleton[current_point[0], current_point[1]] = 0
            endpoints.remove(current_point)
            break
            
        next_point = get_next(current_point, previous_point, skeleton)
        
        if next_point is None:
            if current_point in endpoints:
                endpoints.remove(current_point)
            break
        
        curve.append(next_point)
        
        if next_point in intersections or next_point in endpoints:
            if next_point in endpoints:
                skeleton[next_point[0], next_point[1]] = 0
                endpoints.remove(next_point)
            break
        
        skeleton[next_point[0], next_point[1]] = 0
        previous_point = current_point
        current_point = next_point
    
    return curve


def curve_representation_initial(skeleton):
    intersections = detect_intersections(skeleton)
    endpoints = detect_endpoints(skeleton)
    curves = []
    
    # Process intersections
    if intersections:
        for i in intersections:
            i_neighbors = [n for n in get_neighbors(i, skeleton) if n not in intersections]
            
            for n in i_neighbors:
                if skeleton[n[0], n[1]] == 1:
                    curve = trace_curve(i, n, skeleton, intersections, endpoints)
                    curves.append(curve)
        
        # Remove intersections from the skeleton
        for i in intersections:
            skeleton[i[0], i[1]] = 0
    
    # Process endpoints
    if endpoints:
        for e in endpoints:
            next_point = get_neighbors(e, skeleton)[0]
            curve = trace_curve(e, next_point, skeleton, intersections, endpoints)
            curves.append(curve)
            skeleton[e[0],e[1]] = 0
    
    # Process the remaining curve
    if not intersections and not endpoints:
        starting_point = find_left_top_most_point(skeleton)
        if starting_point != None:
            next_point = get_neighbors(starting_point, skeleton)[0]
            curve = trace_curve(starting_point, next_point, skeleton, intersections, endpoints)
            curves.append(curve)
        
    return curves


def skeleton_from_strokes(original_skeleton, strokes):
    """"
    Creates a skeleton back from the strokes.
    """"
    new_skeleton = np.zeros_like(original_skeleton)
    
    for s in strokes:
        for y,x in s:
            new_skeleton[y,x] = 1
    
    return new_skeleton


def curve_representation(skeleton, stroke_threshold=0.09):
    
    n = len(all_curve_points(skeleton))
    
    current_skeleton = skeleton.copy()
    
    while True:
        intersections = detect_intersections(current_skeleton)
        skeleton_to_compare = current_skeleton.copy()
        current_strokes = curve_representation_initial(current_skeleton)
                
        new_strokes = [s for s in current_strokes if len(s)>=int(stroke_threshold*n) or (s[0] in intersections and s[-1] in intersections)]
        
        new_skeleton = skeleton_from_strokes(skeleton, new_strokes)
        
        if np.array_equal(skeleton_to_compare, new_skeleton):
            return new_strokes
        
        current_skeleton = new_skeleton
        
    return new_strokes



# 4) Visualization



# If we want to show how a part of the skeleton looks like

def create_curve_image(skeleton, curve):
    image = np.zeros_like(skeleton)
    
    for x,y in curve:
        image[x,y] = 1
    
    image_to_show = Image.fromarray(image*255)
    
    return image_to_show