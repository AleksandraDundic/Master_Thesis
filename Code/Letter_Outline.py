import cv2
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFilter


# ## 1) Outline Curves Representation



def curve_representation(image):
    """
    Extracts curve representations (contours) from an image and identifies contour types.

    Parameters:
    - image (PIL.Image): Input binary image with white background and black shapes.

    Returns:
    - tuple: 
        - curve_list (list of lists): Each sublist contains points of a contour (y, x format).
        - contour_type_list (list): List indicating contour type (0 = outer, 1 = inner).
    """
    image_array = 1 - np.array(image)//255
    
    contours, hierarchy = cv2.findContours(image_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Rewrite curves as a list of lists
    curve_list = []
    contour_type_list = []
    
    for i, contour in enumerate(contours):
        contour_list = []
        for c in contour:
            x, y = c[0]
            contour_list.append(tuple((y,x)))
        
        if contour_list and contour_list[0] != contour_list[-1]:
            contour_list.append(contour_list[0])
            
        if hierarchy[0][i][3] == -1:
            contour_type_list.append(0)  # Outer contour
        else:
            contour_type_list.append(1)  # Inner contour

        curve_list.append(contour_list)

    return curve_list, contour_type_list



def draw_outline(boundary, size):
    """
    Creates a binary image of the contour boundary.

    Parameters:
    - boundary (list of lists): List of contours, each represented as a list of (y, x) points.
    - size (int): Size of the output image (assumes square dimensions).

    Returns:
    - PIL.Image: Image with contours drawn (black shapes on white background).
    """
    new_image_array = np.zeros((size, size), dtype=np.uint8)
    new_image_array += 255  # Set all pixels to white

    # Loop through each boundary and set the corresponding points to black (0)
    for b in boundary:
        for point in b:
            # Unwrap the nested array structure and get the x, y coordinates
            new_image_array[point[0], point[1]] = 0  # Set the point to black
    
    # Convert the array into a PIL image and return it
    return Image.fromarray(new_image_array)


# ## 2) Bézier Representation of the Outline into Curve Represenntation



def bezier_point(t, p0, p1, p2, p3):
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

# Function to discretize a Bézier curve segment
def discretize_bezier_segment(control_points, num_points=100):
    p0, p1, p2, p3 = control_points
    points = [tuple(np.round(bezier_point(t, p0, p1, p2, p3)).astype(int))
              for t in np.linspace(0, 1, num_points)]
    return pd.Series(points).unique().tolist()



def bezier_to_outline(bez_repr, num_points=1000):
    outline = []
    
    for contour in bez_repr:
        _, curve_segments = contour
        contour_points = []
        
        for segment in curve_segments:
            # Discretize each segment
            segment_points = discretize_bezier_segment(segment, num_points)
            
            # Add points to the contour, avoid duplicate start points
            if contour_points and segment_points[0] == contour_points[-1]:
                contour_points.extend(segment_points[1:])  # Avoid duplicating the start point
            else:
                contour_points.extend(segment_points)
        
        # Ensure the contour is closed
        if contour_points and contour_points[0] != contour_points[-1]:
            contour_points.append(contour_points[0])
        
        # Add this contour to the outline
        outline.append(contour_points)
    
    return outline


# ## 3) Fill Letter Determined by its Bézier Representation



def fill_letter(bez_repr, contours_type, size=128, num_points = 500):
    contours = bezier_to_outline(bez_repr, num_points)
    
    filled_image_array = np.ones((size, size), dtype=np.uint8) * 255
    
    contours_np = [np.array([(x, y) for y, x in contour], dtype=np.int32) for contour in contours]

    for i, contour in enumerate(contours_np):
        if contours_type[i] == 0:
            cv2.fillPoly(filled_image_array, [contour], color=0)
        else:
            cv2.fillPoly(filled_image_array, [contour], color=255)
    
    for i, contour in enumerate(contours_np):
        if contours_type[i] == 0:  # Only apply to outer contours
            cv2.polylines(filled_image_array, [contour], isClosed=True, color=0, thickness=1)

    
    return Image.fromarray(filled_image_array)

