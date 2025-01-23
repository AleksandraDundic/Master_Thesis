from __future__ import print_function
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize
from numpy import *

import matplotlib.pyplot as plt


# 1) Make a Bézier representation



# Evaluates cubic bezier curve (determined by a list of control points cp) at t
def q(cp, t):
    return (1.0-t)**3 * cp[0] + 3*(1.0-t)**2 * t * cp[1] + 3*(1.0-t)* t**2 * cp[2] + t**3 * cp[3]


# Evaluates first derivative at t
def q_first_derivative(cp, t):
    return 3*(1.0-t)**2 * (cp[1]-cp[0]) + 6*(1.0-t) * t * (cp[2]-cp[1]) + 3*t**2 * (cp[3]-cp[2])


# Evaluates second derivative at t
def q_second_derivative(cp, t):
    return 6*(1.0-t) * (cp[2]-2*cp[1]+cp[0]) + 6*(t) * (cp[3]-2*cp[2]+cp[1])



""" Python implementation of
    Algorithm for Automatically Fitting Digitized Curves
    by Philip J. Schneider
    "Graphics Gems", Academic Press, 1990
"""


# Fit one (or more) Bezier curves to a set of points
def fitCurve(points, maxError):
    leftTangent = normalize(points[1] - points[0])
    rightTangent = normalize(points[-2] - points[-1])
    return fitCubic(points, leftTangent, rightTangent, maxError)


def fitCubic(points, leftTangent, rightTangent, error):
    # Use heuristic if region only has two points in it
    if (len(points) == 2):
        dist = linalg.norm(points[0] - points[1]) / 3.0
        bezCurve = [points[0], points[0] + leftTangent * dist, points[1] + rightTangent * dist, points[1]]
        return [bezCurve]

    # Parameterize points, and attempt to fit curve
    u = chordLengthParameterize(points)
    bezCurve = generateBezier(points, u, leftTangent, rightTangent)
    # Find max deviation of points to fitted curve
    maxError, splitPoint = computeMaxError(points, bezCurve, u)
    if maxError < error:
        return [bezCurve]

    # If error not too large, try some reparameterization and iteration
    if maxError < error**2:
        for i in range(20):
            uPrime = reparameterize(bezCurve, points, u)
            bezCurve = generateBezier(points, uPrime, leftTangent, rightTangent)
            maxError, splitPoint = computeMaxError(points, bezCurve, uPrime)
            if maxError < error:
                return [bezCurve]
            u = uPrime

    # Fitting failed -- split at max error point and fit recursively
    beziers = []
    centerTangent = normalize(points[splitPoint-1] - points[splitPoint+1])
    beziers += fitCubic(points[:splitPoint+1], leftTangent, centerTangent, error)
    beziers += fitCubic(points[splitPoint:], -centerTangent, rightTangent, error)

    return beziers


def generateBezier(points, parameters, leftTangent, rightTangent):
    bezCurve = [points[0], None, None, points[-1]]

    # compute the A's
    A = zeros((len(parameters), 2, 2))
    for i, u in enumerate(parameters):
        A[i][0] = leftTangent  * 3*(1-u)**2 * u
        A[i][1] = rightTangent * 3*(1-u)    * u**2

    # Create the C and X matrices
    C = zeros((2, 2))
    X = zeros(2)

    for i, (point, u) in enumerate(zip(points, parameters)):
        C[0][0] += dot(A[i][0], A[i][0])
        C[0][1] += dot(A[i][0], A[i][1])
        C[1][0] += dot(A[i][0], A[i][1])
        C[1][1] += dot(A[i][1], A[i][1])

        tmp = point - q([points[0], points[0], points[-1], points[-1]], u)

        X[0] += dot(A[i][0], tmp)
        X[1] += dot(A[i][1], tmp)

    # Compute the determinants of C and X
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X  = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1  = X[0] * C[1][1] - X[1] * C[0][1]

    # Finally, derive alpha values
    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1

    # If alpha negative, use the Wu/Barsky heuristic (see text) */
    # (if alpha is 0, you get coincident control points that lead to
    # divide by zero in any subsequent NewtonRaphsonRootFind() call. */
    segLength = linalg.norm(points[0] - points[-1])
    epsilon = 1.0e-6 * segLength
    if alpha_l < epsilon or alpha_r < epsilon:
        # fall back on standard (probably inaccurate) formula, and subdivide further if needed.
        bezCurve[1] = around(bezCurve[0] + leftTangent * (segLength / 3.0), 2)
        bezCurve[2] = around(bezCurve[3] + rightTangent * (segLength / 3.0), 2)

    else:
        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned at alpha distance out
        # on the tangent vectors, left and right, respectively
        bezCurve[1] = around(bezCurve[0] + leftTangent * alpha_l, 2)
        bezCurve[2] = around(bezCurve[3] + rightTangent * alpha_r, 2)

    return bezCurve


def reparameterize(bezier, points, parameters):
    return [newtonRaphsonRootFind(bezier, point, u) for point, u in zip(points, parameters)]


def newtonRaphsonRootFind(bez, point, u):
    d = q(bez, u)-point
    numerator = (d * q_first_derivative(bez, u)).sum()
    denominator = (q_first_derivative(bez, u)**2 + d * q_first_derivative(bez, u)).sum()

    if denominator == 0.0:
        return u
    else:
        return u - numerator/denominator


def chordLengthParameterize(points):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i-1] + linalg.norm(points[i] - points[i-1]))

    for i, _ in enumerate(u):
        u[i] = u[i] / u[-1]

    return u


def computeMaxError(points, bez, parameters):
    maxDist = 0.0
    splitPoint = len(points)/2
    for i, (point, u) in enumerate(zip(points, parameters)):
        dist = linalg.norm(q(bez, u)-point)**2
        if dist > maxDist:
            maxDist = dist
            splitPoint = i

    return maxDist, splitPoint


def normalize(v):
    if any(isnan(v)) or any(isinf(v)):
        print('raise')
        raise ValueError("Input vector contains NaN or Inf values")
    
    norm = linalg.norm(v)
    
    # Handle zero vector case
    if norm == 0:
        return v  # or return a zero vector, or handle as needed
    
    return v / norm


# Returns Bézier representation for a given curve representation and maximum error
# letter_cr - curve representation of the letter skeleton
def bezier_representation(letter_cr, max_error):
    # letter is a list of lists
    letter_curves = []
    for c in letter_cr:
        c_list = []
        c_bezier = fitCurve(array(c), max_error)
        c_list.append(len(c_bezier))
        c_list.append(c_bezier)
        
        letter_curves.append(c_list)
    return letter_curves



# 2) Plot Bézier Representation of the Letter




def rotate_point_90_clockwise(point):
    x, y = point
    return array((y, -x))


def plot_bezier_representation(bez_repr, curve_repr):
    fig, ax = plt.subplots()
    
    for stroke in bez_repr:
        _, bez_curves = stroke
        for bc in bez_curves:
            # Get the control points as numpy arrays
            cp0, cp1, cp2, cp3 = [rotate_point_90_clockwise(cp) for cp in bc]

            # Plot the control points
            ax.plot([cp0[0], cp1[0], cp2[0], cp3[0]], [cp0[1], cp1[1], cp2[1], cp3[1]], 'ro')  # Red dots

            # Plot the control polygon
            ax.plot([cp0[0], cp1[0]], [cp0[1], cp1[1]], 'r--', linewidth = 1)  # Dashed line for control polygon
            ax.plot([cp1[0], cp2[0]], [cp1[1], cp2[1]], 'r--', linewidth = 1)
            ax.plot([cp2[0], cp3[0]], [cp2[1], cp3[1]], 'r--', linewidth = 1)
            
            t_values = linspace(0, 1, 100)
            bezier_points = array([q([cp0, cp1, cp2, cp3], t) for t in t_values])
            ax.plot(bezier_points[:, 0], bezier_points[:, 1], 'b-', linewidth = 4)  # Blue line for the Bezier curve

    for stroke_curve in curve_repr:
        curve_points = array([rotate_point_90_clockwise(p) for p in stroke_curve])
        ax.plot(curve_points[:, 0], curve_points[:, 1], 'go', markersize = 3)  # Green for actual curve points

    plt.title('Bezier Curve Fitting')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()
    


# 3) Generate an image of a letter from its Bézier representation


def generate_letter_from_bezier(bez_repr, size=128, num_points=1000):
    new_letter = ones((size, size), uint8)
    
    for stroke in bez_repr:
        for cp in stroke[1]:
            curve_points = [q(cp,t) for t in linspace(0, 1, num_points)]
            curve_points = [(int(p[0]), int(p[1])) for p in curve_points]
            
            for p in curve_points:
                if 0 <= p[0] < size and 0 <= p[1] < size:
                    new_letter[p[0],p[1]] = 0
    
    return Image.fromarray(new_letter*255)