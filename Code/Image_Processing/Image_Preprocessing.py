import numpy as np
import os

from PIL import Image, ImageOps




def image_preprocessing(image, target_size=(128, 128), letter_size=120, padding_size=20):
    """
    Preprocesses a single image by extracting its bounding box, resizing, and normalizing it.

    Parameters:
    - image (PIL): input image.
    - target_size (tuple): Desired output size (width, height).
    - letter_size (int): Desired size of the letter.
    - padding_size (int): Desired size of the white margines around the letter.

    Returns:
    - Image object: The preprocessed image.
    """
    # Ensure grayscale mode
    image = image.convert("L")

    # Determine background color (white or black) based on pixel intensity
    pixel_values = np.array(image)
    white_pixels = np.sum(pixel_values == 255)
    black_pixels = np.sum(pixel_values == 0)

    # Invert if the background is black
    if black_pixels > white_pixels:
        image = ImageOps.invert(image)

    
    image_array = np.array(image)
    
    # Extract the bounding box.
    image_coordinates = np.column_stack(np.where(image_array < 255))
    if image_coordinates.size == 0:
        return image
    
    x_min, y_min = image_coordinates.min(axis=0)
    x_max, y_max = image_coordinates.max(axis=0)

    # Crop the image.
    cropped_image = image.crop((y_min, x_min, y_max+1, x_max+1))
    
    # Resize the image.
    cropped_width, cropped_height = cropped_image.size
    aspect_ratio = cropped_width / cropped_height
    
    if cropped_width > cropped_height:
        new_width = letter_size
        new_height = int(letter_size / aspect_ratio)
    else:
        new_height = letter_size
        new_width = int(letter_size * aspect_ratio)

    resized_image = cropped_image.resize((new_width, new_height), Image.ANTIALIAS)

    # Add padding to fit into the target size
    padded_image = add_padding(resized_image, target_size, padding_size)

    binarized_image = padded_image.point(lambda p: 0 if p < 200 else 255, mode='L')
    
    return binarized_image




def add_padding(image, target_size, padding_size):
    old_size = image.size  # (width, height)
    desired_size = max(old_size)  # Make it square by taking the largest dimension
    
    # Calculate padding
    delta_w = padding_size + (target_size[0] - old_size[0]) // 2
    delta_h = padding_size + (target_size[1] - old_size[1]) // 2
    
    # Add padding with extra white space
    padded_image = ImageOps.expand(image, (delta_w, delta_h), fill='white')

    # Now resize the image to the final target size (256x256)
    return padded_image.resize(target_size)