from typing import Tuple
import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Genera una matriz de pesos para el carril izquierdo con un gradiente suave.
    
    En lugar de -2 y -1 constantes, usaremos un gradiente:
    - Más negativo en el borde izquierdo, gradualmente menos negativo hacia el centro.
    """
    height, width = img_shape
    steer_matrix = np.zeros((height, width), dtype=np.float32)

    # Creamos un vector de pesos que va de -2 (en la izquierda) a -0.5 (cerca del centro)
    # y luego se mantiene más cercano a -0.5 hacia la derecha.
    # Esto genera una dirección negativa más fuerte a la izquierda y más suave al acercarse al centro.
    left_side = np.linspace(-2, -1, width // 2, endpoint=True)
    right_side = np.full(width - width // 2, -1, dtype=np.float32)
    weights = np.concatenate([left_side, right_side])
    
    # Asignamos este perfil a todas las filas
    steer_matrix[:] = weights

    return steer_matrix

def get_steer_matrix_right_lane_markings(img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Genera una matriz de pesos para el carril derecho con un gradiente suave.
    
    Comenzamos desde un valor ligeramente negativo en la izquierda (-0.5)
    y vamos subiendo a positivo fuerte en la derecha (por ejemplo 2).
    """
    height, width = img_shape
    steer_matrix = np.zeros((height, width), dtype=np.float32)

    # Gradiente de -0.5 en el centro-izquierda a 2 en el borde derecho
    left_side = np.full(width // 2, -1, dtype=np.float32)
    right_side = np.linspace(-1, 2, width - width // 2, endpoint=True)
    weights = np.concatenate([left_side, right_side])
    
    # Asignamos este perfil a todas las filas
    steer_matrix[:] = weights

    return steer_matrix

def detect_lane_markings(image_bgr):
    """
    Detect the left (yellow) and right (white) lane markings from a BGR image.

    Args:
        image_bgr (np.ndarray): Input image in BGR color space.

    Returns:
        tuple: Binary masks for the left (yellow) and right (white) lane markings.
    """
    # Convert the image to HSV color space
    imghsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Define HSV bounds for yellow and white lane markings
    white_lower_hsv = np.array([10, 0, 150])
    white_upper_hsv = np.array([179, 50, 255])
    yellow_lower_hsv = np.array([0, 175, 0])
    yellow_upper_hsv = np.array([179, 255, 255])

    # Create binary masks for yellow and white lane markings
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    return mask_yellow, mask_white


