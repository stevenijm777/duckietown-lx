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
    for y in range(height):
        for x in range(width):
            if x < width // 2:
                steer_matrix[y, x] = - 0.3 * (x / (width // 2))

    return steer_matrix

def get_steer_matrix_right_lane_markings(img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Genera una matriz de pesos para el carril derecho con un gradiente suave.
    
    Comenzamos desde un valor ligeramente negativo en la izquierda (-0.5)
    y vamos subiendo a positivo fuerte en la derecha (por ejemplo 2).
    """
    height, width = img_shape
    steer_matrix = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            if x > width // 2:
                steer_matrix[y, x] = 0.1 - (0.1 * ((x - (width // 2)) / (width // 2)))

    return steer_matrix

import cv2
import numpy as np

def detect_lane_markings(image_bgr):
    """
    Detecta las marcas de carril izquierdas (amarillas) y derechas (blancas) de una imagen BGR,
    separando cada carril en su lado correspondiente de la imagen.

    Args:
        image_bgr (np.ndarray): Imagen de entrada en el espacio de color BGR.

    Returns:
        tuple: Máscaras binarias para las marcas de carril izquierdas (amarillas) y derechas (blancas).
    """
    height, width, _ = image_bgr.shape

    # Convertir la imagen al espacio de color HSV
    imghsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # --- Límites de color y máscara de horizonte (sin cambios) ---
    white_lower_hsv = np.array([0, 0, 150])
    white_upper_hsv = np.array([179, 50, 255])
    yellow_lower_hsv = np.array([20, 100, 100])
    yellow_upper_hsv = np.array([30, 255, 255])
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
    mask_ground = np.zeros_like(mask_white)
    mask_ground[180:, :] = 255
    mask_white = cv2.bitwise_and(mask_white, mask_ground)
    mask_yellow = cv2.bitwise_and(mask_yellow, mask_ground)

    # --- Detección de bordes (sin cambios) ---
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_gray_masked = cv2.bitwise_and(img_gray, img_gray, mask=mask_ground)
    sobel_y = cv2.Sobel(img_gray_masked, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobel_y = np.absolute(sobel_y)
    sobel_edges = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))
    combined_mask_white = cv2.bitwise_and(mask_white, sobel_edges)
    combined_mask_yellow = cv2.bitwise_and(mask_yellow, sobel_edges)

    # --- ¡NUEVO! Máscaras para dividir la imagen en mitades ---
    # Se crea una máscara negra del mismo tamaño que la imagen
    mask_left_half = np.zeros_like(combined_mask_yellow)
    mask_right_half = np.zeros_like(combined_mask_white)

    # Se rellena de blanco solo la mitad izquierda para la máscara amarilla
    cv2.rectangle(mask_left_half, (0, 0), (width // 2, height), 255, -1)
    # Se rellena de blanco solo la mitad derecha para la máscara blanca
    cv2.rectangle(mask_right_half, (width // 2, 0), (width, height), 255, -1)


    # Aplicar las máscaras de mitad a cada detección de color
    final_mask_yellow = cv2.bitwise_and(combined_mask_yellow, mask_left_half)
    final_mask_white = cv2.bitwise_and(combined_mask_white, mask_right_half)

    return final_mask_yellow, final_mask_white

