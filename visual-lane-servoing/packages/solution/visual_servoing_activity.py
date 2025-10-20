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

def detect_lane_markings(image_bgr):
    """
    Detecta las marcas de carril basándose únicamente en la segmentación por color
    y la posición espacial (izquierda/derecha).

    Args:
        image_bgr (np.ndarray): Imagen de entrada en el espacio de color BGR.

    Returns:
        tuple: Máscaras binarias para las marcas de carril izquierdas (amarillas) y derechas (blicas).
    """
    height, width, _ = image_bgr.shape

    # 1. Convertir la imagen al espacio de color HSV
    # ¡AQUÍ ESTÁ LA CORRECCIÓN!
    imghsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 2. Definir rangos de color y crear máscaras
    white_lower_hsv = np.array([0, 0, 150])
    white_upper_hsv = np.array([179, 50, 255])
    yellow_lower_hsv = np.array([20, 100, 100])
    yellow_upper_hsv = np.array([30, 255, 255])

    mask_white_full = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow_full = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    # 3. Crear máscara para eliminar el horizonte
    mask_ground = np.zeros_like(mask_white_full)
    mask_ground[180:, :] = 255

    # 4. Crear máscaras para dividir la imagen en mitades
    mask_left_half = np.zeros_like(mask_white_full)
    cv2.rectangle(mask_left_half, (0, 0), (width // 2, height), 255, -1)
    mask_right_half = np.zeros_like(mask_white_full)
    cv2.rectangle(mask_right_half, (width // 2, 0), (width, height), 255, -1)

    # 5. Combinar las máscaras para obtener el resultado final
    final_mask_yellow = cv2.bitwise_and(mask_yellow_full, mask_ground)
    final_mask_yellow = cv2.bitwise_and(final_mask_yellow, mask_left_half)

    final_mask_white = cv2.bitwise_and(mask_white_full, mask_ground)
    final_mask_white = cv2.bitwise_and(final_mask_white, mask_right_half)

    return final_mask_yellow, final_mask_white

