from common_imports import *

def img2ascii(
        image, 
        output_width=100, 
        save_path = None, 
        original_colors=True, 
        only_edges = False):
    """
    Draws ASCII characters based on edge direction using histogram analysis on 
    original image batches for each pixel in the resized image.
    """
    ASCII_CHARS_FOR_EDGES = {
        'horizontal': '-',
        'vertical': '|',
        'diagonal': '/',
        'anti-diagonal': '\\',
        'no_edge': ' '
    }
    ASCII_CHARS = "■@?OPoc:. "[::-1]
    ASCII_CHARS = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,^`'."[::-1]

    # Sobel filter on original edges
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)  # degrees

    height, width = gray_image.shape[:2]
    aspect_ratio = height / width
    new_height = int(output_width * aspect_ratio)

    # Determine the batch size for each resized pixel
    batch_width = width // output_width
    batch_height = height // new_height

    font_scale = 0.2
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    character_width = character_height = 8
    ascii_image_width = character_width * output_width
    ascii_image_height = character_height * new_height
    ascii_image = np.ones((ascii_image_height, ascii_image_width, 3), dtype=np.uint8) * 50

    for y in range(new_height):
        for x in range(output_width):
            # Extract the batch from the original image that corresponds to the current pixel in the resized image
            batch_magnitude = magnitude[y*batch_height:(y+1)*batch_height, x*batch_width:(x+1)*batch_width]
            batch_angle = angle[y*batch_height:(y+1)*batch_height, x*batch_width:(x+1)*batch_width]
            batch_gray = gray_image[y*batch_height:(y+1)*batch_height, x*batch_width:(x+1)*batch_width]
            batch_colors = image[y*batch_height:(y+1)*batch_height, x*batch_width:(x+1)*batch_width]

            if np.mean(batch_magnitude) > 130:
                predominant_angle = np.median(batch_angle)
                if -22.5 <= predominant_angle <= 22.5 or predominant_angle >= 157.5 or predominant_angle <= -157.5:
                    ascii_char = ASCII_CHARS_FOR_EDGES['horizontal']
                elif 67.5 <= predominant_angle <= 112.5 or -112.5 <= predominant_angle <= -67.5:
                    ascii_char = ASCII_CHARS_FOR_EDGES['vertical']
                elif 22.5 < predominant_angle < 67.5 or -157.5 < predominant_angle < -112.5:
                    ascii_char = ASCII_CHARS_FOR_EDGES['diagonal']
                else:
                    ascii_char = ASCII_CHARS_FOR_EDGES['anti-diagonal']
            else:
                # For non-edge pixels
                if not only_edges:
                    avg_intensity = np.mean(batch_gray)
                    ascii_index = int(avg_intensity / 255 * (len(ASCII_CHARS) - 1))
                    ascii_char = ASCII_CHARS[ascii_index]
                else:
                    ascii_char = " "

            # Set character color based on average color in batch
            avg_color = np.mean(batch_colors, axis=(0, 1)) if original_colors else (255, 255, 255)
            pos_x = x * character_width
            pos_y = y * character_height + character_height

            cv2.putText(
                ascii_image,
                ascii_char,
                (pos_x, pos_y),
                font,
                font_scale,
                (int(avg_color[0]), int(avg_color[1]), int(avg_color[2])),
                font_thickness,
                lineType=cv2.LINE_AA
            )

    plt.imshow(ascii_image)

    if save_path:
        cv2.imwrite(save_path, ascii_image)
        print(f"Image saved to {save_path}")
