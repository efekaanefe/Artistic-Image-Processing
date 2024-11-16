from common_imports import *

def thread_and_nails(
        image, 
        num_nails = 100, 
        num_threads = 300,
        save_path = None):
    
    # Function to get pixels on the line between two points
    def get_line_pixels(img, point1, point2):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.line(mask, point1, point2, (255), 1)
        y_coords, x_coords = np.where(mask == 255)
        line_pixels = list(zip(x_coords, y_coords))
        return line_pixels

    image = image.copy()

    width, height, _ = image.shape
    r = min(width//2,height//2)
    width = height = 2*r # cropped width and height

    angles = np.linspace(0, 2*np.pi, num_nails)
    x1 = width//2 + width*0.5*np.sin(angles)
    y1 = height//2 + height*0.5*np.cos(angles)

    canvas = np.ones((height, width, 1), dtype=np.uint8)* 255 # canvas to put threads on

    nails = np.array([x1, y1], dtype = "int").T
    nail = nails[0] # initial nail

    for i in range(num_threads):
        next_nail = nail
        best_line = None
        color_sum_ratio = 255*3*2*r # if gray image, divide by 3
        for j in range(1,num_nails):
            line = np.array([nail, nails[j]])
            line_pixels = get_line_pixels(image, nail, nails[j])
            B = len(line_pixels)
            line_pixels_array = np.array(line_pixels)
            pixel_colors = image[line_pixels_array[:,1], line_pixels_array[:,0]]
            A = np.sum(pixel_colors)
            if A/B < color_sum_ratio: # obscurity: where is the darkest line between two nails
                color_sum_ratio = A/B
                next_nail = nails[j]
                best_line = line

        cv2.line(image, nail, next_nail, (255,255,255), 1) # to not draw the same line over and over again
        cv2.line(canvas, nail, next_nail, (0), 1)
        nail = next_nail

    plt.imshow(canvas)

    if save_path:
        cv2.imwrite(save_path, canvas)
        print(f"Image saved to {save_path}")
    # cv2_imshow(image)
