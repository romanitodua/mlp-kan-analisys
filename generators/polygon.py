import math

from PIL import Image, ImageDraw
import random


def calculate_polygon_coords(center_x, center_y, size, n, angle_offset=0):
    coords = []
    angle = math.radians(360 / n)
    for i in range(n):
        x = center_x + size * math.cos(i * angle + math.radians(angle_offset))
        y = center_y + size * math.sin(i * angle + math.radians(angle_offset))
        coords.append((x, y))
    return coords


class Polygon:
    @staticmethod
    def draw_with_coords(polygon, center_x, center_y, size):
        for i in range(10):
            image = Image.new('RGB', (28, 28), 'white')
            draw = ImageDraw.Draw(image)
            match polygon:
                case "pentagon":
                    polygon_coords = calculate_polygon_coords(center_x, center_y, size, 5, random.randint(0, 360))
                    draw.polygon(polygon_coords, outline='black')
                    image.save(f'data/4p{i}.png')
                case "triangle":
                    polygon_coords = calculate_polygon_coords(center_x, center_y, size, 3, random.randint(0, 360))
                    draw.polygon(polygon_coords, outline='black')
                    image.save(f'data/0p{i}.png')
                case "heptagon":
                    polygon_coords = calculate_polygon_coords(center_x, center_y, size, 7, random.randint(0, 360))
                    draw.polygon(polygon_coords, outline='black')
                    image.save(f'data/2p{i}.png')
                case "square":
                    polygon_coords = calculate_polygon_coords(center_x, center_y, size, 4, random.randint(0, 360))
                    draw.sq(polygon_coords, outline='black')
                    image.save(f'data/1p{i}.png')
                case "hexagon":
                    polygon_coords = calculate_polygon_coords(center_x, center_y, size, 6, random.randint(0, 360))
                    draw.polygon(polygon_coords, outline='black')
                    image.save(f'data/3p{i}.png')
                case _:
                    print(
                        "non-valid polygon shape with provided, available shapes - pentagon,triangle,rombus,square,"
                        "hexagon")
