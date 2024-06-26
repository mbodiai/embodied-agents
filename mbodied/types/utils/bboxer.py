from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def draw_bounding_box_with_label(
    image_path: str, bbox: tuple, label: str, color: str = "red", width: int = 2
) -> Image:
    """Draws a bounding box with a label and shading on an image.

    Args:
        image_path (str): Path to the image file.
        bbox (tuple): Bounding box coordinates as (left, top, right, bottom).
        label (str): Label text for the bounding box.
        color (str): Color of the bounding box and label text. Default is red.
        width (int): Width of the bounding box lines. Default is 2.

    Returns:
        Image: Image object with the bounding box and label drawn.

    Example:
        image_with_bbox = draw_bounding_box_with_label("path/to/image.jpg", (50, 50, 150, 150), "Object")
    """
    # Open an image file
    with Image.open(image_path) as im:
        # Create a drawing object
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()

        # Draw the bounding box shadow
        shadow_offset = 3
        shadow_color = "black"
        for i in range(width):
            draw.rectangle(
                [
                    bbox[0] + shadow_offset - i,
                    bbox[1] + shadow_offset - i,
                    bbox[2] + shadow_offset + i,
                    bbox[3] + shadow_offset + i,
                ],
                outline=shadow_color,
            )

        # Draw the bounding box
        for i in range(width):  # Adjust for line width
            draw.rectangle([bbox[0] - i, bbox[1] - i, bbox[2] + i, bbox[3] + i], outline=color)

        # Calculate text size and position
        text_size = draw.textsize(label, font)
        text_position = (bbox[0], bbox[1] - text_size[1] - 5)  # Adjusted to give some padding

        # Draw text background rectangle with shading
        bg_rect = [text_position[0], text_position[1], text_position[0] + text_size[0], text_position[1] + text_size[1]]
        draw.rectangle(
            [
                bg_rect[0] + shadow_offset,
                bg_rect[1] + shadow_offset,
                bg_rect[2] + shadow_offset,
                bg_rect[3] + shadow_offset,
            ],
            fill="black",
        )
        draw.rectangle(bg_rect, fill=color)

        # Draw text
        draw.text(text_position, label, fill="white", font=font)

        return im


if __name__ == "__main__":
    image_path = Path("resources") / "xarm.jpg"
    bbox = (50, 50, 150, 150)
    label = "Object"
    image_with_bbox = draw_bounding_box_with_label(image_path, bbox, label)
    image_with_bbox.show()
