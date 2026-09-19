# Ryon Connery
# Simple Feature Detection

import os
import cv2


def build_output_path(base_path, transformation_name):
    """Create the required output filename from the base image path and transformation name."""
    base_root, _ = os.path.splitext(base_path)
    return f"{base_root}-{transformation_name}.png"


def save_active_image(base_path, transformation_name, image):
    """Save the currently displayed image using the required naming structure."""
    output_path = build_output_path(base_path, transformation_name)
    if cv2.imwrite(output_path, image):
        print(f"Saved: {output_path}")
    else:
        print("The image could not be saved.")


def create_transforms(base_image):
    """Create the four required Section 1 transformations from the original base image."""
    rotate_image = cv2.rotate(base_image, cv2.ROTATE_90_CLOCKWISE)
    flip_image = cv2.flip(base_image, 1)

    gaussian_kernel = (9, 9)
    blur_image = cv2.GaussianBlur(base_image, gaussian_kernel, 0)

    # Canny requires a single-channel image, so convert the base image to greyscale first.
    if base_image.ndim == 2:
        grey_image = base_image
    else:
        grey_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    edge_image = cv2.Canny(grey_image, 100, 200)

    return {
        "rotate-clockwise": rotate_image,
        "flip-horizontal": flip_image,
        "gaussian-blur": blur_image,
        "canny-edge": edge_image,
    }, gaussian_kernel


def draw_contour_bounds(base_image, bounds_type):
    """Detect contours and draw the selected enclosing boundary on a copy of the base image."""
    if base_image.ndim == 2:
        grey_image = base_image
        display_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        grey_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        display_image = base_image.copy()

    # Create a binary image for contour detection.
    _, binary_image = cv2.threshold(
        grey_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw each detected contour and its selected boundary.
    for contour in contours:
        if cv2.contourArea(contour) <= 0:
            continue

        cv2.drawContours(display_image, [contour], -1, (0, 255, 0), 2)

        if bounds_type == "rectangle":
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(
                display_image,
                (x, y),
                (x + width, y + height),
                (255, 0, 0),
                2,
            )

        elif bounds_type == "minimum-rectangle":
            rectangle = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rectangle)
            box = box.astype(int)
            cv2.drawContours(display_image, [box], 0, (255, 0, 0), 2)

        elif bounds_type == "minimum-circle":
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            cv2.circle(display_image, center, int(radius), (255, 0, 0), 2)

    return display_image


def main():
    # Prompt the user for the required base image path.
    base_path = input("Enter the path to the input IMAGE file: ").strip().strip('"')

    base_image = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)

    if base_image is None:
        print("The image could not be read. Check the file path and file name.")
        raise SystemExit

    transforms, gaussian_kernel = create_transforms(base_image)

    # Section 3 requires contour detection with saved output for all three bounding types.
    contour_images = {
        "contour-rectangle": draw_contour_bounds(base_image, "rectangle"),
        "contour-minimum-rectangle": draw_contour_bounds(base_image, "minimum-rectangle"),
        "contour-minimum-circle": draw_contour_bounds(base_image, "minimum-circle"),
    }

    print("Keyboard controls:")
    print("R - Rotate 90 degrees clockwise")
    print("F - Flip horizontally")
    print(f"B - Gaussian blur using kernel size {gaussian_kernel}")
    print("E - Canny edge detection")
    print("1 - Contour with bounding rectangle")
    print("2 - Contour with minimum bounding rectangle")
    print("3 - Contour with minimum enclosing circle")
    print("S - Save the currently displayed image")
    print("Press the same transformation key again to return to the base image.")
    print("Press any other key to close the image display.")

    cv2.namedWindow("Simple Feature Detection", cv2.WINDOW_NORMAL)

    active_name = "base"
    active_image = base_image
    cv2.imshow("Simple Feature Detection", active_image)

    key_map = {
        ord("r"): ("rotate-clockwise", transforms["rotate-clockwise"]),
        ord("R"): ("rotate-clockwise", transforms["rotate-clockwise"]),
        ord("f"): ("flip-horizontal", transforms["flip-horizontal"]),
        ord("F"): ("flip-horizontal", transforms["flip-horizontal"]),
        ord("b"): ("gaussian-blur", transforms["gaussian-blur"]),
        ord("B"): ("gaussian-blur", transforms["gaussian-blur"]),
        ord("e"): ("canny-edge", transforms["canny-edge"]),
        ord("E"): ("canny-edge", transforms["canny-edge"]),
        ord("1"): ("contour-rectangle", contour_images["contour-rectangle"]),
        ord("2"): (
            "contour-minimum-rectangle",
            contour_images["contour-minimum-rectangle"],
        ),
        ord("3"): ("contour-minimum-circle", contour_images["contour-minimum-circle"]),
    }

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("s"), ord("S")):
            save_active_image(base_path, active_name, active_image)
            continue

        if key in key_map:
            selected_name, selected_image = key_map[key]

            # Pressing the same key again returns to the original base image.
            if active_name == selected_name:
                active_name = "base"
                active_image = base_image
            else:
                active_name = selected_name
                active_image = selected_image

            cv2.imshow("Simple Feature Detection", active_image)
            continue

        break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
