# Ryon Connery
# Working With Image Data

import cv2


# Prompt the user for the path to the base image required by the assessment.
image_path = input("Enter the path to the input IMAGE file: ").strip().strip('"')

# Read the image without forcing a color conversion so the original channel
# structure can be reported accurately.
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if image is None:
    print("The image could not be read. Check the file path and file name.")
    raise SystemExit

# Report the number of channels in the image.
if image.ndim == 2:
    channels = 1
else:
    channels = image.shape[2]

# Report the image resolution as rows and columns of pixels.
rows, columns = image.shape[:2]
print(f"Number of color channels: {channels}")
print(f"Resolution: {rows} rows x {columns} columns")

# Create a color version and a greyscale version for the Section 2 display.
if image.ndim == 2:
    greyscale_image = image.copy()
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
elif image.shape[2] == 4:
    color_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    greyscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
else:
    color_image = image.copy()
    greyscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Display the image in a resizable OpenCV window.
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

print("Press G in the active image window to switch between color and greyscale.")
print("Press any other key in the active image window to close the window.")

show_greyscale = False

while True:
    if show_greyscale:
        cv2.imshow("Image", greyscale_image)
    else:
        cv2.imshow("Image", color_image)

    key = cv2.waitKey(0) & 0xFF

    # G toggles between the color and greyscale display modes.
    if key in (ord("g"), ord("G")):
        show_greyscale = not show_greyscale
    else:
        break

cv2.destroyWindow("Image")

# Save the image to the user-specified path and file name.
saved_image_path = input(
    "Enter the path and file name to save the image: "
).strip().strip('"')

if cv2.imwrite(saved_image_path, image):
    print(f"Image saved successfully: {saved_image_path}")
else:
    print("The image could not be saved.")

# Save the greyscale image as required by the Section 2 scoring criterion.
greyscale_output_path = input(
    "Enter the path and file name to save the greyscale image: "
).strip().strip('"')

if cv2.imwrite(greyscale_output_path, greyscale_image):
    print(f"Greyscale image saved successfully: {greyscale_output_path}")
else:
    print("The greyscale image could not be saved.")

# Prompt the user for the mask image required by Section 3.
mask_path = input("Enter the path to the Mask IMAGE file: ").strip().strip('"')

# Read the mask as a one-channel greyscale image.
mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask_image is None:
    print("The mask image could not be read. Check the file path and file name.")
    raise SystemExit

# The mask and base image must use the same pixel dimensions.
if color_image.shape[:2] != mask_image.shape[:2]:
    print("The base image and mask must have the same resolution.")
    raise SystemExit

# Ensure the mask contains only the required values of 0 or 255.
_, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

# Copy the base image so the original remains available for display.
masked_image = color_image.copy()

# Where the mask pixel is 255, clear the corresponding base-image pixel
# to black, represented by (0, 0, 0) in a three-channel BGR image.
masked_image[binary_mask == 255] = (0, 0, 0)

# Display the Base, Mask, and Base + Mask images in three separate windows.
cv2.namedWindow("Base", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Base + Mask", cv2.WINDOW_NORMAL)

cv2.imshow("Base", color_image)
cv2.imshow("Mask", binary_mask)
cv2.imshow("Base + Mask", masked_image)

print("Press any key in an active OpenCV window to close the three images.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the masked image to the path and file name supplied by the user.
masked_output_path = input(
    "Enter the path and file name to save the masked image: "
).strip().strip('"')

if cv2.imwrite(masked_output_path, masked_image):
    print(f"Masked image saved successfully: {masked_output_path}")
else:
    print("The masked image could not be saved.")
