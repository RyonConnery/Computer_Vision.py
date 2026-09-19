# Ryon Connery
# Real-World Application and Public Data Sets
#
# Real-world use case:
# Automatically detect and blur human faces before images are shared or
# published, supporting privacy-preserving image processing.
#
# Public dataset:
# Microsoft FaceSynthetics - 100-image sample dataset
# https://github.com/microsoft/FaceSynthetics
#
# Dataset citation:
# Wood, E., Baltrusaitis, T., Hewitt, C., Dziadzio, S., Cashman, T. J.,
# & Shotton, J. (2021). Fake it till you make it: Face analysis in the wild
# using synthetic data alone. Proceedings of the IEEE/CVF International
# Conference on Computer Vision, 3681-3691.

from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import cv2


DATASET_URL = (
    "https://facesyntheticspubwedata.z6.web.core.windows.net/"
    "iccv-2021/dataset_100.zip"
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def download_progress(block_number, block_size, total_size):
    """Display simple download progress for the public dataset."""
    if total_size <= 0:
        return

    downloaded = block_number * block_size
    percent = min(100, int(downloaded * 100 / total_size))
    print(f"\rDownloading public dataset: {percent}%", end="", flush=True)


def prepare_public_dataset(base_folder):
    """Download and extract the Microsoft FaceSynthetics 100-image sample."""
    dataset_folder = base_folder / "Assessment7_Data"
    archive_path = base_folder / "dataset_100.zip"

    existing_images = get_dataset_images(dataset_folder)

    if existing_images:
        print(
            f"Public dataset already available: "
            f"{len(existing_images)} image(s) found."
        )
        return dataset_folder

    dataset_folder.mkdir(parents=True, exist_ok=True)

    print("Downloading Microsoft FaceSynthetics 100-image public dataset...")

    try:
        urlretrieve(
            DATASET_URL,
            archive_path,
            reporthook=download_progress
        )
        print()
    except Exception as error:
        print()
        raise RuntimeError(
            "The public dataset could not be downloaded. "
            "Check the internet connection and run Assessment7.py again."
        ) from error

    print("Extracting public dataset...")

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(dataset_folder)

    archive_path.unlink(missing_ok=True)

    images = get_dataset_images(dataset_folder)

    if not images:
        raise RuntimeError(
            "The dataset downloaded successfully, but no source images "
            "were found after extraction."
        )

    print(f"Public dataset ready: {len(images)} source image(s) found.")
    return dataset_folder


def get_dataset_images(dataset_folder):
    """Return only source face images, excluding segmentation-mask images."""
    if not dataset_folder.exists():
        return []

    image_files = []

    for path in dataset_folder.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        # FaceSynthetics includes segmentation images named *_seg.png.
        # Those annotations are not input photographs for this application.
        if path.stem.lower().endswith("_seg"):
            continue

        image_files.append(path)

    image_files.sort()
    return image_files


def load_face_classifier():
    """Load OpenCV's installed frontal-face Haar cascade classifier."""
    cascade_path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )

    classifier = cv2.CascadeClassifier(str(cascade_path))

    if classifier.empty():
        raise RuntimeError(
            f"OpenCV could not load its face classifier: {cascade_path}"
        )

    return classifier, cascade_path


def create_blur_kernel(face_region):
    """Return a valid odd-numbered Gaussian kernel for a detected face."""
    height, width = face_region.shape[:2]
    smallest_dimension = min(height, width)

    kernel_size = min(99, smallest_dimension)

    if kernel_size % 2 == 0:
        kernel_size -= 1

    return max(3, kernel_size)


def process_dataset(dataset_folder, output_folder, classifier):
    """Detect faces, blur them, and save the processed public dataset."""
    image_files = get_dataset_images(dataset_folder)

    if not image_files:
        raise RuntimeError("No supported dataset images were found.")

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nNumber of public dataset images loaded: {len(image_files)}")
    print("Processing images...\n")

    processed_count = 0
    images_with_detections = 0
    total_faces_blurred = 0

    for index, image_path in enumerate(image_files, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            print(
                f"[{index}/{len(image_files)}] "
                f"Unable to read: {image_path.name}"
            )
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detections = classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        output_image = image.copy()
        faces_in_image = 0

        for x, y, width, height in detections:
            face_region = output_image[
                y:y + height,
                x:x + width
            ]

            if face_region.size == 0:
                continue

            kernel_size = create_blur_kernel(face_region)

            blurred_face = cv2.GaussianBlur(
                face_region,
                (kernel_size, kernel_size),
                0
            )

            output_image[
                y:y + height,
                x:x + width
            ] = blurred_face

            faces_in_image += 1
            total_faces_blurred += 1

        output_path = output_folder / image_path.name

        if not cv2.imwrite(str(output_path), output_image):
            print(
                f"[{index}/{len(image_files)}] "
                f"Unable to save: {output_path.name}"
            )
            continue

        processed_count += 1

        if faces_in_image > 0:
            images_with_detections += 1

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path.name}: {faces_in_image} face(s) blurred"
        )

    print("\nAssessment 7 processing complete.")
    print(f"Images processed: {processed_count}")
    print(
        f"Images with detected faces: "
        f"{images_with_detections}"
    )
    print(
        f"Total detected face regions blurred: "
        f"{total_faces_blurred}"
    )
    print(f"Processed images saved to: {output_folder}")


def main():
    """Run the complete Assessment 7 public-dataset workflow."""
    base_folder = Path(__file__).resolve().parent

    print("CSC-FPX4040 Assessment 7")
    print("OpenCV Face-Blurring Privacy Application")
    print(
        "Public dataset: Microsoft FaceSynthetics "
        "100-image sample"
    )
    print(
        "Use case: detect and blur faces before images "
        "are shared or published.\n"
    )

    dataset_folder = prepare_public_dataset(base_folder)

    classifier, cascade_path = load_face_classifier()

    print(f"OpenCV face classifier: {cascade_path}")

    output_folder = base_folder / "Assessment7_Output"

    process_dataset(
        dataset_folder,
        output_folder,
        classifier
    )


if __name__ == "__main__":
    main()
