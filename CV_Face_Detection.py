# Ryon Conery
# Detecting Faces

import os
import cv2

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def image_files_in_folder(folder):
    """Return every supported image file from the top level of the dataset folder."""
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(SUPPORTED_EXTENSIONS):
            files.append(name)
    return sorted(files)


def load_cascade(path):
    """Load one Haar cascade XML classifier."""
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise ValueError(f"Could not load Haar cascade: {path}")
    return cascade


def detect(image, cascade):
    """Run multiscale face detection on one image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


def make_result_folders(root):
    """Create the required Positives and Negatives folders."""
    positives = os.path.join(root, "Positives")
    negatives = os.path.join(root, "Negatives")
    os.makedirs(positives, exist_ok=True)
    os.makedirs(negatives, exist_ok=True)
    return positives, negatives


def run_single_classifier(dataset_folder, xml_path, output_folder):
    """Batch-process every dataset image with one Haar cascade."""
    files = image_files_in_folder(dataset_folder)
    print(f"\nSingle-classifier run: {len(files)} image files found.")

    cascade = load_cascade(xml_path)
    positives, negatives = make_result_folders(output_folder)

    positive_count = 0
    negative_count = 0

    for index, name in enumerate(files, start=1):
        source = os.path.join(dataset_folder, name)
        image = cv2.imread(source)

        if image is None:
            print(f"[{index}/{len(files)}] Could not read: {name}")
            continue

        faces = detect(image, cascade)

        if len(faces) > 0:
            result = image.copy()
            for x, y, w, h in faces:
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imwrite(os.path.join(positives, name), result)
            positive_count += 1
            print(f"[{index}/{len(files)}] Positive: {name}")
        else:
            cv2.imwrite(os.path.join(negatives, name), image)
            negative_count += 1
            print(f"[{index}/{len(files)}] Negative: {name}")

    print(f"Single-classifier results: {positive_count} positive, {negative_count} negative.")


def run_multiple_classifiers(dataset_folder, xml_paths, output_folder):
    """Batch-process every dataset image with every selected Haar cascade."""
    files = image_files_in_folder(dataset_folder)
    print(f"\nMultiple-classifier run: {len(files)} image files found.")

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
    ]

    classifiers = []
    for index, path in enumerate(xml_paths):
        classifiers.append((load_cascade(path), colors[index % len(colors)]))

    positives, negatives = make_result_folders(output_folder)

    positive_count = 0
    negative_count = 0

    for index, name in enumerate(files, start=1):
        source = os.path.join(dataset_folder, name)
        image = cv2.imread(source)

        if image is None:
            print(f"[{index}/{len(files)}] Could not read: {name}")
            continue

        result = image.copy()
        found_face = False

        for cascade, color in classifiers:
            faces = detect(image, cascade)

            if len(faces) > 0:
                found_face = True
                for x, y, w, h in faces:
                    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        if found_face:
            cv2.imwrite(os.path.join(positives, name), result)
            positive_count += 1
            print(f"[{index}/{len(files)}] Positive: {name}")
        else:
            cv2.imwrite(os.path.join(negatives, name), image)
            negative_count += 1
            print(f"[{index}/{len(files)}] Negative: {name}")

    print(f"Multiple-classifier results: {positive_count} positive, {negative_count} negative.")


def main():
    # One dataset folder is entered once. The program then processes every JPG/JPEG/PNG inside it.
    dataset_folder = input("Enter the DATASET FOLDER path: ").strip().strip('"')

    if not os.path.isdir(dataset_folder):
        print("Dataset folder not found.")
        return

    files = image_files_in_folder(dataset_folder)
    print(f"\nFound {len(files)} image files in the dataset folder.")

    single_xml = input("Enter the single Haar cascade XML path: ").strip().strip('"')
    single_output = input("Enter the SINGLE-CLASSIFIER output folder: ").strip().strip('"')

    run_single_classifier(dataset_folder, single_xml, single_output)

    print("\nEnter two or more Haar cascade XML paths separated by semicolons.")
    multiple_input = input("Cascade XML paths: ").strip()

    xml_paths = [
        item.strip().strip('"')
        for item in multiple_input.split(";")
        if item.strip()
    ]

    if len(xml_paths) < 2:
        print("At least two Haar cascade XML files are required for the multiple-classifier run.")
        return

    multiple_output = input("Enter the MULTIPLE-CLASSIFIER output folder: ").strip().strip('"')

    run_multiple_classifiers(dataset_folder, xml_paths, multiple_output)

    print("\nDataset batch processing complete.")


if __name__ == "__main__":
    main()
