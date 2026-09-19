# Ryon Connery
# Neural Networks

import json
import os

import cv2


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")
FACE_THRESHOLD = 0.50
PERSON_THRESHOLD = 0.50


def get_image_files(folder_path):
    """Return supported image files from the top level of the selected dataset folder."""
    image_files = []

    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)

        if os.path.isfile(full_path) and file_name.lower().endswith(SUPPORTED_EXTENSIONS):
            image_files.append(file_name)

    image_files.sort()
    return image_files


def create_output_folders(output_path):
    """Create the required Positives and Negatives result folders."""
    positives_path = os.path.join(output_path, "Positives")
    negatives_path = os.path.join(output_path, "Negatives")

    os.makedirs(positives_path, exist_ok=True)
    os.makedirs(negatives_path, exist_ok=True)

    return positives_path, negatives_path


def load_tensorflow_model(pb_path, pbtxt_path):
    """Load a pretrained TensorFlow network using OpenCV DNN."""
    return cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)


def process_face_dataset(model, dataset_path, output_path):
    """Batch-process one dataset with the supplied OpenCV TensorFlow face detector."""
    image_files = get_image_files(dataset_path)
    print(f"\nNumber of image files loaded: {len(image_files)}")

    positives_path, negatives_path = create_output_folders(output_path)

    positive_count = 0
    negative_count = 0

    for index, file_name in enumerate(image_files, start=1):
        source_path = os.path.join(dataset_path, file_name)
        image = cv2.imread(source_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"[{index}/{len(image_files)}] Unable to read: {file_name}")
            continue

        height, width = image.shape[:2]

        # The supplied OpenCV face-detection TensorFlow model expects a 300x300 blob
        # with the standard mean values used by this pretrained network.
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )

        model.setInput(blob)
        detections = model.forward()

        output_image = image.copy()
        found_face = False

        # TensorFlow SSD detections are returned as:
        # [batch_id, class_id, confidence, left, top, right, bottom].
        for detection_index in range(detections.shape[2]):
            confidence = float(detections[0, 0, detection_index, 2])

            if confidence >= FACE_THRESHOLD:
                found_face = True

                left = int(detections[0, 0, detection_index, 3] * width)
                top = int(detections[0, 0, detection_index, 4] * height)
                right = int(detections[0, 0, detection_index, 5] * width)
                bottom = int(detections[0, 0, detection_index, 6] * height)

                left = max(0, min(left, width - 1))
                top = max(0, min(top, height - 1))
                right = max(0, min(right, width - 1))
                bottom = max(0, min(bottom, height - 1))

                cv2.rectangle(
                    output_image,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2
                )

        if found_face:
            cv2.imwrite(os.path.join(positives_path, file_name), output_image)
            positive_count += 1
            print(f"[{index}/{len(image_files)}] Positive: {file_name}")
        else:
            cv2.imwrite(os.path.join(negatives_path, file_name), image)
            negative_count += 1
            print(f"[{index}/{len(image_files)}] Negative: {file_name}")

    print(
        f"\nFace-detection results: "
        f"{positive_count} positive, {negative_count} negative."
    )
    print(f"Face-detection threshold: {FACE_THRESHOLD:.2f}")


def load_classes(class_json_path):
    """Read the course-provided JSON class list used by the object detector."""
    with open(class_json_path, "r", encoding="utf-8") as class_file:
        classes = json.load(class_file)

    return classes


def process_person_dataset(model, classes, dataset_path, output_path):
    """Batch-process one dataset and keep only detections whose class is person."""
    image_files = get_image_files(dataset_path)
    print(f"\nNumber of image files loaded: {len(image_files)}")

    person_class_id = classes.index("person")
    positives_path, negatives_path = create_output_folders(output_path)

    positive_count = 0
    negative_count = 0

    for index, file_name in enumerate(image_files, start=1):
        source_path = os.path.join(dataset_path, file_name)
        image = cv2.imread(source_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"[{index}/{len(image_files)}] Unable to read: {file_name}")
            continue

        height, width = image.shape[:2]

        # The supplied SSD Inception TensorFlow model processes a 300x300 RGB blob.
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(300, 300),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False
        )

        model.setInput(blob)
        detections = model.forward()

        output_image = image.copy()
        found_person = False

        for detection_index in range(detections.shape[2]):
            class_id = int(detections[0, 0, detection_index, 1])
            confidence = float(detections[0, 0, detection_index, 2])

            if class_id == person_class_id and confidence >= PERSON_THRESHOLD:
                found_person = True

                left = int(detections[0, 0, detection_index, 3] * width)
                top = int(detections[0, 0, detection_index, 4] * height)
                right = int(detections[0, 0, detection_index, 5] * width)
                bottom = int(detections[0, 0, detection_index, 6] * height)

                left = max(0, min(left, width - 1))
                top = max(0, min(top, height - 1))
                right = max(0, min(right, width - 1))
                bottom = max(0, min(bottom, height - 1))

                cv2.rectangle(
                    output_image,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    output_image,
                    classes[class_id],
                    (left, max(20, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        if found_person:
            cv2.imwrite(os.path.join(positives_path, file_name), output_image)
            positive_count += 1
            print(f"[{index}/{len(image_files)}] Positive: {file_name}")
        else:
            cv2.imwrite(os.path.join(negatives_path, file_name), image)
            negative_count += 1
            print(f"[{index}/{len(image_files)}] Negative: {file_name}")

    print(
        f"\nPerson-detection results: "
        f"{positive_count} positive, {negative_count} negative."
    )
    print(f"Person-detection threshold: {PERSON_THRESHOLD:.2f}")


def main():
    """Run either the face-detection or person-detection neural-network workflow."""
    print("Assessment 6 - Neural Networks")
    print("1 - Face detection")
    print("2 - Person detection")

    mode = input("Select detection type: ").strip()

    if mode not in ("1", "2"):
        print("Invalid selection.")
        return

    pb_path = input("Enter the TensorFlow .pb model path: ").strip().strip('"')
    pbtxt_path = input("Enter the TensorFlow .pbtxt configuration path: ").strip().strip('"')

    dataset_path = input("Enter the DATASET FOLDER path: ").strip().strip('"')

    if not os.path.isdir(dataset_path):
        print("The dataset folder could not be found.")
        return

    output_path = input("Enter the output folder path: ").strip().strip('"')
    os.makedirs(output_path, exist_ok=True)

    model = load_tensorflow_model(pb_path, pbtxt_path)

    if mode == "1":
        process_face_dataset(
            model,
            dataset_path,
            output_path
        )

    else:
        class_json_path = input(
            "Enter the classes.json path: "
        ).strip().strip('"')

        classes = load_classes(class_json_path)

        process_person_dataset(
            model,
            classes,
            dataset_path,
            output_path
        )

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
