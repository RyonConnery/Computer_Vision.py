# Ryon Connery
# Advanced Feature Detection

import os
import cv2


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")
MATCH_THRESHOLD = 5
RATIO_TEST = 0.75


def get_image_files(folder_path):
    """Return supported image files from the top level of the scene folder."""
    image_files = []

    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)

        if os.path.isfile(full_path) and file_name.lower().endswith(SUPPORTED_EXTENSIONS):
            image_files.append(file_name)

    image_files.sort()
    return image_files


def create_descriptor(descriptor_name):
    """Create one of the two required image descriptor algorithms."""
    if descriptor_name == "AKAZE":
        return cv2.AKAZE_create()

    if descriptor_name == "ORB":
        return cv2.ORB_create(nfeatures=2000)

    raise ValueError("Descriptor must be AKAZE or ORB.")


def create_keypoints_and_descriptors(image, descriptor):
    """Create keypoints and binary descriptors for an image."""
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return descriptor.detectAndCompute(greyscale, None)


def find_good_matches(marker_descriptors, scene_descriptors):
    """Match marker descriptors to scene descriptors using a Brute Force matcher."""
    if marker_descriptors is None or scene_descriptors is None:
        return []

    if len(scene_descriptors) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(
        marker_descriptors,
        scene_descriptors,
        k=2
    )

    good_matches = []

    # Lowe's ratio test keeps only stronger descriptor correspondences.
    for pair in matches:
        if len(pair) == 2:
            first_match, second_match = pair

            if first_match.distance < RATIO_TEST * second_match.distance:
                good_matches.append(first_match)

    return good_matches


def save_marker_keypoint_overlay(marker_image, marker_keypoints, output_path):
    """Draw and save the marker image with all detected keypoint locations."""
    marker_overlay = cv2.drawKeypoints(
        marker_image,
        marker_keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(output_path, marker_overlay)

    cv2.namedWindow("Marker Keypoints", cv2.WINDOW_NORMAL)
    cv2.imshow("Marker Keypoints", marker_overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_scenes(
    scene_folder,
    output_folder,
    descriptor,
    marker_keypoints,
    marker_descriptors
):
    """Batch-process every scene and save positive and negative marker detections."""
    scene_files = get_image_files(scene_folder)
    print(f"Number of scene image files loaded: {len(scene_files)}")

    positives_folder = os.path.join(output_folder, "Positives")
    negatives_folder = os.path.join(output_folder, "Negatives")

    os.makedirs(positives_folder, exist_ok=True)
    os.makedirs(negatives_folder, exist_ok=True)

    positive_count = 0
    negative_count = 0

    for index, file_name in enumerate(scene_files, start=1):
        scene_path = os.path.join(scene_folder, file_name)
        scene_image = cv2.imread(scene_path, cv2.IMREAD_COLOR)

        if scene_image is None:
            print(f"[{index}/{len(scene_files)}] Unable to read: {file_name}")
            continue

        scene_keypoints, scene_descriptors = create_keypoints_and_descriptors(
            scene_image,
            descriptor
        )

        good_matches = find_good_matches(
            marker_descriptors,
            scene_descriptors
        )

        if len(good_matches) >= MATCH_THRESHOLD:
            # Draw only the scene keypoints that matched the marker image.
            matched_scene_keypoints = [
                scene_keypoints[match.trainIdx]
                for match in good_matches
            ]

            positive_overlay = cv2.drawKeypoints(
                scene_image,
                matched_scene_keypoints,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            output_path = os.path.join(positives_folder, file_name)
            cv2.imwrite(output_path, positive_overlay)

            positive_count += 1
            print(
                f"[{index}/{len(scene_files)}] Positive: "
                f"{file_name} ({len(good_matches)} matches)"
            )

        else:
            output_path = os.path.join(negatives_folder, file_name)
            cv2.imwrite(output_path, scene_image)

            negative_count += 1
            print(
                f"[{index}/{len(scene_files)}] Negative: "
                f"{file_name} ({len(good_matches)} matches)"
            )

    print(
        f"\nResults: {positive_count} positive, "
        f"{negative_count} negative."
    )
    print(
        f"Positive detection threshold: "
        f"{MATCH_THRESHOLD} good keypoint matches."
    )


def main():
    """Run the marker keypoint and scene matching workflow."""
    marker_path = input(
        "Enter the path to the marker IMAGE file: "
    ).strip().strip('"')

    marker_image = cv2.imread(marker_path, cv2.IMREAD_COLOR)

    if marker_image is None:
        print("The marker image could not be read.")
        return

    print("\nDescriptor options:")
    print("1 - AKAZE")
    print("2 - ORB")

    descriptor_choice = input(
        "Select the image descriptor algorithm: "
    ).strip()

    if descriptor_choice == "1":
        descriptor_name = "AKAZE"
    elif descriptor_choice == "2":
        descriptor_name = "ORB"
    else:
        print("Invalid descriptor selection.")
        return

    descriptor = create_descriptor(descriptor_name)

    marker_keypoints, marker_descriptors = create_keypoints_and_descriptors(
        marker_image,
        descriptor
    )

    print(
        f"{descriptor_name} created "
        f"{len(marker_keypoints)} marker keypoints."
    )

    marker_output_path = input(
        "Enter the path and filename for the marker keypoint overlay: "
    ).strip().strip('"')

    save_marker_keypoint_overlay(
        marker_image,
        marker_keypoints,
        marker_output_path
    )

    scene_folder = input(
        "Enter the path to the scene IMAGE folder: "
    ).strip().strip('"')

    if not os.path.isdir(scene_folder):
        print("The scene folder could not be found.")
        return

    output_folder = input(
        "Enter the output folder for scene results: "
    ).strip().strip('"')

    os.makedirs(output_folder, exist_ok=True)

    print(
        f"\nProcessing scenes with {descriptor_name} "
        f"using a {MATCH_THRESHOLD}-match threshold..."
    )

    process_scenes(
        scene_folder,
        output_folder,
        descriptor,
        marker_keypoints,
        marker_descriptors
    )

    print("\nScene batch processing complete.")


if __name__ == "__main__":
    main()
