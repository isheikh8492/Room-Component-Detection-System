import cv2
import numpy as np
import os
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Directory containing the cropped icons
icons_dir = "components"  # Modify this with the actual path to your icons directory

# Room plan image
room_plan_path = "data/RoomPlan.jpg"  # Modify this with your room plan image path
room_plan = cv2.imread(room_plan_path)

if room_plan is None:
    logging.error(f"Failed to load room plan image from {room_plan_path}")
    exit()
else:
    logging.info(f"Room plan image loaded successfully from {room_plan_path}")

# Threshold for template matching (adjust based on results)
threshold = 0.8

# Dictionary to store the count of matches for each component
match_counts = defaultdict(int)

# Output file to store results
output_file_path = "match_counts.txt"


# Function to apply non-maximum suppression
def non_max_suppression(locations, threshold_distance=10):
    """Suppresses overlapping detections"""
    if len(locations[0]) == 0:
        return []

    logging.debug(f"Non-maximum suppression on {len(locations[0])} locations")

    # Convert locations into a list of (x, y) coordinates
    points = list(zip(locations[1], locations[0]))

    # Sort points based on their score (descending)
    points = sorted(points, key=lambda p: (p[0], p[1]))

    suppressed = []
    for p in points:
        if all(
            np.linalg.norm(np.array(p) - np.array(s)) > threshold_distance
            for s in suppressed
        ):
            suppressed.append(p)

    logging.debug(f"{len(suppressed)} locations after suppression")
    return suppressed


# Sliding window and image pyramid function
def sliding_window_pyramid(room_plan, icon, pyramid_scale=1.5, window_step=50):
    """Creates an image pyramid and applies sliding window for template matching"""
    scale_count = 0
    while room_plan.shape[0] >= icon.shape[0] and room_plan.shape[1] >= icon.shape[1]:
        result = cv2.matchTemplate(room_plan, icon, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        yield scale_count, locations
        # Scale down the room plan image
        room_plan = cv2.resize(
            room_plan,
            (
                int(room_plan.shape[1] / pyramid_scale),
                int(room_plan.shape[0] / pyramid_scale),
            ),
        )
        scale_count += 1


# Function to process matching of an icon
def process_icon(icon_file):
    """Processes template matching for a given icon file"""
    icon_path = os.path.join(icons_dir, icon_file)
    logging.info(f"Processing icon: {icon_file}")

    # Load the icon image
    icon = cv2.imread(icon_path, cv2.IMREAD_COLOR)
    if icon is None:
        logging.warning(f"Failed to load icon image from {icon_path}")
        return None

    # Use the entire file name (without the extension) as the key
    component_name = os.path.splitext(icon_file)[0]
    count = 0

    # Perform sliding window and pyramid matching
    for scale_count, locations in sliding_window_pyramid(room_plan, icon):
        logging.debug(
            f"Found {len(locations[0])} potential matches for {component_name} at pyramid scale {scale_count}"
        )
        filtered_locations = non_max_suppression(locations)
        count += len(filtered_locations)

    logging.info(f"Total matches found for {component_name}: {count}")
    return component_name, count


# Main function to process icons sequentially
def main():
    # Sequentially process each icon file in the directory
    for icon_file in os.listdir(icons_dir):
        result = process_icon(icon_file)
        if result:
            component_name, count = result
            if count > 0:
                match_counts[component_name] += count

    # Save the match counts to a file
    with open(output_file_path, "w") as f:
        for component, count in match_counts.items():
            f.write(f"Component: {component}, Count: {count}\n")
            logging.info(f"Component: {component}, Count: {count}")

    logging.info(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    main()
