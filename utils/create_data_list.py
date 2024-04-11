import os
import csv
import yaml

def create_dataset_file(data_folder, output_file=None, format=None):
    """
    Creates a dataset file with image-filepath, label_filepath pairs.

    Args:
        data_folder (str): Path to the folder containing images and labels.
        output_file (str, optional): Path to the output file. If not provided,
            the format will be inferred from the filename extension and the file
            will be created inside the data_folder.
        format (str, optional): Output file format ("csv" or "yaml"). If not
            provided, it will be inferred from the filename extension.
    """

    image_paths = []
    label_paths = []

    images_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")

    for filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, filename)
        label_path = os.path.join(labels_folder, filename)
        image_paths.append(image_path)
        label_paths.append(label_path)

    if not output_file:
        output_file = os.path.join(data_folder, "dataset_list.csv")  # Default output file inside data_folder

    if not format:
        _, ext = os.path.splitext(output_file)
        format = ext[1:]  # Extract format from extension (e.g., ".csv" -> "csv")

    with open(output_file, "w", newline="") as f:
        if format == "csv":
            writer = csv.writer(f)
            writer.writerow(["image_filepath", "label_filepath"])
            for image_path, label_path in zip(image_paths, label_paths):
                writer.writerow([image_path, label_path])
        elif format == "yaml":
            data = [{"image_filepath": image_path, "label_filepath": label_path}
                    for image_path, label_path in zip(image_paths, label_paths)]
            yaml.dump(data, f)
        else:
            raise ValueError("Invalid format. Choose 'csv' or 'yaml'.")

    print(f"Dataset file created at: {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create image segmentation dataset file.")
    parser.add_argument("-d", "--data_folder", required=True, help="Path to the data folder.")
    parser.add_argument("-o", "--output_file", help="Path to the output file (optional, defaults to 'dataset.csv').")
    args = parser.parse_args()

    create_dataset_file(args.data_folder, args.output_file)