import pandas as pd
from pathlib import Path
from glob import glob


# specify base data directory
data_dir = Path("data")

# specify text file name
text_file = "image_paths.csv"

def create_image_paths_file(data_dir, text_file):
    if not (data_dir / text_file).exists():
        image_files = sorted(glob(str(data_dir / 'images' / '*')))
        label_files = sorted(glob(str(data_dir / 'labels' / '*')))

        df = pd.DataFrame({'image_path': image_files, 'label_path': label_files})
        file = df.to_csv(data_dir / text_file, index=False)
        print("Created image_paths.csv file")