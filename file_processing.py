import os
import shutil

source_dir = 'wmt21-news-systems/txt/references'
output_dir = 'reference_files'

source_files_dir = 'source_files'

source_files = os.listdir(source_files_dir)
print(source_files)

for filename in os.listdir(source_dir):
    print(filename)
    file_path = os.path.join(source_dir, filename)
    if filename in source_files:
        shutil.copy(file_path, output_dir)