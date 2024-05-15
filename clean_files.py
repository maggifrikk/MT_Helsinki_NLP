import os
import sys

output_dir = 'output_files'  # Adjust the directory path if necessary
output_dir2 = 'output_files2'

# Iterate through each file in the output directory
for filename in os.listdir(output_dir):
    if filename.endswith("_translation"):  # Ensuring we're only modifying the correct files
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()  # Read all lines in the file
        
        lines = lines[:-1]  # Remove the last line

        file_path2 = os.path.join(output_dir2, filename)
        # Write the modified lines back to the file
        with open(file_path2, 'w', encoding='utf-8') as file:
            file.writelines(lines)
