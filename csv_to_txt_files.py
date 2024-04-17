import csv
import os
import sys

def convert_csv_to_txt(csv_file_path, output_dir, filename_template):
    # Create the directory for the output files
    os.makedirs(output_dir, exist_ok = True)

    with open(csv_file_path, mode = 'r', encoding = 'utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Check if 'Id' field is present and not empty
            id_value = row.get('\ufeffId', '').strip()
            if not id_value:
                continue # Skip any rows without an ID

            # Create filename based on the template and ID
            file_name = filename_template.format(id = id_value)
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, mode = 'w', encoding = 'utf-8') as text_file:
                text_file.write(row['Content'])

    print('Content writing complete.')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python csv_to_txt.py <csv_file> <output_dir> <filename_template>')
        sys.exit(1)

    csv_file_path = sys.argv[1]
    output_dir = sys.argv[2]
    filename_template = sys.argv[3]

    convert_csv_to_txt(csv_file_path, output_dir, filename_template)

