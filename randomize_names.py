import os
import csv
import random
import string

def generate_random_string(length=16):
    """Generate a random alphanumeric string."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def rename_files_with_mapping(directory_path, mapping_file="file_mapping.csv"):
    """Rename files in directory to random strings and save mapping to CSV."""
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return
    
    # Open CSV file to store mapping
    with open(mapping_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Original Filename", "New Filename"])  # header row
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if os.path.isfile(file_path):
                # Extract file extension
                _, file_extension = os.path.splitext(filename)
                
                # Generate new random filename
                new_filename = generate_random_string() + file_extension
                new_file_path = os.path.join(directory_path, new_filename)
                
                # Rename file
                os.rename(file_path, new_file_path)
                
                # Write mapping to CSV
                writer.writerow([filename, new_filename])
                
                print(f"Renamed: {filename} -> {new_filename}")
    
    print(f"\nMapping saved to {mapping_file}")

if __name__ == "__main__":
    # Change this path to the directory you want to process
    directory = "/home/matthew-morales/LabelStudioData/MeatSegmentation/ground-truth"
    rename_files_with_mapping(directory)
