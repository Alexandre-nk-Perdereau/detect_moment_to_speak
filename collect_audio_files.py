import os
import shutil

def collect_audio_files(source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.ogg') and 'Robot' not in file:
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_dir)
                dest_file_dir = os.path.join(dest_dir, relative_path)
                
                if not os.path.exists(dest_file_dir):
                    os.makedirs(dest_file_dir)
                
                dest_file_path = os.path.join(dest_file_dir, file)
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied: {src_file_path} to {dest_file_path}")

if __name__ == "__main__":
    source_directory = "D:\\audio"
    destination_directory = "data"
    collect_audio_files(source_directory, destination_directory)
