import os
import shutil
from tqdm import tqdm

def move_files_with_substring(source_folder, target_folder, substring):
    """
    Move files from source_folder to target_folder that contain the substring in their filenames.
    
    Args:
    - source_folder (str): The path to the source directory.
    - target_folder (str): The path to the target directory.
    - substring (str): The substring to search for in filenames.
    """
    # Ensure target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all files in the source folder
    files_to_move = os.listdir(source_folder)
    #files = random.sample([f for f in all_files if os.path.isfile(os.path.join(source_folder, f))], 100) if len(all_files) > 100 else [f for f in all_files if os.path.isfile(os.path.join(source_folder, f))]
    
    # Filter files that contain the substring
    #files_to_move = [f for f in files if f.startswith(substring)]
    #files_to_move = [f for f in files if substring in f]
    import random
    files_to_move = random.sample(files_to_move, 100) if len(files_to_move) > 100 else files_to_move

    print(len(files_to_move))
    # Move files with tqdm progress bar
    for file in tqdm(files_to_move, desc="Moving files"):
        #shutil.copy(os.path.join(source_folder, file), os.path.join(target_folder, file))
        shutil.copy(os.path.join("/home/karan/sda_link/datasets/Ferret_data/Dataset_2/Mixed_16KHz/", file), os.path.join(target, file))

        #shutil.move(os.path.join(source_folder, file), os.path.join(target_folder, file))
        

# Example usage
source = "/home/karan/sda_link/datasets/Ferret_data/Dataset_2/embeddings/wavs/female_alone_random/"
target = "/home/karan/sda_link/datasets/Ferret_data/Dataset_2/embeddings/wavs/mix_random/"
substring = "ffa_fbe_fku"

move_files_with_substring(source, target, substring)

#source = "/home/karan/sda_link/datasets/Ferret_data/Dataset_2/test_Mixed_16KHz/"
#target = ""
#substring = "mfa_mbe_mku"

#move_files_with_substring(source, target, substring)


