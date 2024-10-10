import os
import random
import torchaudio

def pick_random_wavs(folder_path, num_files=100):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
    selected_files = random.sample(all_files, num_files)
    return selected_files

# Example usage
folder_path = "/home/karan/sda_link/datasets/Ferret_data/Dataset_2/embeddings/wavs/female_alone_random/"        #/home/karan/sda_link/datasets/Ferret_data/Dataset_2/test_F_mixed_16KHz/"
picked_files = pick_random_wavs(folder_path)

import torchaudio
from tqdm import tqdm
# function to load a wav from one folder and load the same wav from another folder
def load_wav(path_original, path_mixed, path_save):
    for file in tqdm(path_original):
        wav_original = torchaudio.load(file)[0]
        name = file.split("/")[-1]
        wav_mixed = torchaudio.load(f"{path_mixed}/{name}")[0]
        wav = wav_mixed - wav_original
        # save the wav
        torchaudio.save(f"{path_save}/{name}", wav, 16000)
    return 

load_wav(path_original=picked_files, path_mixed="/home/karan/sda_link/datasets/Ferret_data/Dataset_2/Mixed_16KHz/", path_save="/home/karan/sda_link/datasets/Ferret_data/Dataset_2/embeddings/wavs/male_alone_random")