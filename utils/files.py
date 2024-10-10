import glob
from tqdm import tqdm

def flatten_list(list_of_lists):
    flat_list = [item for sublist in list_of_lists for item in sublist]
    return flat_list

def get_files(path, extension=".wav"):
    filenames = []
    for filename in tqdm(glob.iglob(f"{path}/**/*{extension}", recursive=True)):
        print(filename)
        filenames += [filename]
    return filenames

def get_wav_files(path, extension=".wav"):
    filenames = []
    for filename in glob.iglob(f"{path}/*{extension}", recursive=True):
        filenames += [filename]
    return filenames

