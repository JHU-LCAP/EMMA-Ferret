import os
import numpy as np
import librosa
from utils.files import flatten_list
from utils.files import get_wav_files as get_files
from config import SE_Config
from random import *
from tqdm import tqdm
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import torch

def gen_log_space(limit, n):
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.int64)

y_log = gen_log_space(500, 128)

def make_loader(dataset, batch_size, model_type = "coherence_net"):
    if model_type == "coherence_net":
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size, 
                                             shuffle=True,
                                             collate_fn = collate_batch_coherence_net,
                                             pin_memory=True, num_workers=2)
    return loader

class Edinb_SE_Dataset(Dataset):
    def __init__(self, clean_files, noisy_files, 
                model_type = "coherence_net", 
                eval = False, 
                pad = False, 
                input_frame_size = 40, 
                tunning = None):
        """
        Arguments:
        clean_files: list - containing the path to the clean audio files 
        noisy_files: list - containing the path to the noisy audio files
        model_type: str - the type of model to be trained
            ["coherence_net", "CNN", "RNN"]
        eval: bool - if mode is eval or not
        pad: bool - if pad include small audio files in training after paddding, else reject the small examples
        """
        self.model_type = model_type
        self.eval = eval
        self.pad = pad
        self.window  = torch.hann_window(SE_Config.L_FRAME)
        self.input_frame_size = input_frame_size

        self.clean_dir = clean_files
        self.noisy_dir = noisy_files

        
        if not self.pad:
            if tunning is not None:
                self.clean_files = self._filter_files(self.get_files_tunning(clean_files, tunning))
                self.noisy_files = self._filter_files(self.get_files_tunning(noisy_files, tunning))
            else:
                self.clean_files = self._filter_files(get_files(clean_files))
                self.noisy_files = self._filter_files(get_files(noisy_files))
            
        if self.pad:
            if tunning is not None:
                self.clean_files = self.get_files_tunning(clean_files, tunning)
                self.noisy_files = self.get_files_tunning(noisy_files, tunning)
            else:
                self.clean_files = get_files(clean_files)
                self.noisy_files = get_files(noisy_files)
            
    def get_files_tunning(self, main_dir, tunning):
        """
        Converts the main dir + name_of_file in format to get the main path
        """
        files_list = []
        txt_file = open(tunning, "r")
        file_content = txt_file.read()
        content_list = file_content.split(",")
        txt_file.close()
        content_list = content_list[0].split("\n")

        for item in content_list[0:-1]:
            files_list.append(main_dir + item + ".wav")
        
        return files_list
        
    def __len__(self):
        """
        len depends on the type of the model,
        the coherence net learns both the noise and the speech depending on the indicator variable hence 2 times the samples
        :return: num_of_samples
        """
        if self.model_type == "coherence_net":
            len_dataset = len(self.clean_files)
        if self.model_type == "CNN":
            len_dataset = len(self.clean_files)
        
        return len_dataset
    
    def load_wav(self, path):
        wav, sr = torchaudio.load(path)
        # Calculate the padding length in samples (0.2 seconds * sample rate)
        padding_length = int(0.2 * sr)

        # Create padding tensor of zeros for both the start and end
        padding = torch.zeros(padding_length, dtype=wav.dtype).reshape(1, -1)
        wav = torch.cat((padding, wav, padding), dim=1)

        return wav, sr

    def __getitem__(self, idx):
        """
        outputs:-
        trainMixed: list  - A list of input mag (tensors) of spec (one for clean and one for noisy)
        trainSource: list  - A list of target mag (tensors) of spec (one for clean and one for noisy)
        trainLab: list  - A list of labels of spec ( [0, 1] for clean and [1, 0] for noisy)
        trainPhase: list  - A list of mixed signal phase of spec for retreival later (one for clean and one for noisy)
        wavs: list  - A list wavs (tensor) (one for clean and one for noisy)
        names: list  - A list of names of the wavs(one for clean and one for noisy)
        """
        
        ## make sure the same clean and noisy file is chosen for processing
        file_name_noisy = self.noisy_files[idx].split("/")[-1].split(".")[0]
        file_name_clean = self.clean_dir + "/" + file_name_noisy + ".wav" #self.clean_files[idx].split("/")[-1].split(".")[0]

        #print("The noisy file is", self.noisy_files[idx])
        #print("The clean file is", self.clean_files[idx])
    
        # load the audio file
        mixed_wav, sr_mixed_wav = self.load_wav(self.noisy_files[idx])    # Shape is [1, N]
        
        vocals_wav, sr_vocals_wav = self.load_wav(self.clean_dir + "/" + file_name_noisy + ".wav")  # Shape is [1, N],  self.clean_files[idx]
    
        noise_wav = mixed_wav - vocals_wav                                  # Shape is [1, N]

        # shifting business
        import random

        # Randomly decide whether to shift the pitch or not (50% chance)
        #if random.random() < 0.5:
        #    print("Shifting")
        #vocals_wav = pitch_shift_librosa(vocals_wav, sr_vocals_wav)
        #noise_wav = pitch_shift_librosa(noise_wav, sr_vocals_wav)
        #mixed_wav = vocals_wav + noise_wav

        
        if self.model_type == "coherence_net":
            trainMixed = []
            trainSource = []
            trainLab = []
            trainPhase = []
            wavs = []
            names = []
            
            # convert audio to spectrogram
            mixed_spec = torch.stft(mixed_wav, 
                                    hop_length= SE_Config.L_HOP, 
                                    n_fft= SE_Config.L_FRAME, 
                                    window= self.window, 
                                    return_complex= True) # Shape is [1, 2049, T]
            
            mixed_spec_mag = torch.abs(mixed_spec)[0:1, 0:500, 0:-1][0:1, list(y_log), 0:-1] #torch.log(torch.clamp(torch.abs(mixed_spec), min = 1e-5))  # Shape is [1, 2049, T]
            mixed_spec_phase = torch.angle(mixed_spec)[0:1, 0:500, 0:-1][0:1, list(y_log), 0:-1] # Shape is [1, 2049, T]
            
            vocals_spec = torch.stft(vocals_wav, 
                                     hop_length= SE_Config.L_HOP, 
                                     n_fft= SE_Config.L_FRAME, 
                                     window= self.window, 
                                     return_complex= True) #Shape is [1, 2049, T]            
            vocals_spec_mag = torch.abs(vocals_spec)[0:1, 0:500, 0:-1][0:1, list(y_log), 0:-1]  #torch.log(torch.clamp(torch.abs(vocals_spec), min = 1e-5))
            vocals_spec_phase = torch.angle(vocals_spec)[0:1, 0:500, 0:-1][0:1, list(y_log), 0:-1]

            noise_spec = torch.stft(noise_wav, 
                                    hop_length= SE_Config.L_HOP, 
                                    n_fft= SE_Config.L_FRAME, 
                                    window= self.window, 
                                    return_complex= True) #Shape is [1, 2049, T]
            noise_spec_mag = torch.abs(noise_spec)[0:1, 0:500, 0:-1][0:1, list(y_log), 0:-1]   #torch.log(torch.clamp(torch.abs(noise_spec), min = 1e-5))
            noise_spec_phase = torch.angle(noise_spec)[0:1, 0:500, 0:-1][0:1, list(y_log), 0:-1]

            if mixed_spec_mag.shape[-1] < 65:
                if self.pad:
                    mixed_spec_mag = torch.nn.functional.pad(mixed_spec_mag, (0, 64 - mixed_spec_mag.shape[-1]%64),  'constant')
                    mixed_spec_phase = torch.nn.functional.pad(mixed_spec_phase, (0, 64 - mixed_spec_phase.shape[-1]%64),  'constant')
                    
                    vocals_spec_mag = torch.nn.functional.pad(vocals_spec_mag, (0, 64 - vocals_spec_mag.shape[-1]%64),  'constant')

                    noise_spec_mag = torch.nn.functional.pad(noise_spec_mag, (0, 64 - noise_spec_mag.shape[-1]%64),  'constant')

            maxVal = torch.max(mixed_spec_mag)
            
            # normalise and save all the labels

            #if np.random.randn() < 0:
            for j in range(3):
                trainMixed.append(mixed_spec_mag/maxVal)
                trainPhase.append(mixed_spec_phase)
                
            
            #else:
            trainMixed.append(noise_spec_mag/maxVal)
            trainPhase.append(noise_spec_phase)
            trainMixed.append(vocals_spec_mag/maxVal)
            trainPhase.append(vocals_spec_phase)
                
            trainSource.append(noise_spec_mag/maxVal)
            trainLab.append(torch.Tensor([1,0,0]))

            wavs.append(noise_wav)
            names.append("noisy_" + file_name_noisy)

            trainSource.append(vocals_spec_mag / maxVal)
            trainLab.append(torch.Tensor([0,1,0]))
            
            wavs.append(vocals_wav)            
            names.append("clean_" + file_name_noisy)
            
            ### add the identity 
            trainSource.append(mixed_spec_mag/maxVal)
            trainLab.append(torch.Tensor([0,0,1]))

            trainSource.append(noise_spec_mag/maxVal)
            trainLab.append(torch.Tensor([0,0,1]))

            trainSource.append(vocals_spec_mag/maxVal)
            trainLab.append(torch.Tensor([0,0,1]))

            #wavs.append(noise_wav)
            #names.append("identity_" + file_name_noisy)
            
            #wavs.append(vocals_wav)
            #names.append("identity_" + file_name_noisy)
        
        return trainMixed, trainSource, trainLab, trainPhase, wavs, names
    
    def to_wav_torch(self, spec, phase, audio_len = None):
        invers_transformb = torchaudio.transforms.InverseMelScale(sample_rate=20000, n_stft=2049, n_mels=128)(spec)
        wav = torchaudio.transforms.GriffinLim(n_fft=4096, hop_length=512, momentum = 0.9)(invers_transformb)

        #spec = torch.exp(spec)
        #matrix = spec * torch.exp(1.j * phase) #output_len = librosa.frames_to_samples(in_frames, SE_Config.L_HOP, SE_Config.L_FRAME)
        #if audio_len is not None:
        #    wav = torch.istft(matrix, n_fft = SE_Config.L_FRAME, hop_length= SE_Config.L_HOP, length = audio_len)
        #else:
        #    wav = torch.istft(matrix, n_fft = SE_Config.L_FRAME, hop_length= SE_Config.L_HOP)
        return wav
    
    def __resample_if_necessary(self, audio, sr):
        if sr != SE_Config.SR:
            resampler = torchaudio.transforms.Resampler(sr, SE_Config.SR)
            audio_resampled = resampler(audio) 
        else:
            audio_resampled = audio
        return audio_resampled
    
    def _filter_files(self, list_of_wav_file_paths):
        """
        Removes the files from the list that have frames less than self.input_frame_size
        
        Inputs 
        List of files containing the wav paths
        
        Output:
        List of files containing the names of the files that have frames more than self.input_frame_size
        """
        filtered_files_list = []
        for files in list_of_wav_file_paths:
            wav, sr_wav = torchaudio.load(files)    # Shape is [1, N]
            #wav = self.__resample_if_necessary(wav, sr_wav)
            spec = torch.stft(wav,
                              hop_length= SE_Config.L_HOP, 
                              n_fft= SE_Config.L_FRAME, 
                              window= self.window, 
                              return_complex= True) # Shape is [1, 2049, T]
            if spec.shape[-1] > self.input_frame_size:
                filtered_files_list.append(files)
        
        return filtered_files_list.sort()

    def collate_batch_coherence_net(self, batch):
        """
        Inputs:- 
        List containing all items from _get_item in the dataset of len batch size

        Outputs:- 


        """
        input_spec = [] # list of all the input specs - len = 2*batch_size
        output_spec = []  # list of all the target specs - len = 2*batch_size
        indicator = []   # The indicator variable corresponding to the target spec - len = 2*batch_size
        phase_mixed = []
        raw_wavs_target = []
        speaker_ids = []

        for data in batch:
            
            for n_or_s in [0, 1]:
                trainMixed, trainSource, trainLab, trainPhase, wavs, names = data

                start = 0 #randint(0, trainMixed[0].shape[-1] - self.input_frame_size) # find a random start point for the

                input_spec.append(trainMixed[n_or_s][:, :, start: start + self.input_frame_size])  #start: start + self.input_frame_size
                output_spec.append(trainSource[n_or_s][:, :, start: start + self.input_frame_size]) # start: start + self.input_frame_size
                indicator.append(trainLab[n_or_s])
                raw_wavs_target.append(wavs[n_or_s])
                speaker_ids.append(names[n_or_s])
                phase_mixed.append(trainPhase[n_or_s])

            random_choice = choice([2, 3, 4])
            input_spec.append(trainMixed[random_choice][:, :, start: start + self.input_frame_size])  #start: start + self.input_frame_size
            output_spec.append(trainSource[random_choice][:, :, start: start + self.input_frame_size]) # start: start + self.input_frame_size
            indicator.append(trainLab[random_choice])
            #raw_wavs_target.append(wavs[random_choice])
            #speaker_ids.append(names[random_choice])
            #phase_mixed.append(trainPhase[random_choice])

                
        ## converting to tensors
        input_spec = torch.stack(input_spec, dim = 0)
        output_spec = torch.stack(output_spec, dim = 0)
        indicator = torch.stack(indicator, dim = 0)

        return input_spec, output_spec, indicator, phase_mixed, raw_wavs_target, speaker_ids
    
    def collate_batch_coherence_net_eval(self, batch, j = 1):
        """
        Inputs:- 
        List containing all items from _get_item in the dataset of len batch size
        j = noise or clean aka indicator variable
        Outputs:- 

        """
        input_spec = [] # list of all the input specs - len = 2*batch_size
        output_spec = []  # list of all the target specs - len = 2*batch_size
        indicator = []   # The indicator variable corresponding to the target spec - len = 2*batch_size
        phase_mixed = []
        raw_wavs_target = []
        speaker_ids = []

        for data in batch:
            trainMixed, trainSource, trainLab, trainPhase, wavs, names = data
            split_samples = trainMixed[j].shape[-1]//self.input_frame_size  + 1 #the number of samples after spliting

            for start in range(0, split_samples): #len(trainMixed)
                if start*self.input_frame_size + self.input_frame_size > trainMixed[j].shape[-1]:
                    trainMixed_padded = torch.nn.functional.pad(trainMixed[j], (0, start*self.input_frame_size + self.input_frame_size - trainMixed[j].shape[-1] + 1),  'constant')
                    input_spec.append(trainMixed_padded[:, :, start*self.input_frame_size : start*self.input_frame_size + self.input_frame_size])
                    
                    Output_padded = torch.nn.functional.pad(trainSource[j], (0, start*self.input_frame_size + self.input_frame_size - trainMixed[j].shape[-1] + 1),  'constant')
                    output_spec.append(Output_padded[:, :, start*self.input_frame_size : start*self.input_frame_size + self.input_frame_size])
                    
                    #phase_mixed.append(trainPhase[j][:, :, start*self.input_frame_size : -1])
                else:
                    input_spec.append(trainMixed[j][:, :, start: start + self.input_frame_size])
                    
                    output_spec.append(trainSource[j][:, :, start: start + self.input_frame_size])

            indicator.append(trainLab[j])
            raw_wavs_target.append(wavs[j])
            speaker_ids.append(names[j])
            phase_mixed.append(trainPhase[j])
            
        return  input_spec, output_spec, indicator, phase_mixed, raw_wavs_target, speaker_ids
    
    
class Edinb_SE_TestDataset(Dataset):
    def __init__(self, noisy_files, model_type = "coherence_net", pad = True, input_frame_size = 32, speaker_name_file = None):
        """
        Arguments:
        noisy_files: list - containing the path to the noisy audio files
        model_type: str - the type of model to be trained
            ["coherence_net", "CNN", "RNN"]
        eval: bool - if mode is eval or not
        pad: bool - if pad include small audio files in training after paddding, else reject the small examples
        """
        self.model_type = model_type
        self.eval = eval
        self.pad = pad
        self.window  = torch.hann_window(SE_Config.L_FRAME)
        self.input_frame_size = input_frame_size
        
        self.noisy_dir = noisy_files
        
        if not self.pad:
            if speaker_name_file is not None:
                self.noisy_files = self._filter_files(self.get_files_tunning(noisy_files, speaker_name_file))
            else:
                self.noisy_files = self._filter_files(get_files(noisy_files))
            
        if self.pad:
            if speaker_name_file is not None:
                self.noisy_files = self.get_files_tunning(noisy_files, speaker_name_file)
            else:
                self.noisy_files = get_files(noisy_files)
                
    def get_files_tunning(self, main_dir, tunning):
        """
        Converts the main dir + name_of_file in format to get the main path
        """
        files_list = []
        txt_file = open(tunning, "r")
        file_content = txt_file.read()
        content_list = file_content.split(",")
        txt_file.close()
        content_list = content_list[0].split("\n")

        for item in content_list[0:-1]:
            files_list.append(main_dir + item + ".wav")
        
        return files_list
        
    def __len__(self):
        """
        len depends on the type of the model,
        the coherence net learns both the noise and the speech depending on the indicator variable hence 2 times the samples
        :return: num_of_samples
        """
        if self.model_type == "coherence_net":
            len_dataset = len(self.noisy_files)
        if self.model_type == "CNN":
            len_dataset = len(self.noisy_files)
        
        return len_dataset

    def __getitem__(self, idx):
        """
        outputs:-
        trainMixed: list  - A list of input mag (tensors) of spec (one for clean and one for noisy)
        trainSource: list  - A list of target mag (tensors) of spec (one for clean and one for noisy)
        trainLab: list  - A list of labels of spec ( [0, 1] for clean and [1, 0] for noisy)
        trainPhase: list  - A list of mixed signal phase of spec for retreival later (one for clean and one for noisy)
        wavs: list  - A list wavs (tensor) (one for clean and one for noisy)
        names: list  - A list of names of the wavs(one for clean and one for noisy)
        """
        
        ## make sure the same clean and noisy file is chosen for processing
        file_name_noisy = self.noisy_files[idx].split("/")[-1].split(".")[0]
        
        # load the audio file
        mixed_wav, sr_mixed_wav = torchaudio.load(self.noisy_files[idx])   # Shape is [1, N]
        #mixed_wav = librosa.effects.pitch_shift(mixed_wav.numpy(), sr=sr, n_steps=12)
        #mixed_wav = torch.tensor(mixed_wav)
        #mixed_wav = self.__resample_if_necessary(mixed_wav, sr_mixed_wav)
        
        if self.model_type == "coherence_net":
            trainMixed = []
            trainPhase = []
            wavs = []
            names = []
            
            # convert audio to spectrogram
            mixed_spec = torch.stft(mixed_wav, 
                                    hop_length= SE_Config.L_HOP, 
                                    n_fft= SE_Config.L_FRAME, 
                                    window= self.window, 
                                    return_complex= True) # Shape is [1, 2049, T]
            mixed_spec_mag = torch.abs(mixed_spec)[0:1, 0:500, 0:-1][:, list(y_log), :]  # Shape is [1, 2049, T]
            mixed_spec_phase = torch.angle(mixed_spec)[0:1, 0:500, 0:-1][:, list(y_log), :] # Shape is [1, 2049, T]
            
            
            if mixed_spec_mag.shape[-1] < 33:
                if self.pad:
                    mixed_spec_mag = torch.nn.functional.pad(mixed_spec_mag, (0, 32 - mixed_spec_mag.shape[-1]%32),  'constant')
                    mixed_spec_phase = torch.nn.functional.pad(mixed_spec_phase, (0, 32 - mixed_spec_phase.shape[-1]%32),  'constant')
                    
            maxVal = torch.max(mixed_spec_mag)
            
            trainMixed.append(mixed_spec_mag/maxVal)
            trainPhase.append(mixed_spec_phase)

            wavs.append(mixed_wav)
            names.append("noisy_" + file_name_noisy)

        return trainMixed, trainPhase, wavs, names
    
    def to_wav_torch(self, spec, phase, audio_len = None):
        spec = torch.exp(spec)
        matrix = spec * torch.exp(1.j * phase) #output_len = librosa.frames_to_samples(in_frames, SE_Config.L_HOP, SE_Config.L_FRAME)
        if audio_len is not None:
            wav = torch.istft(matrix, n_fft = SE_Config.L_FRAME, hop_length= SE_Config.L_HOP, length = audio_len)
        else:
            wav = torch.istft(matrix, n_fft = SE_Config.L_FRAME, hop_length= SE_Config.L_HOP)
        return wav
    
    def __resample_if_necessary(self, audio, sr):
        if sr != SE_Config.SR:
            resampler = torchaudio.transforms.Resampler(sr, SE_Config.SR)
            audio_resampled = resampler(audio) 
        else:
            audio_resampled = audio
        return audio_resampled
    
    def _filter_files(self, list_of_wav_file_paths):
        """
        Removes the files from the list that have frames less than self.input_frame_size
        
        Inputs 
        List of files containing the wav paths
        
        Output:
        List of files containing the names of the files that have frames more than self.input_frame_size
        """
        filtered_files_list = []
        for files in list_of_wav_file_paths:
            wav, sr_wav = torchaudio.load(files)    # Shape is [1, N]
            wav = self.__resample_if_necessary(wav, sr_wav)
            spec = torch.stft(wav,
                              hop_length= SE_Config.L_HOP, 
                              n_fft= SE_Config.L_FRAME, 
                              window= self.window, 
                              return_complex= True) # Shape is [1, 2049, T]
            if spec.shape[-1] > self.input_frame_size:
                filtered_files_list.append(files)
        
        return filtered_files_list.sort()

    def collate_batch_coherence_net_test(self, batch):
        """
        Inputs:- 
        List containing all items from _get_item in the dataset of len batch size
        j = noise or clean aka indicator variable
        Outputs:- 

        """
        input_spec = [] # list of all the input specs - len = 2*batch_size
        phase_mixed = []
        raw_wavs_input = []
        speaker_ids = []

        for data in batch:
            trainMixed, trainPhase, wavs, names = data
            split_samples = trainMixed[0].shape[-1]//self.input_frame_size  + 1 #the number of samples after spliting

            for start in range(0, split_samples): #len(trainMixed)
                if start*self.input_frame_size + self.input_frame_size > trainMixed[0].shape[-1]:
                    trainMixed_padded = torch.nn.functional.pad(trainMixed[0], (0, start*self.input_frame_size + self.input_frame_size - trainMixed[0].shape[-1] + 1),  'constant')
                    input_spec.append(trainMixed_padded[:, :, start*self.input_frame_size : start*self.input_frame_size + self.input_frame_size])
                    #phase_mixed.append(trainPhase[0][:, :, start*self.input_frame_size : -1])
                else:
                    input_spec.append(trainMixed[0][:, :, start*self.input_frame_size: start*self.input_frame_size + self.input_frame_size])
                    #phase_mixed.append(trainPhase[0][:, :, start: start + self.input_frame_size])

            raw_wavs_input.append(wavs[0])
            speaker_ids.append(names[0])
            phase_mixed.append(trainPhase[0])
            
        return  input_spec, phase_mixed, raw_wavs_input, speaker_ids

def get_data(clean_files, noisy_files, eval = False, pad = False):
    #load data
    trainMixed = []
    trainSource = []
    trainNum = 0
    trainLab = []
    trainPhase = []
    clean_wav = []
    noisy_wav = []
    name = []
    
    for clean_file, noisy_file in tqdm(zip(clean_files, noisy_files), total = len(clean_files), position=0, leave=True):

        # load the audio file
        mixed_wav = librosa.load(noisy_file, sr=SE_Config.SR, mono=True)[0]
        vocals_wav = librosa.load(clean_file, sr=SE_Config.SR, mono=True)[0]
        noise_wav = mixed_wav - vocals_wav
        clean_wav.append(vocals_wav)
        noisy_wav.append(noise_wav)

        # convert audio to spectrogram
        mixed_spec = to_spec(mixed_wav)
        mixed_spec_mag = np.abs(mixed_spec)
        mixed_spec_phase = np.angle(mixed_spec)
        
        file_name_clean = clean_file.split("/")[-1].split(".")[0]
        file_name_noisy = noisy_file.split("/")[-1].split(".")[0]

        if mixed_spec_mag.shape[-1] < 65:
            if not pad:
                continue
            if pad:
                mixed_spec_mag = np.pad(mixed_spec_mag, ((0, 0), (0, 64 - mixed_spec_mag.shape[-1]%64)), 'constant')
                mixed_spec_phase = np.pad(mixed_spec_phase, ((0, 0), (0, 64 - mixed_spec_phase.shape[-1]%64)), 'constant')

        vocals_spec = to_spec(vocals_wav)
        vocals_spec_mag = np.abs(vocals_spec)

        noise_spec = to_spec(noise_wav)
        noise_spec_mag = np.abs(noise_spec)

        maxVal = np.max(mixed_spec_mag)

        # normalise and save all the labels
        ## will have to update here to put up as many channels as we want
        for j in range(2):
                trainMixed.append(mixed_spec_mag / maxVal)
                trainPhase.append(mixed_spec_phase)
        trainSource.append(noise_spec_mag / maxVal)
        trainLab.append([1,0])
        name.append(file_name_noisy)
        
        trainSource.append(vocals_spec_mag / maxVal)
        trainLab.append([0,1])
        name.append(file_name_clean)
        trainNum = trainNum+2

    print('Number of training examples : {}'.format(trainNum))

    if not eval:
        return trainMixed, trainSource, trainNum, trainLab
    if eval:
        return trainMixed, trainSource, trainNum, trainLab, trainPhase, clean_wav, noisy_wav, name

def get_data_test(noisy_files):
    #load data
    trainMixed = []
    trainLab = []
    print('generate train spectrograms')
    for noisy_file in tqdm(noisy_files):

        # load the audio file
        mixed_wav = librosa.load(noisy_file, sr=SE_Config.SR, mono=True)[0]

        # convert audio to spectrogram
        mixed_spec = to_spec(mixed_wav)
        mixed_spec_mag = np.abs(mixed_spec)
        maxVal = np.max(mixed_spec_mag)

        # normalise and save all the labels
        for j in range(2):
                trainMixed.append(mixed_spec_mag / maxVal)
        trainLab.append([1,0])
        trainLab.append([0,1])

    return trainMixed, trainLab


import librosa
import torch

def pitch_shift_librosa(audio_tensor, sample_rate):
    """
    Pitch shifts the input audio tensor by 12 semitones (one octave up) using librosa,
    ensuring the output tensor has the same shape as the input.

    Args:
    audio_tensor (Tensor): The input audio tensor in PyTorch format.
    sample_rate (int): The sampling rate of the audio tensor.

    Returns:
    Tensor: The pitch-shifted audio tensor in PyTorch format, with the same shape as input.
    """
    # Convert PyTorch tensor to NumPy array for processing with librosa
    audio_np = audio_tensor.numpy()

    # Prepare an empty array to hold the pitch-shifted data
    shifted_channels = []

    # Process each channel separately if the audio is not mono
    for i in range(audio_np.shape[0]):  # iterating over the first dimension assuming it is the channel dimension
        # Apply pitch shifting
        shifted_channel = librosa.effects.pitch_shift(audio_np[i], sr=sample_rate, n_steps=12)
        shifted_channels.append(shifted_channel)

    # Stack the processed channels back together along the first axis
    y_shifted_stacked = np.stack(shifted_channels, axis=0)

    # Convert the NumPy array back to PyTorch tensor
    shifted_tensor = torch.from_numpy(y_shifted_stacked)

    return shifted_tensor

if __name__ == "__main__":
    import torch
    dataset = Edinb_SE_Dataset("/home/karan/sda_link/datasets/Ferret_data/Dataset_2/embeddings/wavs/male_alone", 
                        "/home/karan/sda_link/datasets/Ferret_data/Dataset_2/embeddings/wavs/male_alone", 
                        pad = True)
    train_loader = torch.utils.data.DataLoader(dataset= dataset,
                                                batch_size= 1,
                                                shuffle=True,
                                                collate_fn = dataset.collate_batch_coherence_net,
                                                pin_memory=True,
                                                num_workers=1)

    for i, data in enumerate(train_loader):
        print(f"Batch {i+1}:")
        input_spec, output_spec, indicator, phase_mixed, raw_wavs_target, speaker_ids = data
        
        print(input_spec.size(), "The input spec size")
        print(output_spec.size(), "The output spec size")
        print(indicator.size(), "The indicator size")
        print(phase_mixed.size(), "The phase mixed size")
        print(raw_wavs_target.size(), "The raw wavs target size")
        print(speaker_ids.size(), "The speaker ids size")
        print(phase_mixed.size(), "The phase mixed size")