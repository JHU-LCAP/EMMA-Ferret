
'''
Part of the code modified from
https://github.com/andabi/music-source-separation
'''

import librosa
import numpy as np
from config import SE_Config
import soundfile as sf
from mir_eval.separation import bss_eval_sources

def to_spec_torchaudio(audio, config):
    return



def get_chunks(spec, dur = 64, height = 2048):
    chunks = []
    h, w = spec.shape
    if w//dur != 0:
        spec = np.pad(spec, ((0, 0), (0, dur - w%dur)), 'constant')
    h, w_new = spec.shape
    for i in range(0, w_new, dur):
        chunks.append(spec[0:height, i: i + dur])
    return chunks

def calculate_median_sdr(gt, pred, vocals = True):
    sdrs = []
    if vocals:
        for i in range(0, len(gt)):
            sdr = bss_eval_sdr(gt[i], pred[i])
            sdrs.append(sdr)
    if not vocals:
        for i in range(0, len(gt)):
            sdr = bss_eval_sdr(gt[i], pred[i])
            sdrs.append(sdr)
    median_sdr = np.median(sdrs)
    return median_sdr

def get_wav(filename, sr=SE_Config.SR):
    src1_src2 = librosa.load(filename, sr=sr, mono=False)[0]
    mixed = librosa.to_mono(src1_src2)
    src1, src2 = src1_src2[0, :], src1_src2[1, :]
    return mixed, src1, src2

def to_wav_file(mag, phase, len_hop=SE_Config.L_HOP):
    stft_maxrix = get_stft_matrix(mag, phase)
    return np.array(librosa.istft(stft_maxrix, hop_length=len_hop))

def to_spec(wav, len_frame=SE_Config.L_FRAME, len_hop=SE_Config.L_HOP):
    return librosa.stft(wav, n_fft=len_frame, hop_length=len_hop)

def get_stft_matrix(magnitudes, phases):
    return magnitudes * np.exp(1.j * phases)

def write_wav(data, path, sr=SE_Config.SR, format='wav', subtype='PCM_16'):
    sf.write(path, data, sr, format=format, subtype=subtype)

def bss_eval(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len = pred_src1_wav.shape[0]
    src1_wav = src1_wav[:len]
    src2_wav = src2_wav[:len]
    mixed_wav = mixed_wav[:len]
    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), compute_permutation=True)
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                          np.array([mixed_wav, mixed_wav]), compute_permutation=True)
    # sdr, sir, sar, _ = bss_eval_sources(src2_wav,pred_src2_wav, False)
    # sdr_mixed, _, _, _ = bss_eval_sources(src2_wav,mixed_wav, False)
    nsdr = sdr - sdr_mixed
    return nsdr, sdr, sir, sar, len

def bss_eval_sdr(src1_wav, pred_src1_wav):
        len_cropped = pred_src1_wav.shape[0]
        src1_wav = src1_wav[:len_cropped]

        sdr,sir, sar, _ = bss_eval_sources(src1_wav,
                                            pred_src1_wav, compute_permutation=True)
        return sdr


def bss_eval_all(src1_wav, pred_src1_wav):
        len_cropped = pred_src1_wav.shape[0]
        src1_wav = src1_wav[:len_cropped]

        sdr,sir, sar, _ = bss_eval_sources(src1_wav,
                                            pred_src1_wav, compute_permutation=True)
        return sdr,sir,sar
