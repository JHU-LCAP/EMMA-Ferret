from dataset import gen_log_space
from tqdm import tqdm
from dataset import Edinb_SE_Dataset, make_loader
import matplotlib.pyplot as plt
from utils.hparams import HParam
from model.coherence_net.model import CoherenceNet
import torch
import numpy as np

def apply_selection(conv_wt_pt, embedding):
    out_embedding = torch.zeros(embedding.shape)
    weight = conv_wt_pt #torch.load(conv_wt_pt)
    for i in range(0, 512):
        k = weight[:, i:i+1, :, :]
        e = embedding[:, i:i+1, :, :]
        out_embedding[:, i:i+1, :, :] = torch.nn.functional.conv2d(e, k, padding="same")
    return out_embedding

config = HParam("config-128.yaml")
model = CoherenceNet(config.Coherence_Net_Config.layer1, config.Coherence_Net_Config.layer2)
model = model.lay1.stream2
checkpoint = torch.load(config.SE_Config.lay1_stream2_bestchkpt_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

# Adjust these paths and parameters
clean_files_path = "/home/karan/sda_link/datasets/Ferret_data/Dataset_2/embeddings/wavs/female_alone_random/"
noisy_files_path = "/home/karan/sda_link/datasets/Ferret_data/Dataset_2/embeddings/wavs/mix_random/"
batch_size = 1
model_type = "coherence_net"
save_path = "/home/karan/sda_link/datasets/Ferret_data/Dataset_2/embeddings/mix_in_male_attention_random_conv/"
female_attn = False
u = 0 

log_space = gen_log_space(500, 128)

model = model.to("cuda")

import numpy as np

# Instantiate the dataset
dataset = Edinb_SE_Dataset(clean_files_path, noisy_files_path, pad = True)
                    
print(f"Dataset Length: {len(dataset)}")

# Instantiate the DataLoader
loader = torch.utils.data.DataLoader(dataset= dataset,
                                            batch_size= 1,
                                            shuffle=True,
                                            collate_fn = dataset.collate_batch_coherence_net,
                                            pin_memory=True,
                                            num_workers=16)

print("loader complete")

# Iterate over the DataLoader
for (input_spec, output_spec, indicator, phase_mixed, raw_wavs_target, speaker_ids) in tqdm(loader):
    # For now, just print the shape of the loaded data
    #print(speaker_ids[0][6:])

    if female_attn:
        input_spec = input_spec[1:2]
        indicator = indicator[1:2]
        output_spec = output_spec[1:2]
    else:
        input_spec = input_spec[0:1]
        indicator = indicator[0:1]
        output_spec = output_spec[0:1]

    #input_spec = torch.ones_like(input_spec)

    # ouput from the model
    model_pred, model_pred_before, model_pred_after = model(input_spec.to("cuda"), indicator.to("cuda"))

    model_pred_after = apply_selection(model.conv_final[0].weight, model_pred_after)

    model_pred = model_pred.to("cpu")
    model_pred_before = model_pred_before.to("cpu")
    model_pred_after = model_pred_after.to("cpu")

    torch.save(model_pred_before, save_path + "before_" + speaker_ids[0][6:] + ".pt")
    torch.save(model_pred_after, save_path + "after_" + speaker_ids[0][6:] + ".pt")
    
    if u == 0:
        print(input_spec.shape, "The shape of the input spec")
        print(indicator.shape, "The shape of the indicator")
        print(indicator)

        plt.figure(figsize=(20, 10))
        
        print(speaker_ids)

        time_axis = np.arange(0, 64) * 256/16000
        # Visualizing the spectrogram of the first item in the batch
        plt.subplot(2, 3, 1)
        plt.imshow(input_spec[0, 0].numpy(), aspect='auto', origin='lower', cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks([i for i in range(0, 40, 10)], [i*(512/16000) for i in range(0, 40, 10)])
        plt.yticks([i for i in range(0, 128, 20)], [round(log_space[i]*(8000/2048)/1000, 2) for i in range(0, 128, 20)])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (KHz)")
        plt.title("Input Spectrogram")
        
        plt.subplot(2, 3, 2)
        plt.imshow(model_pred[0, 0].detach().numpy(), aspect='auto', origin='lower', cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks([i for i in range(0, 40, 10)], [i*(512/16000) for i in range(0, 40, 10)])
        plt.yticks([i for i in range(0, 128, 20)], [round(log_space[i]*(8000/2048)/1000, 2) for i in range(0, 128, 20)])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency KHz)")
        plt.title("Predicted Spectrogram")
        """
        plt.subplot(2, 3, 3)
        plt.imshow(input_spec[0, 0].numpy(), aspect='auto', origin='lower', cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks([i for i in range(0, 40, 10)], [i*(512/16000) for i in range(0, 40, 10)])
        plt.yticks([i for i in range(0, 128, 20)], [round(log_space[i]*(8000/2048)/1000, 2) for i in range(0, 128, 20)])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (KHz)")
        plt.title("Ground Truth Spectrogram")
        """
        plt.subplot(2, 3, 3)
        plt.imshow(output_spec[0, 0].numpy(), aspect='auto', origin='lower', cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks([i for i in range(0, 40, 10)], [i*(512/16000) for i in range(0, 40, 10)])
        plt.yticks([i for i in range(0, 128, 20)], [round(log_space[i]*(8000/2048)/1000, 2) for i in range(0, 128, 20)])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (KHz)")
        plt.title("Ground Truth Spectrogram")
        

        plt.savefig(save_path + "spectrogram_" + speaker_ids[0][6:] + ".png")

    u += 1