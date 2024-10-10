import os, argparse
import random
import numpy as np
import torch
from tqdm import tqdm
import wandb
from dataset import Edinb_SE_TestDataset  #, collate_batch_coherence_net
from model.coherence_net.model import CoherenceNet
from model.coherence_net.loss import CoherenceNetLoss
from utils.checkpoint_saver import CheckpointSaver
import librosa
import matplotlib.pyplot as plt
from utils.hparams import HParam
from datetime import datetime
from utils.compute_metrics import compute_metrics
import torchaudio
import soundfile as sf

class Tester():
    def __init__(self, credentials, wavs_dir, config_path, project_name, gpu_number = 1, save_path = None, save_audio = True, speaker_name_file = None, ft_checkpoint = None, chkpt_pt = None):
        
        torch.backends.cudnn.deterministic = True
        random.seed(hash("setting random seeds") % 2**16 - 1)
        np.random.seed(hash("improves reproducibility") % 2**32 - 1)
        torch.manual_seed(3407)
        torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

        # Device configuration
        self.device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
        self.config = HParam(config_path)
        
        self.wavs_dir = wavs_dir
        self.save_audio = save_audio
        self.save_path = save_path
        self.chkpt_pt = chkpt_pt

        os.makedirs(self.save_path, exist_ok=True)
       
        if speaker_name_file is None:
            self.output_dir = self.config.Coherence_Net_Config.save_path + "/wavs/" #
        else:
            spk_name = speaker_name_file.split("/")[-1].split("_")[0]
            self.output_dir = "/".join(self.wavs_dir.split("/")[0:-1]) + f"/output_{project_name}/{spk_name}/"
        
        os.makedirs(self.output_dir, exist_ok=True)
            
        wandb.login(key = credentials)
        
        # Make the data
        if speaker_name_file is None:
            self.dataset = Edinb_SE_TestDataset(wavs_dir, pad = True)
            
        if speaker_name_file is not None:
            self.dataset = Edinb_SE_TestDataset(wavs_dir, pad = True, speaker_name_file = speaker_name_file)
        
        self.test_loader = torch.utils.data.DataLoader(dataset= self.dataset,
                                                  batch_size = 1,
                                                  shuffle=True,
                                                  collate_fn = self.dataset.collate_batch_coherence_net_test,
                                                  pin_memory=True, 
                                                  num_workers=1)
        # Make the model
        self.model = CoherenceNet(self.config.Coherence_Net_Config.layer1, self.config.Coherence_Net_Config.layer2).to(self.device)
        
        self.project_name = project_name
        self.ft_checkpoint = ft_checkpoint
        
    def load_model(self, path, model):
        checkpoint = torch.load(path, map_location= self.device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model
        
    def run_model_pipeline(self, layer = 0, stream = 1, speech = False):
        """
        Args:
        layer: int = The training layer (1, 2, ..)
        stream: int = the stream to be trained 
                0 - trains the stream integerator
                1, 2, 3 - trains the stream
        """
        if not speech:
            indicator = torch.Tensor([1., 0.]).view(-1, 2)
        if speech:
            indicator = torch.Tensor([0., 1.]).view(-1, 2)
            
        dt = datetime.now()    
        run_name = f'layer{layer}-stream{stream}_{dt.month}-{dt.day}-{self.config.Coherence_Net_Config.NAME}'
        with wandb.init(project=self.project_name, config=self.config, name = run_name):
            # make the model, load weights and get ready
            model, prev_layer = self.make_model(layer, stream)
            
            # and use them to train the model
            outputwav, fullpred_spec = self.run(model, layer, stream, prev_layer, indicator, speech)
           
        return outputwav, fullpred_spec
    
    def make_model(self, layer, stream):
        if layer == 1:
            prev_layer = None
            if stream == 1:
                model = self.model.lay1.stream1
                self.load_model(self.config.SE_Config.lay1_stream1_bestchkpt_path, model)
                
            if stream == 2:
                model = self.model.lay1.stream2
                self.load_model(self.chkpt_pt, model) #config.SE_Config.lay1_stream2_bestchkpt_path
                
            if stream == 3:
                model = self.model.lay1.stream3
                self.load_model(self.config.SE_Config.lay1_stream3_bestchkpt_path, model)
                
            if stream == 0:
                model = self.model.lay1
                if self.ft_checkpoint is None:
                    self.load_model(self.config.SE_Config.lay1_bestchkpt_path, model)
                else:
                    self.load_model(self.ft_checkpoint, model)
                    
                ## load all the weights for all the streams
                #self.load_model(self.config.SE_Config.lay1_stream1_bestchkpt_path, model.stream1)
                #self.load_model(self.config.SE_Config.lay1_stream2_bestchkpt_path, model.stream2)
                #self.load_model(self.config.SE_Config.lay1_stream3_bestchkpt_path, model.stream3)

                
        if layer == 2:
            prev_layer = self.model.lay1
            ## load all the weights for all the streams for prev lay
            self.load_model(self.config.SE_Config.lay1_stream1_bestchkpt_path, prev_layer.stream1)
            self.load_model(self.config.SE_Config.lay1_stream2_bestchkpt_path, prev_layer.stream2)
            self.load_model(self.config.SE_Config.lay1_stream3_bestchkpt_path, prev_layer.stream3)
            
            if stream == 1:         
                model = self.model.lay2.stream1
                self.load_model(self.config.SE_Config.lay2_stream1_bestchkpt_path, model)
                
            if stream == 2:
                model = self.model.lay2.stream2
                self.load_model(self.config.SE_Config.lay2_stream2_bestchkpt_path, model)
                
            if stream == 3:
                model = self.model.lay2.stream3
                self.load_model(self.config.SE_Config.lay2_stream3_bestchkpt_path, model)
                
            if stream == 0:
                model = self.model.lay2
                self.load_model(self.config.SE_Config.lay2_bestchkpt_path, model)
                 
                ## load all the weights for all the streams for prev streams
                #self.load_model(self.config.SE_Config.lay2_stream1_bestchkpt_path, model.stream1)
                #self.load_model(self.config.SE_Config.lay2_stream2_bestchkpt_path, model.stream2)
                #self.load_model(self.config.SE_Config.lay2_stream3_bestchkpt_path, model.stream3)
                
        return model, prev_layer
    
    def run(self, model, layer, stream, prev_layer = None, indicator = None, speech = False):
        
        # Run training and track with wandb
        pbar = tqdm(self.test_loader, desc = "Processing Files")
        self.layer = layer
        self.stream = stream
        self.output_dir = self.output_dir + f"layer{layer}-stream{stream}/"
        os.makedirs(self.output_dir, exist_ok=True)

        for input_spec, phase_mixed, raw_wavs_target, speaker_ids in pbar:
            outputwav, fullpred_spec = self.test_batch(input_spec, indicator, model, layer, stream, prev_layer, phase_mixed, speaker_ids)
            if self.save_audio:
                if speech:
                    print(f"Audio written in the {self.output_dir}")
                    sf.write(self.output_dir + f"speech_{speaker_ids[0]}.wav", outputwav, self.config.SE_Config.SR, 'PCM_16')
                else:
                    sf.write(self.output_dir + f"noise_{speaker_ids[0]}.wav", outputwav, self.config.SE_Config.SR, 'PCM_16')
            
        return outputwav, fullpred_spec
    
    def test_batch(self, input_spec_list, indicator, model, layer, stream, prev_layer, phase_mixed, speaker_ids, log_data = True):
        """
        Args:-
        input_spec: torch.Tensor = The input mel spec to the model 
        output_spec:  torch.Tensor = The target mel spec for the model
        indicator: torch.Tensor = The input indicator var to the model
        model: torch.nn.Module = The model to be trained
        optimizer: torch.nn.Module = optimizer for the model to train
        criterion: class = the loss class
        
        Returns:- 
        loss: float = the loss value returned
        """
        lossacc = 0
        out_spec = []
        before = []
        after = [] 
        for input_spec in input_spec_list:
            # Forward pass ➡
            with torch.no_grad():

                input_spec, indicator = input_spec.unsqueeze(0).to(self.device), indicator[0].unsqueeze(0).to(self.device)

                print(input_spec.shape, indicator.shape)

                # Forward pass ➡
                if layer == 1:

                    if stream != 0:
                        final_predictions, before_attn, after_attn = model(input_spec, indicator)
                        out_spec.append(final_predictions)
                        before.append(before_attn)
                        after.append(after_attn)

                    if stream == 0: # 0 is the integrator mode
                        final_predictions_list, embedding_list, memory = model(input_spec, indicator)
                        pred_stream1, pred_stream2, pred_stream3, pred_layer1comb = final_predictions_list
                        embedding_stream1, embedding_stream2, embedding_stream3, embedding_layer1comb = embedding_list
                        out_spec.append(pred_layer1comb)
                        
                if layer == 2:

                    if stream != 0:
                        predictions_lay1, embeddings_lay1 = prev_layer(input_spec, indicator)        
                        pred_stream1_lay1, pred_stream2_lay1, pred_stream3_lay1, pred_layer1comb_lay1 = predictions_lay1
                        embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1, embedding_layer1comb_lay1 = embeddings_lay1

                        x_in_lay2 = torch.concat([embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1], axis = 1) # concat on embed dim torch.Size([4, 474, 2049, 64])
                        final_predictions, embedding = model(x_in_lay2, indicator)

                        out_spec.append(final_predictions)                        

                    if stream == 0: # 0 is the integrator mode
                        predictions_lay1, embeddings_lay1 = prev_layer(input_spec, indicator)        
                        pred_stream1_lay1, pred_stream2_lay1, pred_stream3_lay1, pred_layer1comb_lay1 = predictions_lay1
                        embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1, embedding_layer1comb_lay1 = embeddings_lay1

                        x_in_lay2 = torch.concat([embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1], axis = 1) # concat on embed dim torch.Size([4, 474, 2049, 64])
                        final_predictions_list, embedding_list = model(x_in_lay2, indicator)

                        pred_stream1, pred_stream2, pred_stream3, pred_layer2comb = final_predictions_list
                        embedding_stream1, embedding_stream2, embedding_stream3, embedding_layer2comb = embedding_list

                        out_spec.append(pred_layer2comb)
                        
        fullpred_spec = torch.concat(out_spec, dim = 3)
        fullpred_spec = fullpred_spec.squeeze(0)
        #print(f"%%%%%%%%%%%%%%%%%%%%%%%%{len(out_spec)}-{out_spec[0].shape}%%%%%%%%%%%%%%%%%%%%%%%%%")
        pred_audio_list = []
        start = 0

        embed_before_attn = torch.concat(before, dim = -1)
        embed_after_attn = torch.concat(after, dim = -1)
        torch.save(embed_before_attn, self.save_path + f"/embed_before_attn_{speaker_ids[0]}.pt" )
        torch.save(embed_after_attn, self.save_path + f"/embed_after_attn_{speaker_ids[0]}.pt" )

        """for spec in out_spec:
            spec = spec.squeeze(0)
            if start + 64 < phase_mixed[0].shape[-1]:
                pred_audio_list.append(self.dataset.to_wav_torch(spec, phase_mixed[0][:, :, start: start + 64]).squeeze(0))
            else:
                last_size = phase_mixed[0][:, :, start:].shape[-1]
                pred_audio_list.append(self.dataset.to_wav_torch(spec[:, :, 0: last_size], phase_mixed[0][:, :, start:]).squeeze(0).to(self.device))
            start += 64  
        """        
        #pred_audio = np.concatenate([y_out.detach().numpy() for y_out in pred_audio_list])

        pred_audio = np.zeros((400, 1)) #self.dataset.to_wav_torch(fullpred_spec[:, :, 0:phase_mixed[0].shape[-1]], phase_mixed[0].to(self.device))
        pred_audio = np.zeros((400, 1)) #pred_audio.squeeze(0).detach().cpu().numpy()
        
        if log_data:
            self.log_audio(pred_audio/abs(max(pred_audio)), speaker_ids[0])
            self.log_spectrogram(fullpred_spec[:, :, 0:phase_mixed[0].shape[-1]].squeeze(0).detach().cpu(), torch.concat(input_spec_list, dim = 2)[:, :, 0:phase_mixed[0].shape[-1]].squeeze(0).detach().cpu(), speaker_ids[0])
            
        return pred_audio/abs(max(pred_audio)), (fullpred_spec, torch.concat(input_spec_list, dim = 2))  
    
    def log_scalar(self, name, value, step):
        wandb.log({"step": step, f"{name}": value}, step = step)
        return
    
    def log_image(self, name, array):
        images = wandb.Image(array)  
        wandb.log({f"Mel Specs {name}| Input | Output |" : images})
        return
    
    def log_spectrogram(self, array, gt_array, name):
        fig, axs = plt.subplots(1, 3, figsize=(14,10))
        
        axs[0].imshow(gt_array, origin="lower", aspect="auto") #librosa.amplitude_to_db()
        plt.title('Input')
        axs[1].imshow(array, origin="lower", aspect="auto") #librosa.amplitude_to_db()
        plt.title('Output')
        axs[2].imshow(gt_array - torch.exp(array), origin="lower", aspect="auto") #librosa.amplitude_to_db()
        plt.title('Difference')
        
        self.log_image(name, fig)
        
        return
        
    def log_audio(self, audio, name):
        wandb.log({f"Output {name}": wandb.Audio(audio, sample_rate= self.config.SE_Config.SR)})
        return

if __name__ == "__main__":
    wandb_api_key = "1944427480bbbce6b8a41f2b440e92882e197578"
    
    p = argparse.ArgumentParser()
    
    p.add_argument('--input_dir', 
                   required = False, 
                   type = str, 
                   help = "The input directory containing all the files to denoise", 
                   default = "/home/kthakka2/data-mounya/karan/datasets/se_dataset/noisy_testset_wav_16k/")
    p.add_argument('--name', 
                   required = False, 
                   type = str, 
                   help = "The name of the experiment", 
                  default = "Ashwin_SE_baseline_28")
    p.add_argument('--config', 
                   required = False, 
                   type = str, 
                   default = "config-128.yaml", 
                   help = "The yaml file with the config params")
    
    args = p.parse_args()
    
    test_class = Tester(wandb_api_key, args.input_dir, args.config, args.name, 0, save_audio = True)
    audio, spec = test_class.run_model_pipeline(1, 1)
