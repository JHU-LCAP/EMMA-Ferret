import os
import random
import numpy as np
import torch
from tqdm import tqdm
import wandb
from dataset import Edinb_SE_Dataset  #, collate_batch_coherence_net
from model.coherence_net.model import CoherenceNet
from model.coherence_net.loss import CoherenceNetLoss
from utils.checkpoint_saver import CheckpointSaver
import librosa
import matplotlib.pyplot as plt
from utils.hparams import HParam, get_model_size
#from utils.compute_metrics import compute_metrics
from accelerate import Accelerator
from scipy.io.wavfile import write

class Trainer():
    def __init__(self, credentials, config_path, project_name, gpu_number = 1, multigpu = False, restart_chpt = None):

        torch.backends.cudnn.deterministic = True
        random.seed(hash("setting random seeds") % 2**32 - 1)
        np.random.seed(hash("improves reproducibility") % 2**32 - 1)
        torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
        torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


        self.restart_chpt = restart_chpt

        self.log_file_path = "logs.txt"

        # Device configuration
        self.config = HParam(config_path)

        if multigpu:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
        else:
            self.device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")


        wandb.login(key = credentials)

        # Make the data
        self.dataset = Edinb_SE_Dataset(self.config.SE_Config.clean_files_dir, self.config.SE_Config.noisy_files_dir, pad = True)

        

        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.dataset,
                                                                    [len(self.dataset) - 1000,
                                                                    1000],
                                                                    generator=torch.Generator().manual_seed(42))

        self.train_loader = torch.utils.data.DataLoader(dataset= self.train_dataset,
                                                   batch_size=self.config.SE_Config.batchSize,
                                                   shuffle=True,
                                                   collate_fn = self.dataset.collate_batch_coherence_net,
                                                   pin_memory=True,
                                                   num_workers=16)

        self.valid_loader = torch.utils.data.DataLoader(dataset= self.valid_dataset,
                                                  batch_size=self.config.SE_Config.batchSize,
                                                  shuffle=True,
                                                  collate_fn = self.dataset.collate_batch_coherence_net,
                                                  pin_memory=True,
                                                  num_workers=16)

        self.testset = Edinb_SE_Dataset(self.config.SE_Config.clean_files_eval_dir, self.config.SE_Config.noisy_files_eval_dir, pad = True)

        self.test_loader = torch.utils.data.DataLoader(dataset= self.testset,
                                                  batch_size = 1,
                                                  shuffle=True,
                                                  collate_fn = self.dataset.collate_batch_coherence_net_eval,
                                                  pin_memory=True,
                                                  num_workers=16)


        # Make the model
        self.model = CoherenceNet(self.config.Coherence_Net_Config.layer1, self.config.Coherence_Net_Config.layer2)
        ## remove the grads required for the whole model
        self.change_grad(self.model, remove_grad = True)
        #if multigpu is not None:
        #    self.model = torch.nn.DataParallel(self.model, device_ids= multigpu)

        self.multigpu = multigpu
        # Make the loss and optimizer
        self.criterion = CoherenceNetLoss()

        self.project_name = project_name

        ## define the saver function
        self.save_model = CheckpointSaver(dirpath = self.config.Coherence_Net_Config.checkpoint_path, model_name = self.config.Coherence_Net_Config.NAME, decreasing=True, top_n = 2)

        ## the steps for display and
        self.displayloss = 10 # display loss every kth iteration
        self.save_iter = 1000 # save model every kth iteration
        self.test_iter = 40000000000000 # evaluate the model every kth iteration
        self.log_audio_metrics = False

        print("Model Initialized:")
        print(f"Size of the Model in MB is {get_model_size(self.model)}")

    def load_model(self, path, model):
        checkpoint = torch.load(path)
        chkpt = {}
        if self.multigpu:
            for key in checkpoint.keys():
                chkpt[key[7:]] = checkpoint[key]
            #print(chkpt.keys())
        else:
            chkpt = checkpoint

        model.load_state_dict(chkpt)
        return model

    def change_grad(self, model, remove_grad = True):
        for name, param in model.named_parameters():
            if remove_grad:
                param.requires_grad = False
            if not remove_grad:
                param.requires_grad = True
            #print(name, param.requires_grad)
        return

    def run_model_pipeline(self, layer = 0, stream = 1):
        """
        Args:
        layer: int = The training layer (1, 2, ..)
        stream: int = the stream to be trained
                0 - trains the stream integerator
                1, 2, 3 - trains the stream
        """
        dt = self.save_model.get_time_data()
        if not self.multigpu:
            run_name = f'layer{layer}-stream{stream}_{dt.month}-{dt.day}'
        else:
            run_name = f"{self.accelerator.process_index}"
        with wandb.init(group = f"{self.project_name}-{run_name}", config=self.config, name = run_name):
            # make the model, data, and optimization modules
            model, criterion, optimizer, epochs, prev_layer = self.make_model(layer, stream)

            # and use them to train the model
            self.train(model, criterion, optimizer, epochs, layer, stream, prev_layer)

            # and test its final performance
            #self.test(model, valid_loader)
        return

    def make_model(self, layer, stream):
        if layer == 1:
            prev_layer = None
            if stream == 1:
                model = self.model.lay1.stream1
                self.change_grad(model, remove_grad = False)
                optimizer = torch.optim.Adam(model.parameters(), lr = self.config.Coherence_Net_Config.LR)
                criterion = self.criterion.stream_loss
                epochs = self.config.Coherence_Net_Config.stream1_steps

            if stream == 2:
                model = self.model.lay1.stream2
                self.change_grad(model, remove_grad = False)
                optimizer = torch.optim.Adam(model.parameters(), lr = self.config.Coherence_Net_Config.LR)
                criterion = self.criterion.stream_loss
                epochs = self.config.Coherence_Net_Config.stream2_steps

            if stream == 3:
                model = self.model.lay1.stream3
                self.change_grad(model, remove_grad = False)
                optimizer = torch.optim.Adam(model.parameters(), lr = self.config.Coherence_Net_Config.LR)
                criterion = self.criterion.stream_loss
                epochs = self.config.Coherence_Net_Config.stream3_steps

            if stream == 0:
                model = self.model.lay1
                self.change_grad(model.stream_integrator, remove_grad = False)
                optimizer = torch.optim.Adam(model.stream_integrator.parameters(), lr = self.config.Coherence_Net_Config.LR)
                criterion = self.criterion.stream_loss
                epochs = self.config.Coherence_Net_Config.stream_integerator_steps

                ## load all the weights for all the streams
                self.load_model(self.config.SE_Config.lay1_stream1_bestchkpt_path, model.stream1)
                self.load_model(self.config.SE_Config.lay1_stream2_bestchkpt_path, model.stream2)
                self.load_model(self.config.SE_Config.lay1_stream3_bestchkpt_path, model.stream3)

        if layer == 2:
            prev_layer = self.model.lay1
            ## load all the weights for all the streams for prev lay
            self.load_model(self.config.SE_Config.lay1_stream1_bestchkpt_path, prev_layer.stream1)
            self.load_model(self.config.SE_Config.lay1_stream2_bestchkpt_path, prev_layer.stream2)
            self.load_model(self.config.SE_Config.lay1_stream3_bestchkpt_path, prev_layer.stream3)

            if stream == 1:
                model = self.model.lay2.stream1
                self.change_grad(model, remove_grad = False)
                optimizer = torch.optim.Adam(model.parameters(), lr = self.config.Coherence_Net_Config.LR)
                criterion = self.criterion.stream_loss
                epochs = self.config.Coherence_Net_Config.stream1_steps
            if stream == 2:
                model = self.model.lay2.stream2
                self.change_grad(model, remove_grad = False)
                optimizer = torch.optim.Adam(model.parameters(), lr = self.config.Coherence_Net_Config.LR)
                criterion = self.criterion.stream_loss
                epochs = self.config.Coherence_Net_Config.stream2_steps
            if stream == 3:
                model = self.model.lay2.stream3
                self.change_grad(model, remove_grad = False)
                optimizer = torch.optim.Adam(model.parameters(), lr = self.config.Coherence_Net_Config.LR)
                criterion = self.criterion.stream_loss
                epochs = self.config.Coherence_Net_Config.stream3_steps
            if stream == 0:
                model = self.model.lay2
                self.change_grad(model.stream_integrator, remove_grad = False)
                optimizer = torch.optim.Adam(model.stream_integrator.parameters(), lr = self.config.Coherence_Net_Config.LR)
                criterion = self.criterion.lay2_loss
                epochs = self.config.Coherence_Net_Config.stream_integerator_steps

                ## load all the weights for all the streams for prev streams
                self.load_model(self.config.SE_Config.lay2_stream1_bestchkpt_path, model.stream1)
                self.load_model(self.config.SE_Config.lay2_stream2_bestchkpt_path, model.stream2)
                self.load_model(self.config.SE_Config.lay2_stream3_bestchkpt_path, model.stream3)


        if self.restart_chpt is not None:
            print(f"%%%%%%%%%%%%%%%%%%%%%%% Loading weights from the chkpt {self.restart_chpt} and training %%%%%%%%%%%%%%%%%%%%")
            self.load_model(self.restart_chpt, model)

        if self.multigpu:
            if prev_layer is not None:
                model, prev_layer, optimizer, self.train_loader, self.test_loader, self.valid_loader = self.accelerator.prepare(model.to(self.device), prev_layer.to(self.device), optimizer, self.train_loader, self.test_loader, self.valid_loader)

            else:
                model, optimizer, self.train_loader, self.test_loader, self.valid_loader = self.accelerator.prepare(model.to(self.device), optimizer, self.train_loader, self.test_loader, self.valid_loader)

        else:
            if prev_layer is not None:
                model = model.to(self.device)
                prev_layer = prev_layer.to(self.device)
            else:
                model = model.to(self.device)

        return model, criterion, optimizer, epochs, prev_layer

    def train(self, model, criterion, optimizer, epochs, layer, stream, prev_layer = None):

        # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(model, criterion, log="all", log_freq=10)

        # Run training and track with wandb
        if self.restart_chpt is not None:
            batch_ct_iter = int(self.restart_chpt.split("/")[-1].split("_")[1][5:])
        else:
            batch_ct_iter = 0

        loss_acc = 0

        iterations = epochs*len(self.train_loader)

        for epoch in range(epochs):
            pbar = tqdm(self.train_loader, desc = "Loading train data for present epoch")

            for input_spec, output_spec, indicator, phase_mixed, raw_wavs_target, speaker_ids in pbar:
                model, loss = self.train_batch(input_spec, output_spec, indicator, model, optimizer, criterion, layer, stream, prev_layer)
                batch_ct_iter += 1
                loss_acc += loss.item()
                del loss

                # Report train metrics every 100th epoch
                if ((batch_ct_iter) % self.displayloss) == 0:
                    self.log_scalar("Loss/train", loss_acc/self.displayloss, batch_ct_iter)
                    log_message = f"Average Train Loss {loss_acc/self.displayloss:.8f} | step {batch_ct_iter} | Epoch {epoch}"
                    pbar.set_description(log_message)
                    with open(self.log_file_path, 'a') as log_file:
                        log_file.write(log_message + "\n")
                    loss_acc = 0

                # Validation and save model
                if ((batch_ct_iter) % self.save_iter) == 0:
                    vallossacc = 0
                    ct = 0
                    for input_spec, output_spec, indicator, phase_mixed, raw_wavs_target, speaker_ids in tqdm(self.valid_loader, desc = "Loading valid data for present epoch"):
                        if ct == 0:
                            valid_loss = self.valid_batch(input_spec, output_spec, indicator, model, criterion, layer, stream, prev_layer, batch_ct_iter, log_data = True)
                        else:
                            valid_loss = self.valid_batch(input_spec, output_spec, indicator, model, criterion, layer, stream, prev_layer, batch_ct_iter)
                        vallossacc += valid_loss
                        ct += 1

                    self.log_scalar("Loss_valid", vallossacc/ct, batch_ct_iter)
                    log_message = f"Average Validation Loss {vallossacc/ct:.8f} | step {batch_ct_iter} | Epoch {epoch}"
                    with open(self.log_file_path, 'a') as log_file:
                        log_file.write(log_message + "\n")
                    pbar.set_description(log_message)

                    self.save_model(model, batch_ct_iter, vallossacc/ct, layer, stream)

                # Validation and save model
                """
                if ((batch_ct_iter) % self.test_iter) == 0:
                    testlossacc = 0
                    tct = 0
                    for input_spec, output_spec, indicator, phase_mixed, raw_wavs_target, speaker_ids in self.test_loader:
                        if tct == 0:
                            test_loss, metrics = self.test_batch(input_spec, output_spec, indicator, model, criterion, layer, stream, prev_layer, batch_ct_iter, raw_wavs_target, phase_mixed, speaker_ids, log_data = True)
                            metrics_total = metrics
                        else:
                            test_loss, metrics = self.test_batch(input_spec, output_spec, indicator, model, criterion, layer, stream, prev_layer, batch_ct_iter, raw_wavs_target, phase_mixed, speaker_ids)
                        testlossacc += test_loss
                        tct += 1
                        if self.log_audio_metrics:
                            metrics_total += metrics
                    self.log_scalar("Loss_test", testlossacc/tct, batch_ct_iter)

                    if self.log_audio_metrics:
                        metrics_dict = dict(zip(["metrics/pesq_mos", "metrics/CSIG", "metrics/CBAK", "metrics/COVL", "metrics/segSNR", "metrics/STOI"], metrics_total/tct))
                        wandb.log(metrics_dict, batch_ct_iter)
                """
        return

    def train_batch(self, input_spec, target_spec, indicator, model, optimizer, criterion, layer, stream, prev_layer):
        """
        Args:-
        input_spec: torch.Tensor = The input mel spec to the model
        output_spec:  torch.Tensor = The target mel spec for the model
        indicator: torch.Tensor = The input indicator var to the model
        model: torch.nn.Module = The model to be trained
        optimizer: torch.nn.Module = optimizer for the model to train
        criterion: class = the loss class

        Returns:-
        model: torch.nn.Module = the model with updated weights
        loss: float = the loss value returned
        """
        #input_spec, target_spec, indicator = input_spec.to(self.device), output_spec.to(self.device), indicator.to(self.device)

        # Forward pass ➡
        if layer == 1:

            if stream != 0:
                final_predictions, embedding, memory = model(input_spec.to(self.device), indicator.to(self.device))
                loss = criterion(final_predictions, target_spec.to(self.device))
                del final_predictions

            if stream == 0: # 0 is the integrator mode
                final_predictions_list, embedding_list, memory = model(input_spec.to(self.device), indicator.to(self.device))
                pred_stream1, pred_stream2, pred_stream3, pred_layer1comb = final_predictions_list
                embedding_stream1, embedding_stream2, embedding_stream3, embedding_layer1comb = embedding_list
                loss = criterion(pred_layer1comb, target_spec.to(self.device))
                del pred_layer1comb, final_predictions_list, embedding_list

        if layer == 2:

            if stream != 0:
                predictions_lay1, embeddings_lay1, _  = prev_layer(input_spec.to(self.device), indicator.to(self.device))
                #pred_stream1_lay1, pred_stream2_lay1, pred_stream3_lay1, pred_layer1comb_lay1 = predictions_lay1
                embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1, embedding_layer1comb_lay1 = embeddings_lay1

                x_in_lay2 = torch.concat([embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1], axis = 1) # concat on embed dim torch.Size([4, 474, 2049, 64])
                final_predictions, embedding, memory = model(x_in_lay2, indicator.to(self.device))

                loss = criterion(final_predictions, target_spec.to(self.device))
                del final_predictions

            if stream == 0: # 0 is the integrator mode
                predictions_lay1, embeddings_lay1, _ = prev_layer(input_spec.to(self.device), indicator.to(self.device))
                #pred_stream1_lay1, pred_stream2_lay1, pred_stream3_lay1, pred_layer1comb_lay1 = predictions_lay1
                embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1, embedding_layer1comb_lay1 = embeddings_lay1

                x_in_lay2 = torch.concat([embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1], axis = 1) # concat on embed dim torch.Size([4, 474, 2049, 64])
                final_predictions_list, embedding_list, memory = model(x_in_lay2, indicator.to(self.device))

                pred_stream1, pred_stream2, pred_stream3, pred_layer1comb = final_predictions_list
                embedding_stream1, embedding_stream2, embedding_stream3, embedding_layer1comb = embedding_list

                loss = criterion(pred_layer1comb, target_spec.to(self.device))
                del pred_layer1comb, final_predictions_list, embedding_list
        # Backward pass ⬅
        optimizer.zero_grad()
        if self.multigpu:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        # Step with optimizer
        optimizer.step()

        return model, loss

    def valid_batch(self, input_spec, target_spec, indicator, model, criterion, layer, stream, prev_layer, step, log_data = False):
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
        # Forward pass ➡
        with torch.inference_mode():

            #input_spec, target_spec, indicator = input_spec.to(self.device), output_spec.to(self.device), indicator.to(self.device)

            # Forward pass ➡
            if layer == 1:

                if stream != 0:
                    final_predictions, embedding, memory = model(input_spec.to(self.device), indicator.to(self.device))
                    loss = criterion(final_predictions, target_spec.to(self.device))
                    if log_data:
                        self.log_spectrogram(final_predictions.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory.detach().cpu().numpy(),
                                             f"lay:{layer} | stream:{stream} | GT (left) | Predictions (right)",
                                             f"train/spectrograms/lay:{layer} | stream:{stream} | GT (left) | Predictions (right)",
                                             step)


                if stream == 0: # 0 is the integrator mode
                    final_predictions_list, embedding_list, memory_list = model(input_spec.to(self.device), indicator.to(self.device))
                    pred_stream1, pred_stream2, pred_stream3, pred_layer1comb = final_predictions_list
                    embedding_stream1, embedding_stream2, embedding_stream3, embedding_layer1comb = embedding_list
                    memory_stream1, memory_stream2, memory_stream3, memory_layer1comb = memory_list

                    loss = criterion(pred_layer1comb, target_spec.to(self.device))
                    if log_data:
                        self.log_spectrogram(pred_layer1comb.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_layer1comb.detach().cpu().numpy(),
                                             f"lay:{layer} | stream:{stream} | GT (left) | Predictions (right)",
                                             f"train/spectrograms/lay:{layer} | stream:{stream} | GT (left) | Predictions (right)",
                                             step)
                        self.log_spectrogram(pred_stream1.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_stream1.detach().cpu().numpy(),
                                             f"lay:{layer} | stream:1 | GT (left) | Predictions (right)",
                                             f"train/prev_layers_spectrograms/lay:{layer} | stream:1 | GT (left) | Predictions (right)",
                                             step)
                        self.log_spectrogram(pred_stream2.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_stream2.detach().cpu().numpy(),
                                             f"lay:{layer} | stream:2 | GT (left) | Predictions (right)",
                                             f"train/prev_layers_spectrograms/lay:{layer} | stream:2 | GT (left) | Predictions (right)",
                                             step)
                        self.log_spectrogram(pred_stream3.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_stream3.detach().cpu().numpy(),
                                             f"lay:{layer} | stream:3 | GT (left) | Predictions (right)",
                                             f"train/prev_layers_spectrograms/lay:{layer} | stream:3 | GT (left) | Predictions (right)",
                                             step)



            if layer == 2:

                if stream != 0:
                    predictions_lay1, embeddings_lay1, memory_list = prev_layer(input_spec.to(self.device), indicator.to(self.device))
                    pred_stream1_lay1, pred_stream2_lay1, pred_stream3_lay1, pred_layer1comb_lay1 = predictions_lay1
                    embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1, embedding_layer1comb_lay1 = embeddings_lay1

                    x_in_lay2 = torch.concat([embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1], axis = 1) # concat on embed dim torch.Size([4, 474, 2049, 64])
                    final_predictions, embedding, memory = model(x_in_lay2, indicator.to(self.device))

                    loss = criterion(final_predictions, target_spec.to(self.device))

                    if log_data:
                        self.log_spectrogram(final_predictions.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory.detach().cpu().numpy(),
                                             f"lay:{layer} | stream:{stream} | GT (left) | Predictions (right)",
                                             f"train/spectrograms/lay:{layer} | stream:{stream} | GT (left) | Predictions (right)",
                                             step)

                if stream == 0: # 0 is the integrator mode
                    predictions_lay1, embeddings_lay1, memory_list_lay1 = prev_layer(input_spec.to(self.device), indicator.to(self.device))
                    pred_stream1_lay1, pred_stream2_lay1, pred_stream3_lay1, pred_layer1comb_lay1 = predictions_lay1
                    embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1, embedding_layer1comb_lay1 = embeddings_lay1
                    memory_stream1_lay1, memory_stream2_lay1, memory_stream3_lay1, memory_layer1comb_lay1 = memory_list

                    x_in_lay2 = torch.concat([embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1], axis = 1) # concat on embed dim torch.Size([4, 474, 2049, 64])
                    final_predictions_list, embedding_list, memory_list = model(x_in_lay2, indicator.to(self.device))

                    pred_stream1, pred_stream2, pred_stream3, pred_layer2comb = final_predictions_list
                    embedding_stream1, embedding_stream2, embedding_stream3, embedding_layer2comb = embedding_list
                    memory_stream1, memory_stream2, memory_stream3, memory_layer2comb = memory_list

                    loss = criterion(pred_layer1comb_lay1, pred_layer2comb, target_spec.to(self.device))

                    if log_data:
                        # lay 1 logs

                        self.log_spectrogram(pred_layer1comb_lay1.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_layer1comb_lay1.detach().cpu().numpy(),
                                             f"lay:1 | stream:{stream} | GT (left) | Predictions (right)",
                                             f"train/prev_layers_spectrograms/lay:1 | stream:{stream} | GT (left) | Predictions (right)",
                                             step)
                        self.log_spectrogram(pred_stream1_lay1.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_stream1_lay1.detach().cpu().numpy(),
                                             f"lay:1 | stream:1 | GT (left) | Predictions (right)",
                                             f"train/prev_layers_spectrograms/lay:1 | stream:1 | GT (left) | Predictions (right)",
                                             step)
                        self.log_spectrogram(pred_stream2_lay1.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_stream2_lay1.detach().cpu().numpy(),
                                             f"lay:1 | stream:2 | GT (left) | Predictions (right)",
                                             f"train/prev_layers_spectrograms/lay:1 | stream:2 | GT (left) | Predictions (right)",
                                             step)
                        self.log_spectrogram(pred_stream3_lay1.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_stream3_lay1.detach().cpu().numpy(),
                                             f"lay:1 | stream:3 | GT (left) | Predictions (right)",
                                             f"train/prev_layers_spectrograms/lay:1 | stream:3 | GT (left) | Predictions (right)",
                                             step)

                        ## lay 2 logs

                        self.log_spectrogram(pred_layer2comb.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_layer2comb.detach().cpu().numpy(),
                                             f"lay:{layer} | stream:{stream} | GT (left) | Predictions (right)",
                                             f"train/spectrograms/lay:{layer} | stream:{stream} | GT (left) | Predictions (right)",
                                             step)
                        self.log_spectrogram(pred_stream1.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_stream1.detach().cpu().numpy(),
                                             f"lay:{layer} | stream:1 | GT (left) | Predictions (right)",
                                             f"train/prev_layers_spectrograms/lay:{layer} | stream:1 | GT (left) | Predictions (right)",
                                             step)
                        self.log_spectrogram(pred_stream2.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_stream2.detach().cpu().numpy(),
                                             f"lay:{layer} | stream:2 | GT (left) | Predictions (right)",
                                             f"train/prev_layers_spectrograms/lay:{layer} | stream:2 | GT (left) | Predictions (right)",
                                             step)
                        self.log_spectrogram(pred_stream3.detach().cpu().numpy(),
                                             target_spec.detach().cpu().numpy(),
                                             memory_stream3.detach().cpu().numpy(),
                                             f"lay:{layer} | stream:3 | GT (left) | Predictions (right)",
                                             f"train/prev_layers_spectrograms/lay:{layer} | stream:3 | GT (left) | Predictions (right)",
                                             step)

        return loss

    def test_batch(self, input_spec_list , output_spec_list, indicator, model, criterion, layer, stream, prev_layer, step, raw_wavs_target, phase_mixed, speaker_ids, log_data = False):
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
        for input_spec, target_spec in zip(input_spec_list, output_spec_list):
            # Forward pass ➡
            with torch.inference_mode():

                #input_spec, target_spec, indicator = input_spec.unsqueeze(0).to(self.device), target_spec.unsqueeze(0).to(self.device), indicator[0].unsqueeze(0).to(self.device)

                # Forward pass ➡
                if layer == 1:

                    if stream != 0:
                        final_predictions, embedding, memory = model(input_spec.unsqueeze(0).to(self.device), indicator[0].unsqueeze(0).to(self.device))
                        loss = criterion(final_predictions, target_spec.unsqueeze(0).to(self.device))

                        out_spec.append(final_predictions)
                        lossacc += loss

                    if stream == 0: # 0 is the integrator mode
                        final_predictions_list, embedding_list, memory_list = model(input_spec.unsqueeze(0).to(self.device), indicator[0].unsqueeze(0).to(self.device))
                        pred_stream1, pred_stream2, pred_stream3, pred_layer1comb = final_predictions_list
                        embedding_stream1, embedding_stream2, embedding_stream3, embedding_layer1comb = embedding_list
                        loss = criterion(pred_layer1comb, target_spec.unsqueeze(0).to(self.device))

                        out_spec.append(pred_layer1comb)
                        lossacc += loss


                if layer == 2:

                    if stream != 0:
                        predictions_lay1, embeddings_lay1, memory_list = prev_layer(input_spec.unsqueeze(0).to(self.device), indicator[0].unsqueeze(0).to(self.device))
                        pred_stream1_lay1, pred_stream2_lay1, pred_stream3_lay1, pred_layer1comb_lay1 = predictions_lay1
                        embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1, embedding_layer1comb_lay1 = embeddings_lay1

                        x_in_lay2 = torch.concat([embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1], axis = 1) # concat on embed dim torch.Size([4, 474, 2049, 64])
                        final_predictions, embedding, memory = model(x_in_lay2, indicator[0].unsqueeze(0).to(self.device))

                        loss = criterion(final_predictions, target_spec.unsqueeze(0).to(self.device))

                        out_spec.append(final_predictions)
                        lossacc += loss


                    if stream == 0: # 0 is the integrator mode
                        predictions_lay1, embeddings_lay1, memory_list = prev_layer(input_spec.unsqueeze(0).to(self.device), indicator[0].unsqueeze(0).to(self.device))
                        pred_stream1_lay1, pred_stream2_lay1, pred_stream3_lay1, pred_layer1comb_lay1 = predictions_lay1
                        embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1, embedding_layer1comb_lay1 = embeddings_lay1

                        x_in_lay2 = torch.concat([embedding_stream1_lay1, embedding_stream2_lay1, embedding_stream3_lay1], axis = 1) # concat on embed dim torch.Size([4, 474, 2049, 64])
                        final_predictions_list, embedding_list, memory_list = model(x_in_lay2, indicator[0].unsqueeze(0).to(self.device))

                        pred_stream1, pred_stream2, pred_stream3, pred_layer2comb = final_predictions_list
                        embedding_stream1, embedding_stream2, embedding_stream3, embedding_layer2comb = embedding_list

                        loss = criterion(pred_layer1comb_lay1, pred_layer2comb, target_spec.unsqueeze(0).to(self.device))

                        out_spec.append(pred_layer2comb)
                        lossacc += loss

        fullpred_spec = torch.concat(out_spec, dim = 3)
        fullpred_spec = fullpred_spec.squeeze(0)
        pred_audio = self.dataset.to_wav_torch(fullpred_spec[:, :, 0:phase_mixed[0].shape[-1]].detach().cpu(), phase_mixed[0].detach().cpu())
        pred_audio = pred_audio.squeeze(0).detach().numpy()
        if self.log_audio_metrics:
            #metrics = compute_metrics(raw_wavs_target[0].squeeze(0).detach().cpu().numpy(), pred_audio/abs(max(pred_audio)), self.config.SE_Config.SR, 0)
            #metrics = np.array(metrics)
            pass
        else:
            metrics = None

        if log_data:
            save_path = os.path.join(self.config.Coherence_Net_Config.checkpoint_path, self.config.Coherence_Net_Config.NAME, f"layer{layer}", "output_testset")
            os.makedirs(save_path, exist_ok = True)
            out_audio = pred_audio/abs(max(pred_audio))
            write(save_path + f"/{speaker_ids[0]}.wav", self.config.SE_Config.SR, out_audio)
            self.log_audio(out_audio, raw_wavs_target[0].squeeze(0).detach().cpu().numpy(), speaker_ids[0], step)
            self.log_spectrogram(fullpred_spec[:, :, 0:phase_mixed[0].shape[-1]].squeeze(0).detach().cpu().numpy(), torch.concat(input_spec_list, dim = 2)[:, :, 0:phase_mixed[0].shape[-1]].squeeze(0).detach().cpu().numpy(), None, speaker_ids[0], f"test/spectrogram" ,step, test = True)

        return loss, metrics  #(pesq_output, csig_output, cbak_output, covl_output, ssnr_output, stoi_output)

    def log_scalar(self, name, value, step):
        wandb.log({"step": step, f"{name}": value}, step = step)
        return

    def log_image(self, caption, array, step, name = "Spectrogram Representation"):
        images = wandb.Image(array, caption = caption)
        wandb.log({"step": step, name: images})
        return

    def log_spectrogram(self, array, gt_array, memory, caption, name,  step, test = False):
        if not test:
            fig, axs = plt.subplots(3, 3, figsize=(14,10))

            if len(axs.shape) != 1:
                for i in range(0, 3):
                    axs[i, 0].imshow(gt_array[i, 0], origin="lower", aspect="auto")

                for i in range(0, 3):
                    axs[i, 1].imshow(array[i, 0], origin="lower", aspect="auto")

                for i in range(0, 3):
                    axs[i, 2].imshow(gt_array[i, 0] - array[i, 0], origin="lower", aspect="auto")

            if len(axs.shape) == 1:
                    axs[0].imshow(gt_array[0, 0], origin="lower", aspect="auto")
                    axs[1].imshow(array[0, 0], origin="lower", aspect="auto")

            self.log_image(caption, fig, step, name)

        if test:
            fig, axs = plt.subplots(1, 2, figsize=(14,10))

            axs[0].imshow(gt_array, origin="lower", aspect="auto")
            axs[1].imshow(array, origin="lower", aspect="auto")

            self.log_image(caption, fig, step, name)

        return

    def log_audio(self, audio, gt_audio, name, step):
        wandb.log({"step": step, "Prediction": wandb.Audio(audio, caption = f"{name}", sample_rate= self.config.SE_Config.SR)})
        wandb.log({"step": step, "Ground Truth": wandb.Audio(gt_audio, caption = f"gt_{name}", sample_rate= self.config.SE_Config.SR)})
        return
