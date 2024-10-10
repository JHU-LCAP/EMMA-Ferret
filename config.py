

## Main class for the model configuration contaning the general hyper parameters for the training of speech enhancement
class SE_Config:
    SR = 16000
    L_FRAME = 4096
    L_HOP = 512
    stage = 1
    batchSize = 1
    clean_files_dir = "/Volumes/Datasets/SE_Dataset/DS_10283_1942/clean_testset_wav_16k"
    noisy_files_dir = "/Volumes/Datasets/SE_Dataset/DS_10283_1942/clean_testset_wav_16k"
    clean_files_eval_dir = "/workspace/data/clean_testset_wav"
    noisy_files_eval_dir = "/workspace/data/noisy_testset_wav"
    
    lay1_stream1_bestchkpt_path = "/workspace/coherence_net/checkpoint/Ashwin_SE/layer1/Stream1_epoch250000.pt"
    lay1_stream2_bestchkpt_path = "/workspace/coherence_net/checkpoint/Ashwin_SE/layer1/Stream2_epoch135000.pt"
    lay1_stream3_bestchkpt_path = "/workspace/coherence_net/checkpoint/Ashwin_SE/layer1/Stream3_epoch430000.pt"
    
    lay2_stream1_bestchkpt_path = ""
    lay2_stream2_bestchkpt_path = ""
    lay2_stream3_bestchkpt_path = ""
    
    

class Coherence_Net_Config:
    NAME = 'Ashwin_SE'
    
    ## Train Config
    checkpoint_path = f"checkpoint/{NAME}/"
    CKPT_PATH = 'checkpoints/' + NAME
    LR = 0.0001
    DECAY_RATE = 0.02
    DECAY_STEP = 215000
    FINAL_STEP =250001
    CKPT_STEP = 25000
    
    train_split = 0.9 # 90 percent 
    
    stream1_steps = 20 # the number of epochs 
    stream2_steps = 20 # the number of epochs
    stream3_steps = 20 # the number of epochs
    stream_integerator_steps = 20 # the number of epochs
    
    ## Layer 1 Config
    class layer1:
        num_layers = 2
        num_streams_per_layer = 3    
        
        input_channels = 1   # The number of input channels
        kernel_size = 3      # The kernel size for convolutions
        embedding_dim = 128  # The embedding dim before the dcs
        dcs_num_layers = 3   # number of conv layers in dcs 
        dcs_output_dim = 10  # The output embedding of the dcs blocks
        dcs_dilations = [3, 6, 9] # The dilations in the dcs block the length should be equal to the number of dcs layers
        num_sources = 2      # The number of sources to sperate
        freq_bins = 2049     # the number of input freq bins
        time_frames = 64     # The time frame of each input

        max_pool_kernel = 2  # The max pool layer kernel 
        max_pool_stride = 2  # The max pool layer stride

        stream_integerator_input_channels = num_streams_per_layer*(embedding_dim + dcs_num_layers*dcs_output_dim)
        stream_integ_embedding = 96
        
        output_channels = 1

    
    ## Layer 2 Config
    class layer2:
        num_layers = 2
        num_streams_per_layer = 3
        
        
        kernel_size = 3      # The kernel size for convolutions
        embedding_dim = 128  # The embedding dim before the dcs
        dcs_num_layers = 3   # number of conv layers in dcs 
        dcs_output_dim = 10  # The output embedding of the dcs blocks
        dcs_dilations = [3, 6, 9] # The dilations in the dcs block the length should be equal to the number of dcs layers
        num_sources = 2      # The number of sources to sperate
        freq_bins = 2049     # the number of input freq bins
        time_frames = 64     # The time frame of each input

        max_pool_kernel = 2  # The max pool layer kernel 
        max_pool_stride = 2  # The max pool layer stride

        stream_integerator_input_channels = num_streams_per_layer*(embedding_dim + dcs_num_layers*dcs_output_dim) 
        
        input_channels = num_streams_per_layer*(embedding_dim + dcs_num_layers*dcs_output_dim)  # The number of input channels to layer 2 
        stream_integ_embedding = 96
        
        output_channels = 1

