import torch 
import torch.nn.functional as F

from .streams import Stream1, Stream2, Stream3
from .stream_integerator import Stream_integerator


class Layer(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        """Initilize Energy predictor module.
        Args:
        input_channels: int = The number of input channels of the input spec,
        kernel_size: int = the kernel size for all the convolutions in the streams,
        embedding_dim: int = the number of embedding dimentions to work on,
        dcs_num_layers: int = the number convs in the DCS block,
        dcs_output_dim:int = The output of the dcs blocks,
        dcs_dilations = the list of dilations per block [3, 6, 9]
        
        """
        super(Layer, self).__init__()
        self.config = config
        self.stream1 = Stream1(input_channels = self.config.input_channels,
                               kernel_size = self.config.kernel_size,
                               embedding_dim = self.config.embedding_dim,
                               dcs_num_layers = self.config.dcs_num_layers,
                               dcs_output_dim = self.config.dcs_output_dim,
                               dcs_dilations = self.config.dcs_dilations,
                               num_sources = self.config.num_sources,
                               freq_bins = self.config.freq_bins,
                               time_frames = self.config.time_frames,
                               output_channels = self.config.output_channels
                              )
        
        self.stream2 = Stream2(input_channels = self.config.input_channels,
                               kernel_size = self.config.kernel_size,
                               embedding_dim = self.config.embedding_dim,
                               dcs_num_layers = self.config.dcs_num_layers,
                               dcs_output_dim = self.config.dcs_output_dim,
                               dcs_dilations = self.config.dcs_dilations,
                               num_sources = self.config.num_sources,
                               freq_bins = self.config.freq_bins,
                               time_frames = self.config.time_frames,
                               max_pool_kernel = self.config.max_pool_kernel,
                               max_pool_stride = self.config.max_pool_stride,
                               output_channels = self.config.output_channels
                              )
        
        self.stream3 = Stream3(input_channels = self.config.input_channels,
                               kernel_size = self.config.kernel_size,
                               embedding_dim = self.config.embedding_dim,
                               dcs_num_layers = self.config.dcs_num_layers,
                               dcs_output_dim = self.config.dcs_output_dim,
                               dcs_dilations = self.config.dcs_dilations,
                               num_sources = self.config.num_sources,
                               freq_bins = self.config.freq_bins,
                               time_frames = self.config.time_frames,
                               max_pool_kernel = self.config.max_pool_kernel,
                               max_pool_stride = self.config.max_pool_stride,
                               output_channels = self.config.output_channels
                              )
        
        self.stream_integrator = Stream_integerator(input_channels = self.config.stream_integerator_input_channels,
                                                    kernel_size = self.config.kernel_size,
                                                    embedding_dim = self.config.stream_integ_embedding,
                                                    dcs_num_layers = self.config.dcs_num_layers,
                                                    dcs_output_dim = self.config.dcs_output_dim,
                                                    dcs_dilations = self.config.dcs_dilations,
                                                    num_sources = self.config.num_sources,
                                                    freq_bins = self.config.freq_bins,
                                                    time_frames = self.config.time_frames,
                                                    output_channels = self.config.output_channels
                                                   )         
        
                
    def forward(self, xs: list, x_indicator: torch.Tensor):
        """Calculate forward propagation.
        Args:
            xs (List of Tensors):List of Tensor outputs from different streams - Batch of input sequences (B, In_chn, Fbin, Tdim).
            x_indicator:  
            stage:
        Returns:
            Tensor: Batch of predicted (B, In_chn + num_blocks*output_channels, Fbin, Tdim).
        """
        pred_stream1, embedding_stream1, Ha_stream1 = self.stream1(xs, x_indicator)
        pred_stream2, embedding_stream2, Ha_stream2 = self.stream2(xs, x_indicator)
        pred_stream3, embedding_stream3, Ha_stream3 = self.stream3(xs, x_indicator)
        
        pred_layer1comb, embedding_layer1comb, Ha_layer1comb = self.stream_integrator([embedding_stream1, embedding_stream2, embedding_stream3], x_indicator)
         
        return [pred_stream1, pred_stream2, pred_stream3, pred_layer1comb], [embedding_stream1, embedding_stream2, embedding_stream3, embedding_layer1comb], [Ha_stream1, Ha_stream2, Ha_stream3, Ha_layer1comb]
