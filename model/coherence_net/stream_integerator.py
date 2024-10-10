import torch 
import torch.nn.functional as F

from .dcs import DCS_multi_lay_resolution
from .attention import Gate_Attention


class Stream_integerator(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 128*3,
        kernel_size: int = 3,
        embedding_dim: int = 96,
        dcs_num_layers: int = 3,
        dcs_output_dim:int = 10,
        dcs_dilations:list = [3, 6, 9],
        num_sources:int = 2,
        freq_bins:int = 2049,
        time_frames:int = 64,
        output_channels: int = 1,
        
    ):
        """Initilize Stream Integerator Module module.
        Args:
        input_channels: int = The number of input channels of the input spec,
        kernel_size: int = the kernel size for all the convolutions in the streams,
        embedding_dim: int = the number of embedding dimentions to work on,
        dcs_num_layers: int = the number convs in the DCS block,
        dcs_output_dim:int = The output of the dcs blocks,
        dcs_dilations = the list of dilations per block [3, 6, 9]
        
        """
        super(Stream_integerator, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, embedding_dim, kernel_size, padding = 'same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(embedding_dim, embedding_dim, kernel_size, padding = 'same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(embedding_dim, embedding_dim, kernel_size, padding = 'same'),
            torch.nn.LeakyReLU(),
        )
        self.dcs = DCS_multi_lay_resolution(input_channels = embedding_dim,
                                            output_channels = dcs_output_dim,
                                            num_blocks = dcs_num_layers,
                                            dilations = dcs_dilations,
                                            kernel_size = kernel_size,
                                            mode = "concat"
                                            )
        
        self.attention = Gate_Attention(input_channels = num_sources,
                                        embedding_dimentions = embedding_dim + dcs_num_layers*dcs_output_dim, 
                                        frequency_content = freq_bins, 
                                        time_frames = time_frames)
        
        self.conv_final = torch.nn.Sequential(
            torch.nn.Conv2d(embedding_dim + dcs_num_layers*dcs_output_dim, output_channels, kernel_size, padding = 'same'), 
            torch.nn.LeakyReLU()  ### remember this line and train again later after layer 0 train, mostly should not effect cause its the integerator anyways
        )
                
    def forward(self, xs: list, x_indicator: torch.Tensor):
        """Calculate forward propagation.
        Args:
            xs (List of Tensors):List of Tensor outputs from different streams - Batch of input sequences (B, In_chn, Fbin, Tdim).
        Returns:
            Tensor: Batch of predicted (B, In_chn + num_blocks*output_channels, Fbin, Tdim).
        """
        xs = torch.concat(xs, axis = 1) # concatenate on the embedding dimention   torch.Size([4, 474, 2049, 64])
        xs = self.conv1(xs) # Output shape - [B, embedding_dim, Freq_contetn, Time_frames]
        xs = self.dcs(xs)   # Output Shape - [B, embedding_dim + num_blocks*dcs_output_channels, Freq_content, Time_frames] as we are concatinating the outputs from the dcs
        embedding, Ha = self.attention(xs, x_indicator) # output shape [B, embedding_dim + num_blocks*dcs_output_channels, Freq_content, Time_frames]
        final_predictions = self.conv_final(embedding)
        
        
        return final_predictions, embedding, Ha

