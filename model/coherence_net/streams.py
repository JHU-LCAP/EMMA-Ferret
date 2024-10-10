import torch 
import torch.nn.functional as F

from .dcs import DCS_multi_lay_resolution
from .attention import Gate_Attention

class Stream1(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        kernel_size: int = 3,
        embedding_dim: int = 128,
        dcs_num_layers: int = 3,
        dcs_output_dim:int = 10,
        dcs_dilations:list = [3, 6, 9],
        num_sources:int = 2,
        freq_bins:int = 2049,
        time_frames:int = 64,
        output_channels:int = 1
        
    ):
        """Initilize module.
        Args:
        input_channels: int = The number of input channels of the input spec,
        kernel_size: int = the kernel size for all the convolutions in the streams,
        embedding_dim: int = the number of embedding dimentions to work on,
        dcs_num_layers: int = the number convs in the DCS block,
        dcs_output_dim:int = The output of the dcs blocks,
        dcs_dilations = the list of dilations per block [3, 6, 9]
        
        """
        super(Stream1, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, embedding_dim, kernel_size, padding = 'same'),
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
            torch.nn.LeakyReLU(),
        )
        
        
    def forward(self, xs: torch.Tensor, x_indicator: torch.Tensor):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        xs = self.conv1(xs) # Output shape - [B, embedding_dim, Freq_contetn, Time_frames]
        xs = self.dcs(xs)   # Output Shape - [B, embedding_dim + num_blocks*dcs_output_channels, Freq_content, Time_frames] as we are concatinating the outputs from the dcs
        embedding, Ha = self.attention(xs, x_indicator) # output shape [B, embedding_dim + num_blocks*dcs_output_channels, Freq_content, Time_frames]
        final_predictions = self.conv_final(embedding)
         
        return final_predictions, embedding, Ha

    def inference(self, xs: torch.Tensor, alpha: float = 1.0):
        """Inference duration.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        return
    
class Stream2(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        kernel_size: int = 3,
        embedding_dim: int = 128,
        dcs_num_layers: int = 3,
        dcs_output_dim:int = 10,
        dcs_dilations:list = [3, 6, 9],
        num_sources:int = 2,
        freq_bins:int = 2049,
        time_frames:int = 64,
        max_pool_kernel:int = 2,
        max_pool_stride:int = 2,
        output_channels:int = 1
        
    ):
        """Initilize module.
        Args:
        input_channels: int = The number of input channels of the input spec,
        kernel_size: int = the kernel size for all the convolutions in the streams,
        embedding_dim: int = the number of embedding dimentions to work on,
        dcs_num_layers: int = the number convs in the DCS block,
        dcs_output_dim:int = The output of the dcs blocks,
        dcs_dilations = the list of dilations per block [3, 6, 9]
        
        """
        super(Stream2, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, embedding_dim, kernel_size, padding = 'same'),
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
            torch.nn.LeakyReLU(),
        )
        self.max_pool = torch.nn.MaxPool2d(max_pool_kernel, stride = max_pool_stride)
        self.upsample = torch.nn.ConvTranspose2d(embedding_dim + dcs_num_layers*dcs_output_dim, embedding_dim + dcs_num_layers*dcs_output_dim, max_pool_kernel, stride=max_pool_stride)
        
        
    def forward(self, xs: torch.Tensor, x_indicator: torch.Tensor):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        conv1_xs = self.conv1(xs) # Output shape - [B, embedding_dim, Freq_contetn, Time_frames]
        
        max_pool_xs = self.max_pool(conv1_xs) # [B, embedding_dim, Freq_contetn/2, Time_frames/2]
        
        dcs_xs = self.dcs(max_pool_xs)   # Output Shape - [B, embedding_dim + num_blocks*dcs_output_channels, Freq_content, Time_frames] as we are concatinating the outputs from the dcs
        
        upsample_xs = self.upsample(dcs_xs, output_size = conv1_xs.shape) ## [B, embedding_dim + num_blocks*dcs_output_channels, Freq_content, Time_frames]

        #torch.save(upsample_xs, "embedding_before_attn_mix_in_female_out.pt")
        
        embedding, Ha = self.attention(upsample_xs, x_indicator) # output shape [B, embedding_dim + num_blocks*dcs_output_channels, Freq_content, Time_frames]
        
        torch.save(Ha, "memory_attn_male_in_male_out.pt")

        final_predictions = self.conv_final(embedding)

        #torch.save(self.conv_final.weight, "mix_in_female_out_final_conv_weight.pt")
        #torch.save(self.conv_final.bias, "mix_in_female_out_final_conv_weight.pt")
         
        return final_predictions, upsample_xs, embedding

    def inference(self, xs: torch.Tensor, alpha: float = 1.0):
        """Inference duration.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        return

    
class Stream3(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        kernel_size: int = 3,
        embedding_dim: int = 128,
        dcs_num_layers: int = 3,
        dcs_output_dim:int = 10,
        dcs_dilations:list = [3, 6, 9],
        num_sources:int = 2,
        freq_bins:int = 2049,
        time_frames:int = 64,
        max_pool_kernel:int = 2,
        max_pool_stride:int = 2,
        output_channels:int = 1
        
        
    ):
        """Initilize module.
        Args:
        input_channels: int = The number of input channels of the input spec,
        kernel_size: int = the kernel size for all the convolutions in the streams,
        embedding_dim: int = the number of embedding dimentions to work on,
        dcs_num_layers: int = the number convs in the DCS block,
        dcs_output_dim:int = The output of the dcs blocks,
        dcs_dilations = the list of dilations per block [3, 6, 9]
        
        """
        super(Stream3, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, embedding_dim, kernel_size, padding = 'same'),
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
            torch.nn.LeakyReLU(),
        )
        
        self.max_pool_1 = torch.nn.MaxPool2d(max_pool_kernel, stride = max_pool_stride)
        self.upsample_1 = torch.nn.ConvTranspose2d(embedding_dim + dcs_num_layers*dcs_output_dim, embedding_dim + dcs_num_layers*dcs_output_dim, max_pool_kernel, stride=max_pool_stride)
        
        self.max_pool_2 = torch.nn.MaxPool2d(max_pool_kernel, stride = max_pool_stride)
        self.upsample_2 = torch.nn.ConvTranspose2d(embedding_dim + dcs_num_layers*dcs_output_dim, embedding_dim + dcs_num_layers*dcs_output_dim, max_pool_kernel, stride=max_pool_stride)
        
        
    def forward(self, xs: torch.Tensor, x_indicator: torch.Tensor):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        conv1_xs = self.conv1(xs) # Output shape - [B, embedding_dim, Freq_contetn, Time_frames]
        
        max_pool_xs_1 = self.max_pool_1(conv1_xs) # [B, embedding_dim, Freq_contetn/2, Time_frames/2]
        max_pool_xs_2 = self.max_pool_2(max_pool_xs_1) # [B, embedding_dim, Freq_contetn/4, Time_frames/4]
        
        
        dcs_xs = self.dcs(max_pool_xs_2)   # Output Shape - [B, embedding_dim + num_blocks*dcs_output_channels, Freq_content, Time_frames] as we are concatinating the outputs from the dcs
        
        upsample_xs_1 = self.upsample_1(dcs_xs, output_size = max_pool_xs_1.shape) ## [B, embedding_dim, Freq_contetn/2, Time_frames/2]
        upsample_xs_2 = self.upsample_2(upsample_xs_1, output_size = conv1_xs.shape) ## [B, embedding_dim, Freq_contetn, Time_frames]
        
        embedding, Ha = self.attention(upsample_xs_2, x_indicator) # output shape [B, embedding_dim + num_blocks*dcs_output_channels, Freq_content, Time_frames]
        
        final_predictions = self.conv_final(embedding)
         
        return final_predictions, embedding, Ha

    def inference(self, xs: torch.Tensor, alpha: float = 1.0):
        """Inference duration.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        return
