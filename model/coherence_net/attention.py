import torch 
import torch.nn.functional as F


class Gate_Attention(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        embedding_dimentions:int = 158, 
        frequency_content:int = 2049, 
        time_frames:int = 64, 
    ):
        """Initilize Attention block based on Coherence.
        Args:
            input_channels: int = The number of input channels for indicator variable projection, aka number of sources
            embedding_dim: int = The number of output dimention for projection
            frequency_content:int = number of frequency bins 
            time_frames:int = number of time frames
        """
        super(Gate_Attention, self).__init__()
        self.embedding_dimentions = embedding_dimentions
        self.frequency_content = frequency_content
        self.time_frames = time_frames
        
        self.projection_inidicator = torch.nn.Linear(input_channels, self.embedding_dimentions*self.frequency_content)
                
    def forward(self, xs: torch.Tensor, x_indicator: torch.Tensor):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Embed, Fbin, Tdim).
            x_indicator: torch.Tensor = Indicator variable
        Returns:
            Tensor: Batch of predicted (B, In_chn + num_blocks*output_channels, Fbin, Tdim).
        """
        batched_gated_embedding = torch.zeros_like(xs)
        batched_attention_anchor = torch.zeros([xs.shape[0], xs.shape[2], xs.shape[1] ])
        for i in range(0, xs.shape[0]):
            inp = xs[i]
            gated_embedding, attention_anchor = self.coh_fn(inp, x_indicator[i:i+1])
            batched_gated_embedding[i] = gated_embedding
            batched_attention_anchor[i] = attention_anchor
            
        return batched_gated_embedding, batched_attention_anchor 
    
    def coh_fn(self, inp_img, indicator):
        """Gatted Attention block
        Args:
            inp_img: tensor = The input embedding tensor of a single example (not a batch input) shape is [embedding_dimentions, frequency_content, time_frames]
            indicator: tensor = the indicator variable denoting where to focus on, shape [1, num_of_sources]
        Returns:
            prediction: tensor = the tensor after applying attention
            Ha: tensor = 
        """        
        inp_img_squz_ef = inp_img.view(self.time_frames, self.frequency_content*self.embedding_dimentions).T 
        ## Shape is [frequency_content*embedding_dimentions, time_frames]
        
        indicator_projection = self.projection_inidicator(indicator) # Shape is [1, freq*embed] # This is the Ox or the learned memory of the object
        
        Rx = torch.matmul(inp_img_squz_ef.T, indicator_projection.T) # shape is [time_frames, 1]
        
        output_before_sigmoid = torch.matmul(indicator_projection.T, Rx.T) ## Shape [time_frames, freq*embed], the Rx actual
        
        
        Ha = torch.mean(torch.sigmoid(output_before_sigmoid), 0).unsqueeze(1) # Shape is [time_frames, 1], the mean firing per time frame
        
        output_before_sigmoid_unfolded = output_before_sigmoid.view(self.embedding_dimentions, self.frequency_content, self.time_frames)

        output = torch.mul(inp_img, output_before_sigmoid_unfolded) # torch.sigmoid removed
        
        torch.save(output_before_sigmoid_unfolded, 'unfolded_tensor_before_sigmoid.pt')
        
        return output, indicator_projection.view(-1, self.frequency_content, self.embedding_dimentions)

