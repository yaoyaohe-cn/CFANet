import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.alpha = configs.alpha
        
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.freq_filters = nn.Parameter(
            torch.ones(1, self.enc_in, self.seq_len // 2 + 1)
        )
        
        
        self.predictor = nn.Sequential(
            nn.Linear(self.seg_num_x + 1 , self.d_model),  
            nn.ReLU(), # shape: [batch, seg_num_x+1, d_model]
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.seg_num_y)
        )
        
        
    def adaptive_freq_filter(self, x):

        # FFT
        x_fft = torch.fft.rfft(x, dim=2, norm='ortho')

        filtered = x_fft * (1 + self.alpha * torch.tanh(self.freq_filters))
        
        x_filtered = torch.fft.irfft(filtered, n=self.seq_len, dim=2, norm='ortho')
        
        return x_filtered
        

    
    def forward(self, x):
        batch_size = x.shape[0]
        
        seq_mean = x.mean(dim=1, keepdim=True)
        x_norm = (x - seq_mean) 
        
        x_norm = x_norm.permute(0, 2, 1)
        
        x_freq = self.adaptive_freq_filter(x_norm)
        
        # [B, C, L] -> [B*C, 1, L]
        x_trend_input = x_freq.view(-1, 1, self.seq_len)
        
        trend = F.avg_pool1d(x_trend_input, kernel_size=self.period_len, stride=self.period_len)
        trend_diff = trend[:, :, -1:] - trend[:, :, 0:1]  
        
        x_segments = x_freq.view(-1, self.seg_num_x, self.period_len)
        
        # [B*C, seg_num_x, period_len] -> [B*C, period_len, seg_num_x]
        x_segments = x_segments.permute(0, 2, 1) # shape: [B*C, period_len, seg_num_x]
        
        trend_expand = trend_diff.expand(-1, self.period_len, -1) # shape: [B*C, period_len, 1]
        x_with_trend = torch.cat([x_segments, trend_expand], dim=2) # shape: [B*C, period_len, seg_num_x+1]
        
        y = self.predictor(x_with_trend)
        
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)
        
        y = y  + seq_mean
        
        return y

