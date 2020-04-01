"""Top-level model classes.

Author: Dat Nguyen (dpnguyen@stanford.edu)
Modified from CS224n starter code by Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn

from layers import Embedding, EncoderLayer, BiDAFAttention, QANetOutput, PositionEncoder


class QANet(nn.Module):
    """QANet model: https://arxiv.org/pdf/1804.09541.pdf

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        dropout_main (float): Dropout rate between every two main layers.
    """
    def __init__(self, word_vectors, hidden_size, char_embed_size, word_from_char_size, dropout_main,
                 embed_encoder_num_convs, embed_encoder_kernel_size, embed_encoder_num_heads, embed_encoder_num_blocks,
                 model_encoder_num_convs, model_encoder_kernel_size, model_encoder_num_heads, model_encoder_num_blocks):
        super(QANet, self).__init__()
        
        self.emb = Embedding(word_vectors, hidden_size, char_embed_size, word_from_char_size)
        self.embed_encoder = EncoderLayer(hidden_size,
                                          embed_encoder_num_convs, embed_encoder_kernel_size,
                                          embed_encoder_num_heads,
                                          embed_encoder_num_blocks)
        self.cq_att = BiDAFAttention(hidden_size)
        self.model_encoder = EncoderLayer(hidden_size,
                                          model_encoder_num_convs, model_encoder_kernel_size,
                                          model_encoder_num_heads, model_encoder_num_blocks)
        self.output = QANetOutput(hidden_size)
        
        ## Linear layer between context-query attention and model encoder: use to make sure the input has the same size.
        self.pre_model_encoder = nn.Linear(4 * hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_main) ## Dropout between layers

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
    	
    	## Compute masks for context and query. Masks hold value TRUE for pad tokens.
        c_is_pad = torch.zeros_like(cw_idxs) == cw_idxs ## shape (batch_size, c_len)
        q_is_pad = torch.zeros_like(qw_idxs) == qw_idxs ## shape (batch_size, q_len)
        
        
        ## Look up context and query embeddings
        ## shape (batch_size, c_len, hidden_size) and (batch_size, q_len, hidden_size), respectively
        x, y = self.dropout(self.emb(cw_idxs, cc_idxs)), self.dropout(self.emb(qw_idxs, qc_idxs))
        
        ## Compute PositionEncoder(x).
        ## This value is memorized because it will be re-used many times in the model encoder layers.
        ## PositionEncoder(y) will only be used once and just doesn't need to be saved.
        pe_c = PositionEncoder(x) ## shape (c_len, hidden_size)
        
        ## Pass through embedding encoding layer
        ## shape (batch_size, c_len, hidden_size) and (batch_size, q_len, hidden_size), respectively
        x = self.dropout(self.embed_encoder(x, pe_c, c_is_pad)),
        y = self.dropout(self.embed_encoder(y, PositionEncoder(y), q_is_pad))
        
        ## Context-query bi-attention, then map back to hidden_size
        bi_att = self.cq_att(x, y, c_is_pad, q_is_pad) ## shape (batch_size, c_len, 4 * hidden_size)
        bi_att = self.pre_model_encoder(bi_att) ## shape (batch_size, c_len, hidden_size)        
        bi_att = self.dropout(bi_att)
        
        ## Apply model encoders. Shape (batch_size, c_len, hidden_size) each
        M0 = self.model_encoder(bi_att, pe_c, c_is_pad)
        M1 = self.model_encoder(self.dropout(M0), pe_c, c_is_pad)
        M2 = self.model_encoder(self.dropout(M1), pe_c, c_is_pad)
        
        out = self.output(M0, M1, M2, c_is_pad) ## 2 tensors, each (batch_size, c_len)

        return out
