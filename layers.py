"""Assortment of layers for use in models.py.
Modified from CS224n Default Project starter code by Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF.
    Word vectors are obtained by concatenating pre-trained word-level vectors and CharEmbedding vectors.
    Word vectors are further refined using dropout and a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).
    Note that unlike the orignal paper, we apply the projection down to hidden_size BEFORE applying highway network.
    This way the model uses fewer parameters and is faster.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.        
    """
    def __init__(self, word_vectors, hidden_size, char_embed_size, word_from_char_size):
        super(Embedding, self).__init__()
        
        ## Obtain word embeddings from characters
        self.char_embed = WordFromCharEmbedding(char_embed_size, word_from_char_size)
        
        ## Obtain word embeddings from pretrained word vectors
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        
        self.charDropout = nn.Dropout(0.05)
        self.wordDropout = nn.Dropout(0.1)        

        self.wordLinear = nn.Parameter(torch.empty(word_vectors.size(1), hidden_size))
        nn.init.xavier_uniform_(self.wordLinear)
        self.charLinear = nn.Parameter(torch.empty(word_from_char_size, hidden_size))
        nn.init.xavier_uniform_(self.charLinear)
        
        self.hwy = HighwayEncoder(2, hidden_size)        

    def forward(self, text_word_ids, text_char_ids):
    	"""Arguments:
    	text_word_ids: tensor of shape (batch_size, text_len). Containing indices of words in the context/query.
    	text_char_ids: tensor of shape (batch_size, text_len, char_embed_size). Contain indices of words in the context/query.
    	
    	Output: tensor of shape (batch_size, text_len, hidden_size)
    	
    	This method applies linear projections separately instead of concatenating the input tensors to save a little memory.
    	"""
    	## Look up embeddings.
	## Shapes (batch_size, text_len, word_vectors.size(1)) and (batch_size, text_len, word_from_char_size), respectively
    	x, y = self.wordDropout(self.word_embed(text_word_ids)), self.charDropout(self.char_embed(text_char_ids))
    	
    	## Apply linear layers. Shapes both (batch_size, text_len, hidden_size)
    	x, y = torch.matmul(x, self.wordLinear), torch.matmul(y, self.charLinear)
    	
    	return self.hwy(x+y)

		
class WordFromCharEmbedding(nn.Module):
	"""Obtain embedding of words from convoling and maxpooling their characters' embeddings.
	
	Arugments:
	char_embed_size (int): Dimension of each character vector.
	word_from_char_size (int): Dimension of word vector obtained from character vectors.
	"""
	
	def __init__(self, char_embed_size, word_from_char_size):
		super(WordFromCharEmbedding, self).__init__()
		
		## There are 1376 characters used in our dataset
		## More generally, char_vocab_size can be computed by importing char2idx_dic from util and call len(char2idx_dic())
		char_vocab_size = 1376
		
		char_embed_weight = torch.Tensor(char_vocab_size, char_embed_size)
		nn.init.normal_(char_embed_weight)
		
		## Initialize char vector of --NULL-- to be 0. 
		char_embed_weight[0].fill_(0)
		
		## Initialize char vector of --OOV-- to be 0.
		## However, unlike the char vector of --NULL--, this char vector does receive gradients.				
		char_embed_weight[1].fill_(0) 
		self.char_embedding = nn.Embedding.from_pretrained(char_embed_weight, freeze=False, padding_idx=0)
		del char_embed_weight
		
		self.conv = nn.Conv1d(char_embed_size, word_from_char_size, kernel_size = 5, padding = 2)
		
	def forward(self, x):
		"""
		x: input tensor of shape (batch_size, text_len, max_word_len).
			Here text_len is the length of the context/query; max_word_len is the longest word in the batch.
		Output: Tensor of size (batch_size, text_len, word_from_char_size)
		"""		
		x = self.char_embedding(x) ## size (batch_size, text_len, max_word_len, char_embed_size)
		batch_size, text_len, max_word_len, char_embed_size = x.size()
		
		## Reshape and transpose x to follow the convention of nn.Conv1D:
		## This module can only take 3D tensors, and the number of input channels has to be the middle dimension
		## Using view() before transpose() means we don't need to apply contiguous()
		## size: (batch_size * text_len, char_embed_size, max_word_len).
		x = x.view(-1, max_word_len, char_embed_size).transpose(1,2) 
		
		x = self.conv(x) ## size: (batch_size * text_len, word_from_char_size, max_word_len)
		x, _ = torch.max(x, 2) ## size: (batch_size * text_len, word_from_char_size)
		x = F.relu(x) ## size: (batch_size * text_len, word_from_char_size)
		x = x.view(batch_size, text_len, -1) ## size: (batch_size, text_len, word_from_char_size)
		
		return x
		
class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x
        
        
def PositionEncoder(x):
	"""Positional Encoding layer with fixed encoding vector based on sin and cos,
	as in http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks
	Implemented as a function instead of a module because we may not know the shape of x
	(in particular, the text_len dimension) before hand.
	This function returns just the fixed PE vector instead of the sum x + PE:
	This is to avoid computing PE again and again in repeated encoder blocks.
	
	Arguments:
	x: input tensor of shape (batch_size, text_len, input_dim)
	
	Output:
	pe: tensor of shape (text_len, input_dim)
	pe[position, 2i] = sin( position * 10000^(- 2i / input_dim) )
	pe[position, 2i+1] = cos( position * 10000^(- 2i / input_dim) )
	"""
	_, text_len, input_dim = x.size()
	
	position = torch.arange(text_len, dtype = torch.float, device = x.device) ## shape (text_len, )
	
	div_term = torch.arange(0, input_dim, 2, dtype = torch.float, device = x.device) ##shape (input_dim//2, )
	div_term = torch.pow(10000, - div_term/input_dim)
	
	## Compute angles: tensor of shape (text_len, input_dim //2) as the outer product of position and div_term
	## angles[position, i] = position * 10000^(- 2i / input_dim)
	angles = torch.ger(position, div_term)
	
	## Interweave sin(angles) and cos(angles)
	## shape (text_len, input_dim)
	pe = torch.stack( (torch.sin(angles), torch.cos(angles)), dim = 2).view(text_len, input_dim) 
	return pe
	
	
class DepthwiseSeparableConvolution(nn.Module):
	"""Depthwise Separable Convolutional Layer used in QANet encoder block
	Illustration for depthwise separable convolution:
	https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
	Input is first passed through LayerNorm, then a Depthwise Separable Convolutional Layer.
	Leakly ReLU activation is applied and a skip connection is added at the end.		
	
	Arguments:
	input_dim (int): Dimension of each (non-batched) input vector.
		In the Conv1D documentation, this is referred to as the number of input channels. 	
	kernel_size (int): Kernel size.
		Expected to be an odd number so that the output has the same shape as the input,
		otherwise the skip connection doesn't make sense.
	p_dropout (float): Dropout rate.
	"""
	def __init__(self, input_dim, kernel_size, p_dropout):
		super(DepthwiseSeparableConvolution, self).__init__()
		
		## Depthwise convolution layer.
		## Padding size is set to kernel_size // 2. This would guarantee that 
		##	(1) the kernel is never too big, and
		##	(2) the output text_len is the same as the input text_len.
		## Bias is set to False because we will add bias in the pointwise convolution layer.
		self.depthwise = nn.Conv1d(input_dim, input_dim, kernel_size, padding = kernel_size // 2,
					   groups = input_dim, bias = False)
		
		## Pointwise convolution layer
		## We use nn.Linear instead of nn.Conv1D with kernel size 1 - they do the same thing
		## We are setting output_dim to be equal to input_dim even though it doesn't have to be in general.
		## This is so that a skip connection can be used.
		self.pointwise = nn.Linear(input_dim, input_dim)
		
		## Layer normalization across the features, i.e. across the last dimension that is equal to input_dim
		self.layernorm = nn.LayerNorm(input_dim)
		
		self.dropout = nn.Dropout(p_dropout)
		
	def forward(self, x):
		"""
		x: input tensor of shape (batch_size, text_len, input_dim).
			Here text_len is the length of the context/question.
		The shape stays the same (batch_size, text_len, input_dim) through every step.
		"""
		skip_connection = x
		x = self.layernorm(x)
		
		## Call transpose(1,2) back and forth because nn.Conv1D requires the number of input channels to be
		## the MIDDLE dimension.
		x = self.depthwise(x.transpose(1,2)).transpose(1,2)
		
		x = self.pointwise(x)		
		x = F.leaky_relu(x)
		return self.dropout(x) + skip_connection
		
class SelfAttention(nn.Module):
	"""Multihead Attention with scaled dot product attention, as in "Attention is all you need"
	Input is first passed through LayerNorm, then nn.MultiheadAttention. A skip connection is added at the end.
	
	Note that in nn.MultiheadAttention, kdim and vdim don't mean the same thing as they do in the paper.
	In particular, here we don't need to manually set them to input_dim // num_heads.
	
	Arguments:
	input_dim (int): Dimension of each (non-batched) input vector.
	num_heads (int): Number of attention heads.
	p_dropout (float): Dropout rate.
	"""
	def __init__(self, input_dim, num_heads, p_dropout):
		super(SelfAttention, self).__init__()
		
		self.attention = nn.MultiheadAttention(input_dim, num_heads)
		self.dropout = nn.Dropout(p_dropout)
		
		## Layer normalization across the features, i.e. across the last dimension that is equal to input_dim
		self.layernorm = nn.LayerNorm(input_dim)
		
		
	def forward(self, x, is_pad):
		"""
		x: input tensor of shape (batch_size, text_len, input_dim).
			Here text_len is the length of the context/question.
		is_pad: tensor of shape(batch_size, text_len). Hold value TRUE for pad tokens. 
		Output: tensor of the same shape as the input, (batch_size, text_len, input_dim)
		"""
		skip_connection = x
		
		x = self.layernorm(x) ## shape (batch_size, text_len, input_dim)
		
		## shape (text_len, batch_size, input_dim).
		## Here transpose() is needed because of the convention of nn.MultiheadAttention.
		x = x.transpose(0,1)		
		x, _ = self.attention(x, x, x, key_padding_mask = is_pad, need_weights=False) 
		
		x = x.transpose(0,1) ## shape (batch_size, text_len, input_dim)		
		return self.dropout(x) + skip_connection

class FeedForward(nn.Module):
	"""Feed forward layer with ReLU activation.
	Input is first passed through LayerNorm, then a linear layer, then non-linear activation, then another linear layer.
	A skip connection is added at the end.
	
	Arguments:
	input_dim (int): Dimension of each (non-batched) input vector.
	p_dropout: Dropout rate.
	"""
	def __init__(self, input_dim, p_dropout):
		super(FeedForward, self).__init__()
		
		self.linear1 = nn.Linear(input_dim, input_dim)
		self.linear2 = nn.Linear(input_dim, input_dim)
		self.dropout = nn.Dropout(p_dropout)
		
		## Layer normalization across the features, i.e. across the last dimension that is equal to input_dim
		self.layernorm = nn.LayerNorm(input_dim)
	def forward(self, x):
		"""
		x: input tensor of shape (batch_size, text_len, input_dim).
		The shape stays the same (batch_size, text_len, input_dim) through every step.		
		"""
		skip_connection = x
		
		x = self.layernorm(x)
		x = self.linear1(x)
		x = F.relu(x)
		x = self.linear2(x)
		
		return self.dropout(x) + skip_connection

class EncoderBlock(nn.Module):
	"""One encoder block in the QANet model:	
	Composition of: PositionEncoder -> DepthwiseSeparableConvolution * num_convs -> SelfAttention -> FeedForward.
	
	REMARK: Earlier layers have smaller dropout rates, as described in the QANet paper:
	..."within EACH embedding or model encoder layer, each sublayer l has survival probability p_l= 1−l/L (1−p_L),
	where L is the last layer and p_L= 0.9." 
	
	Arguments:
	input_dim (int): Dimension of each (non-batched) input vector.
		The output vector of each sublayer will also have the same dimension
	num_convs (int): Number of convolutional layers inside the block
	kernel_size (int): Kernel size of each convolutional layer
	num_heads (int): Number of attention heads in each block
	num_blocks (int): Number of EncoderBlock(s) in the embedding/model encoder layer.
		This is needed to compute the dropout rate, see REMARK above and examples below.
	block_index (int): The (0-based) index of the current EncoderBlock in the embedding/model encoder layer.
		This is needed to compute the dropout rate, see REMARK above and examples below.
	
	Examples:
	In the original paper, for the model encoder layer, num_block = 7, and block_index ranges from 0 to 6.
	For the embedding encoder layer, num_block = 1 and block_index = 0 for the only block in the layer.
	"""
	def __init__(self, input_dim, num_convs, kernel_size, num_heads, num_blocks, block_index):
		super(EncoderBlock, self).__init__()
		
		## Compute dropout rates, see the REMARK 1 above
		## The layers in each block are:
		## PositionEncoder, num_convs * DepthwiseSeparableConvolution, SelfAttention, and FeedForward.
		layers_per_block = 3 + num_convs
		
		## Total number of layers in num_block blocks. This is the big L in the dropout rate formula above
		L = layers_per_block*num_blocks 
		
		## The (1-based) index of the the first sublayer of the current block, which is PositionEncoder
		## This is the small l in the dropout rate formula above
		l = 1 + layers_per_block*block_index 
		
		self.PE_dropout = nn.Dropout(l * 0.1/L)
		
		## Convolutional layers.
		self.convs = nn.Sequential(*[DepthwiseSeparableConvolution(input_dim, kernel_size, (l + i) * 0.1/L )
					     for i in range(1,1+num_convs)])
		
		## Self-attention layer.
		## This is the (2 + num_convs)-th sublayer in the block, so the dropout rate is (l + 1 + num_convs)*0.1/L
		self.attention = SelfAttention(input_dim, num_heads, (l + 1 + num_convs)*0.1/L )
		
		## FeedForward layers.
		## This is the (3 + num_convs)-th layer in the block, so the dropout rate is (l + 2 + num_convs)*0.1/L
		self.feedfwd = FeedForward(input_dim, (l + 2 + num_convs)*0.1/L )
		
	def forward(self, x, pe, is_pad):
		"""
		x: input tensor of shape (batch_size, text_len, input_dim)
		pe: expected to be PositionEncoder(x), shape (text_len, input_dim)
		is_pad: tensor of shape(batch_size, text_len). Hold value TRUE for pad tokens. 
		output: tensor of the same shape (batch_size, text_len, input_dim)
		"""
		x = self.PE_dropout(x + pe)		
		x = self.convs(x) ## shape (batch_size, text_len, input_dim)
		x = self.attention(x, is_pad)
		x = self.feedfwd(x)
		return x

class EncoderLayer(nn.Module):
	"""Wrap multiple encoder blocks together.
	This module is used to construct one Embedding Encoder Layer or one Model Encoder Layer in QANet.
	Note that in the case of Model Encoder, this is just ONE layer in the QANet diagram, not 3 repeated layers.
	
	Arguments:
	input_dim (int): Dimension of each (non-batched) input vector.	
	num_convs (int): Number of convolution sublayers in each encoder block.
	kernel_size (int): Kernel size of each convolution sublayer.
	num_heads (int): Number of attention heads in each encoder block.
	num_blocks (int): Number of encoder blocks in each embedding encoder layer.
	"""
	def __init__(self, input_dim, num_convs, kernel_size, num_heads, num_blocks):
		super(EncoderLayer, self).__init__()
		
		self.encoder_blocks = nn.ModuleList([
			EncoderBlock(input_dim, num_convs, kernel_size, num_heads, num_blocks, block_index)
			for block_index in range(num_blocks)])
		
	def forward(self, x, pe, is_pad):
		"""
		x: input tensor of shape (batch_size, text_len, input_dim)
		pe: expected to be PositionEncoder(x), shape (text_len, input_dim)
		is_pad: tensor of shape(batch_size, text_len). Hold value TRUE for pad tokens. 
		output: tensor of the same shape (batch_size, text_len, input_dim)
		"""
		for encoder_block in self.encoder_blocks:
			x = encoder_block(x, pe, is_pad)
		return x

class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.
    This can be reused in our QANet model without any modification.
    Here hidden_size means the same thing as input_dim in other modules in this file.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 4 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.        
    """
    def __init__(self, hidden_size):
        super(BiDAFAttention, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.c_weight = nn.Parameter(torch.empty(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.empty(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.empty(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = self.dropout(c)  # (bs, c_len, hid_size)
        q = self.dropout(q)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s
        
class QANetOutput(nn.Module):
    """Output layer used by QANet for question answering.
    Args:
        hidden_size (int): Hidden size used in the model.
    """
    def __init__(self, hidden_size):
        super(QANetOutput, self).__init__()
        
        self.proj01 = nn.Parameter(torch.empty(hidden_size, 1))
        nn.init.xavier_uniform_(self.proj01)
        self.proj11 = nn.Parameter(torch.empty(hidden_size, 1))
        nn.init.xavier_uniform_(self.proj11)
        self.proj02 = nn.Parameter(torch.empty(hidden_size, 1))
        nn.init.xavier_uniform_(self.proj02)
        self.proj22 = nn.Parameter(torch.empty(hidden_size, 1))
        nn.init.xavier_uniform_(self.proj22)
        
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.bias2 = nn.Parameter(torch.zeros(1))
    def forward(self, M0, M1, M2, is_pad):
    	"""
    	M0, M1, M2: tensors of shape (batch_size, text_len, hidden_size)
    	is_pad: tensor of shape(batch_size, text_len). Hold value TRUE for pad tokens.
    	
    	This method applies linear projections separately instead of concatenating the input tensors to save a little memory.
    	"""
    	A1 = torch.matmul(M0, self.proj01) + torch.matmul(M1, self.proj11) + self.bias1 ## shape (batch_size, text_len, 1)
    	A2 = torch.matmul(M0, self.proj02) + torch.matmul(M2, self.proj22) + self.bias2 ## shape (batch_size, text_len, 1)
    	
    	# Shapes: (batch_size, text_len)
    	log_p1 = masked_softmax(A1.squeeze(dim=2), is_pad, log_softmax=True)
    	log_p2 = masked_softmax(A2.squeeze(dim=2), is_pad, log_softmax=True)
    	
    	return log_p1, log_p2
