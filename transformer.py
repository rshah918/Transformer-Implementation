import torch
import torch.nn as nn
import math

'''
Implementation of a decoder-only Transformer architecture in Pytorch
'''
class Attention(nn.Module):
    def __init__(self, len_embedding, head_dimension):
        super().__init__()
        self.len_embedding = len_embedding
        # weights for Q,K,V 
        # in the case of multi-head attention, we are basically doing a compression of the input seq to size head_dimension
        self.q_linear = nn.Linear(len_embedding, head_dimension)
        self.k_linear = nn.Linear(len_embedding, head_dimension)
        self.v_linear = nn.Linear(len_embedding, head_dimension)

    def forward(self, sequence, mask=False):
        # input shape (seq, emb)
        Q = self.q_linear(sequence)
        V = self.v_linear(sequence)
        K = self.k_linear(sequence)
        attention_scores = torch.matmul(Q,K.transpose(0,-1)) # shape is (seq,n_emb)*(n_emb,seq) = (seq,seq)
        attention_scores = attention_scores / math.sqrt(self.len_embedding)
        if mask:
            #used during traning. Make sure each token only attends to prior tokens 
            attention_scores = torch.triu(attention_scores)
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_scores, V)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, len_embedding):
        super(MultiHeadAttention, self).__init__()
        if len_embedding % num_heads != 0:
            print("Number of attention heads has to be divisible by the embedding dimension!")
            exit(1)
        self.num_heads = num_heads
        self.head_dimension = len_embedding // num_heads
        self.attention_layers = [Attention(len_embedding, self.head_dimension) for i in range(num_heads)]
        self.final_linear = nn.Linear(len_embedding, len_embedding)
    
    def forward(self, sequence, mask=False):
        attention_layer_outputs = []
        for attention_layer in self.attention_layers:
            attention_layer_outputs.append(attention_layer.forward(sequence, mask))
        concatenated_heads = torch.cat(attention_layer_outputs, dim=-1) #concat along embedding dimension
        out = self.final_linear(concatenated_heads)
        return out

class LayerNorm(nn.Module):
    def __init__(self, len_embedding):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(len_embedding))
        self.beta = nn.Parameter(torch.zeros(len_embedding))
        self.eps = 0.000001

    def forward(self, x):
        mean = torch.mean(x, -1, keepdim=True) # shape=(len_seq, 1)
        std = torch.std(x, -1, keepdim=True) # shape = (len_seq, 1)
        # 1: subract each element by the mean along the embedding dim to set the average to 0.
        # 2: Divide by the standard deviation to set the variance to 1
        # 3: gamma and beta are trainable parameters. The NN  updates gamma to choose a more appropriate scale for variance
            # and updates beta to choose a more appropriate offset for the mean.
        return (self.gamma * ((x-mean) / (std + self.eps))) + self.beta
    

class FeedForward(nn.Module):
    '''
    from "Attention is All You Need": FFN(x) = max(0, xW1 + b1)W2 + b2
    - Equation implies nonlinear L1 w/ Relu activation and linear L2 
    - x.shape = (len_sequence, len_embedding)
    - W1.shape (len_embedding, len_feedforward)
    - W2.shape (len_feedforward, len_embedding)

    len_feedforward is an arbitrary constant that is larger than len_embedding. We are effectively performing an expansion and 
        contraction back to the original embedding dimensionality
    '''
    def __init__(self, len_embedding, len_feedforward):
        super(FeedForward, self).__init__()
        self.layer_1 = nn.Linear(len_embedding, len_feedforward)
        self.layer_2 = nn.Linear(len_feedforward, len_embedding)

    def forward(self,x):
        l1 = nn.functional.relu(self.layer_1(x))
        l2 = self.layer_2(l1)
        return l2
        

class DecoderLayer(nn.Module):
    def __init__(self, num_attention_heads, len_embedding, len_sequence):
        super(DecoderLayer, self).__init__()
        self.multihead_self_attention = MultiHeadAttention(num_attention_heads, len_embedding)
        self.layernorm1 = LayerNorm(len_embedding)
        self.layernorm2 = LayerNorm(len_embedding)
        self.feedforward = FeedForward(len_embedding, len_sequence)

    def forward(self, x):
        # 1: MultiHead Self Attention
        self_attention = self.multihead_self_attention.forward(x)
        # 2: Skip connection 
        skip_connection = self_attention + x
        # 3: Layer Normalization
        layernorm_1 = self.layernorm1.forward(skip_connection)
        # 4: Feedforward layer
        feedforward = self.feedforward(layernorm_1)
        # 5: Another skip connection
        skip_connection = feedforward + layernorm_1
        # 6: More Layer Nornalization
        out = self.layernorm2(skip_connection)
        return out
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, num_decoder_layers, num_attention_heads, len_embedding, len_sequence):
        super(Decoder, self).__init__()
        # chain together mutlipel decoder layers
        self.decoder_layers = nn.Sequential(*[DecoderLayer(num_attention_heads, len_embedding, len_sequence) for i in range(num_decoder_layers)])
        # output layer creates a probability distribution across your vocabulary for each token in the input sequence
        self.output_layer = nn.Linear(len_embedding, vocab_size)
    def forward(self, x):
        decoder_layers_output = self.decoder_layers.forward(x)
        logits = self.output_layer(decoder_layers_output) #shape [seq_length, vocab_size]
        # convert raw output to a probability distribution across the vocabulary at each position
        probabilities = nn.functional.softmax(logits, dim=-1)
        # To get the next token prediction, take the probability distribution at the last position of the sequence
        return probabilities[-1]



a = Decoder(5, 1, 5,10,3)
input = torch.empty([3,10])
out = a.forward(input)
print(out.shape)
