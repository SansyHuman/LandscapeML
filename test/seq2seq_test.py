import csv
import os.path
from common.utils import *
import math
import json
from common.sci_parser import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, 1)
        self.Wb = nn.Linear(hidden_size, hidden_size)
        self.Wc = nn.Linear(hidden_size, hidden_size)

    def forward(self, encoder_hidden, decoder_hidden):
        score = self.Wa(F.tanh(torch.add(self.Wb(decoder_hidden), self.Wc(encoder_hidden))))
        score = score.squeeze(2)

        weight = F.softmax(score, dim=-1)
        weight = weight.unsqueeze(1)

        context = torch.bmm(weight, encoder_hidden)
        return context


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, input):
        output, (hidden, _) = self.rnn(input)
        return output, hidden


SOS_token = 0


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, output_size: int, length: int, device: torch.device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.length = length
        self.device = device

        self.attention = BahdanauAttention(hidden_size)
        self.rnn = nn.RNN(output_size + hidden_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, self.output_size, device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing 포함: 목표를 다음 입력으로 전달
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
                decoder_input = decoder_output

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden, encoder_outputs):
        decoder_hidden = hidden[-1,:,:].unsqueeze(0).permute(1,0,2)
        context = self.attention(encoder_outputs, decoder_hidden)

        input_rnn = torch.cat((input, context), dim=2)
        output, hidden = self.rnn(input_rnn, hidden)
        output = self.out(output)

        return output, hidden


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_hidden = torch.zeros(32, 4, 9)
decoder_hidden = torch.zeros(32, 1, 9)
attention = BahdanauAttention(9)

print(attention(encoder_hidden, decoder_hidden).shape)

input_t = torch.zeros(32, 4, 3).to(device)
encoder = EncoderRNN(3, 9, 3).to(device)
decoder = DecoderRNN(9, 3, 1, 4, device).to(device)

encoder_outputs, encoder_hidden = encoder(input_t)
decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)
print(decoder_outputs.shape)