import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import dataset as dataset
import datapreparation as datp
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_tags, gpu, n_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_tags = num_tags
        self.gpu = gpu
        self.n_layers = n_layers

        self.embedding  = nn.Embedding(num_tags, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.notes_decoder = nn.Linear(in_features=hidden_size, out_features=input_size)



    def forward(self, input, hidden=None):
        gpu = torch.cuda.is_available()
        if hidden is None:
            if gpu:
                hidden = torch.zeros(self.n_layers, 1, self.hidden_size).cuda()
            else:
                hidden = torch.zeros(self.n_layers, 1, self.hidden_size)

        #input = self.notes_encoder(input)
        output, hidden = self.gru(input, hidden)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, input_size, n_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.n_layers = n_layers
        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.notes_decoder = nn.Linear(in_features=hidden_size, out_features=input_size)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden=None):
        gpu = torch.cuda.is_available()
        if hidden is None:
            if gpu:
                hidden = torch.zeros(self.n_layers, 1, self.hidden_size).cuda()
            else:
                hidden = torch.zeros(self.n_layers, 1, self.hidden_size)


        input = self.notes_encoder(input)

        output, hidden = self.gru(input, hidden)

        output = self.notes_decoder(input)
        output = self.sigmoid(output)
        return output, hidden

data = dataset.pianoroll_dataset_batch("C:\DeepLearning\\neural-composer-assignement\datasets\\training\\piano_roll_fs5")

gpu=torch.cuda.is_available()
encoder = Encoder(int(data.num_keys()), 128, int(data.num_tags()), gpu)
decoder = Decoder(int(data.num_keys()), 128)
if(gpu):
    encoder.cuda()
    decoder.cuda()

# Loss functions
loss_fn = torch.nn.BCELoss()
#loss_fn = torch.nn.SmoothL1Loss()

# Optimizers
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.003)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.003)
# optimizer = optim.SGD(model.parameters(), lr=0.01)


## ONE SONG AT A TIME
num_epochs = 50
teacher_forcing = 0.00


def train_sequence(teacher_forcing, num_epochs):

    for x in range(num_epochs):

        print('Epoch: {}'.format(x))

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # for x in random.sample(range(0, 42), 10):
        loss = 0
        song_no = 0
        input_tensor, tag_tensor, output_tensor = data[song_no]
        if (gpu):
            song = output_tensor.cuda()
            tag = tag_tensor.cuda()
        else:
            song = output_tensor
            tag = tag_tensor

        input_length = output_tensor.shape[0] - 1

        for x in range(1):

            encoder_hidden = None
            encoder_outputs = None
            for timestep in range(input_length-1):
                encoder_output, encoder_hidden = encoder(song[timestep].unsqueeze(1), encoder_hidden)
                if timestep == 0:
                    encoder_outputs = encoder_output
                else:
                    encoder_outputs = torch.cat((encoder_outputs, encoder_output))

            decoder_hidden = encoder_hidden
            decoder_input = encoder_outputs[0].unsqueeze(1)

            for timestep in range(input_length -1):

                if random.random() < teacher_forcing:
                    use_teacher_forcing = False
                else:
                    use_teacher_forcing = True

                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                if use_teacher_forcing:
                    decoder_input = song[timestep + 1].unsqueeze(1)
                    loss += loss_fn(decoder_output.squeeze(1), song[timestep + 1])
                else:
                    decoder_input = torch.round(decoder_output/torch.max(decoder_output))
                    loss += loss_fn(decoder_output.squeeze(1), song[timestep + 1])
                    if timestep % 50 == 0:
                        print("output", decoder_output)

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            print('Loss: {:6.4f}'.format(loss.item() / input_length))

            teacher_forcing += 1 / num_epochs

    dir_path = os.path.dirname(os.path.realpath(__file__))

    torch.save(encoder.state_dict(), dir_path + '\\encoder.pth')
    torch.save(decoder.state_dict(), dir_path + '\\decoder.pth')


def generate_music():
    with torch.no_grad():
        matches, total = 0, 0
        hidden = None
        for x in range(1):

            input_tensor, tag_tensor, output_tensor = data[x]
            if (gpu):
                song = output_tensor.cuda()
                tag = tag_tensor.cuda()
            else:
                song = output_tensor
                tag = tag_tensor
            if gpu:
                play, song_prediction, out_values = gen_music(encoder, decoder, 100, init=data[0][2][0:5].cuda(), composer=0,
                                                          fs=5)
            else:
                play, song_prediction, out_values = gen_music(encoder, decoder, 100, init=data[0][2][0:5],
                                                              composer=0,
                                                              fs=5)

def generate_round(encoder, decoder, tag, n, k=1, init=None):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if (init is None):
        init = torch.zeros(size=(k, 1, encoder.input_size), device=device)
    else:
        k = init.shape[0]

    res = init
    out_unrounded = init
    hidden = None

    #Encode sequence, use encoder hidden as decoder hidden
    print("First input", init)
    init, encoder_hidden = encoder.forward(init, hidden)


    print("Encoder hidden seed", encoder_hidden)
    print(res)


    decoder_output, decoder_hidden = decoder.forward(init, encoder_hidden)
    decoder_input = decoder_output[-1].unsqueeze(1)
    #Let the decoder generate a new sequence
    for i in range(n):
        decoder_output, decoder_hidden = decoder.forward(decoder_input, decoder_hidden)
        print("Output", i, (decoder_output / torch.max(decoder_output)))

        # For keeping track of both outputs
        out_unrounded = torch.cat((out_unrounded, decoder_output))
        decoder_input = torch.round(decoder_output / torch.max(decoder_output))
        print("Input", decoder_input)
        res = torch.cat((res, decoder_input))

    return res, out_unrounded

def generate_smooth(encoder, decoder, tag, n, init):
    res = init
    lstm_out = init
    hidden = None
    encoder_output, encoder_hidden = encoder.forward(init, tag, hidden)
    decoder_hidden = encoder_hidden
    for i in range(n):
        decoder_output, decoder_hidden = decoder.forward(init, tag, hidden)
        init_new = init_new[-1:]
        lstm_out = init_new[-1:]
        init_new = torch.round(init_new / torch.max(init_new))
        res = torch.cat((res, init_new))
        init = torch.cat((init[1:], init_new))
        lstm_out = torch.cat((init[1:], init_new))
    return res, lstm_out


def gen_music(encoder, decoder, length=1000, init=None, composer=0, fs=5):
    device = 'cpu'
    if (init is None):
        song, lstm_out = generate_round(encoder, decoder, torch.LongTensor([composer], device=device).unsqueeze(1), length, 1)
    else:
        song, lstm_out = generate_round(encoder, decoder, torch.LongTensor([composer], device=device).unsqueeze(1), length, 1, init)
    res = (song.squeeze(1).detach().cpu().numpy()).astype(int).T
    out = (lstm_out.squeeze(1).detach().cpu().numpy()).astype(int).T
    datp.visualize_piano_roll(res, fs)
    return datp.embed_play_v1(res, fs), res, lstm_out

def normalize(x):
    x_normed = nn.functional.normalize(x, p=2, dim=2, eps=1e-12)
    return x_normed

train_sequence(0, num_epochs)
generate_music()

