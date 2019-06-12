import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import dataset as dataset
import datapreparation as datp
import numpy as np
import random
import copy
import sys

class Generalist(nn.Module):

    def __init__(self, input_size, hidden_size, num_tags, n_layers=2):

        super(Generalist, self).__init__()
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.n_layers = n_layers
        self.dropout = nn.Dropout(0.2)

        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.notes_decoder = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.lstm = nn.GRU(hidden_size, hidden_size, n_layers)
        self.output = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, input_sequence, tag, hidden=None, device='cuda', sigmoid=False):

        if hidden is None:
            hidden = torch.zeros(self.n_layers, 1, self.hidden_size).to(device=device)

        input_sequence = self.dropout(input_sequence)
        lstm_out, hidden = self.lstm(self.notes_encoder(input_sequence), hidden)

        ### TEST
        lstm_out = self.relu(lstm_out)
        lstm_out = self.notes_decoder(lstm_out)

        if sigmoid:
            lstm_out = self.output(lstm_out)
        ###


        #lstm_out = self.output(self.notes_decoder(lstm_out))

        return lstm_out, hidden

def printcsv(output, name):
    if output.dim() == 3:
        output = torch.t(output.squeeze(1)).cpu()
    nparray = output.detach().numpy()
    np.savetxt(str(name) + '.csv', nparray, fmt='%.10f')

### Main training loop



def train_sequence(model, num_epochs, data, optimizer, loss_log):

    device='cpu'
    gpu = torch.cuda.is_available()


    if (gpu):
        model.cuda()
        device='cuda'
    model.train()

    ## Loss functions
    #loss_fn = torch.nn.BCELoss()
    ## Out of 128 notes, expect 124 zeroes and 4 ones = 124 / 4 = 31 weight
    pos_weight = torch.FloatTensor([31])
    pos_weight = pos_weight.to(device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("BCE")
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.SmoothL1Loss()

    ## optimizer
    if(len(loss_log) != 0):
        best_previous_loss = min(loss_log)
        best_epoch_num = loss_log.index(min(loss_log))
        print('Previous loss: ', best_previous_loss, 'in epoch ', best_epoch_num)
    else:
        best_previous_loss = 1000.0
        best_epoch_num = 0

    previous_epoch = best_epoch_num
    best_weights = copy.deepcopy(model.state_dict())
    best_opt = copy.deepcopy(optimizer.state_dict())
    best_loss = best_previous_loss

    epochs_since_improvement = 0

    for epoch in range(num_epochs):

        total_loss = 0
        print('Epoch: {}'.format(epoch+previous_epoch))

        for x in random.sample(range(0, len(data)), len(data)):

            input_tensor, tag_tensor, output_tensor = data[x]
            #Zero gradients before backpropagating
            model.zero_grad()

            prediction, hidden = model(input_tensor.to(device=device), tag_tensor.to(device=device), device=device)
            prediction = prediction.transpose(1,2)
            output_tensor = output_tensor.transpose(1,2)

            loss = loss_fn(prediction, output_tensor.to(device=device))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss = total_loss
        print('Epoch loss: {:6.4f}'.format(epoch_loss))
        loss_log.append(epoch_loss)

        if epoch_loss < best_loss:
            best_weights = copy.deepcopy(model.state_dict())
            best_opt = copy.deepcopy(optimizer.state_dict())
            best_loss = epoch_loss
            best_epoch_num = epoch
            best_loss_log = loss_log
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement == 500:
            break

    print('Loss at start: ', best_previous_loss)
    print('Loss at end: ', best_loss)

    # Load best
    print('Best loss: ', best_loss, ' in epoch', previous_epoch + best_epoch_num)
    model.load_state_dict(best_weights)
    optimizer.load_state_dict(best_opt)
    loss_log = best_loss_log

    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'loss_log':loss_log}
    return state

#def load(model, optimizer, filename, device='cuda'):
def load(model, filename, device='cuda'):
    sys.setrecursionlimit(10000)
    if device == 'cuda':
        print("CUDA")
        state = torch.load(filename)
        model.load_state_dict(state['state_dict'])
        model.cuda()
    else:
        print("Not CUDA")
        state = torch.load(filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(state['state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(state['optimizer'])
    model.loss_log = state['loss_log']
    return model, optimizer

def gen_music_seconds_smooth(model,init,composer=0,fs=5,gen_seconds=10,init_seconds=5, device='cpu'):
    model.eval()
    model.cpu()
    init = init.cpu()
    init_index = int(init_seconds*fs)
    tag = torch.LongTensor([composer]).unsqueeze(1).to(device=device)
    song, song_raw = generate_smooth(model,tag,(gen_seconds-init_seconds+1)*fs,init[1:(init_index+1)], device)
    res= ( song.squeeze(1).detach().to(device='cpu').numpy()).astype(float).T
    song_raw =  ( song_raw.squeeze(1).detach().to(device='cpu').numpy()).astype(float).T

    datp.visualize_piano_roll(res,fs)
    return datp.embed_play_v1(res, song_raw, fs)

def generate_smooth(model,tag,n,init, device):
    res = init
    res_raw = init
    hidden = None

    for i in range(n):
        init_new,hidden = model.forward(init,tag,hidden, device, True)
        init_new = init_new[-1:]
        init_raw = init_new

        ## Choose top 4 suggested values as new input
        init_top, indices = torch.topk(init_new, 4)
        init_new = torch.zeros_like(init_new)
        for index in indices:
            init_new[0][0][index] = 1


        res_raw = torch.cat((res_raw, init_raw))
        res = torch.cat ( ( res, init_new ) )
        init = torch.cat( (init[1:], init_new) )
    return res, res_raw
