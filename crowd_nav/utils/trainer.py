import logging
from telnetlib import X3PAD
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import random

class Trainer(object):
    def __init__(self, model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None

    def set_learning_rate(self, learning_rate):
        self.model.train()
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            for data in self.data_loader:
                inputs, values = data
                inputs = Variable(inputs)
                values = Variable(values)
                # TODO: May need to add other loss
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()

            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        for _ in tqdm(range(num_batches)):
            inputs, values = next(iter(self.data_loader))
            inputs = Variable(inputs)
            values = Variable(values)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss

class LiliTrainer(object):
    def __init__(self, model, memory, traj_memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.Q_criterion = nn.MSELoss().to(device)
        self.rep_criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.traj_memory = traj_memory
        self.data_loader = None
        self.traj_data_loader = None
        self.batch_size = batch_size
        self.optimizer = None  

    def set_learning_rate(self, Q_learning_rate, dec_learning_rate=1e-3):
        logging.info('Current learning rate: %f %f', Q_learning_rate, dec_learning_rate)
        self.model.train()
        if not self.model.lili_flag:
            self.Q_optimizer = optim.Adam([param for param in self.model.phi_e.parameters()] + 
                                       [param for param in self.model.psi_h.parameters()] + 
                                       [param for param in self.model.attention.parameters()] + 
                                       [param for param in self.model.Q.parameters()], lr=Q_learning_rate)
        else:
            self.Q_optimizer = optim.Adam([param for param in self.model.Q.parameters()], lr=Q_learning_rate)
     
        self.enc_optimizer = optim.Adam(self.model.encoder.parameters(), lr=Q_learning_rate)
        self.dec_optimizer = optim.Adam(self.model.decoder.parameters(), lr=dec_learning_rate)

    def optimize_epoch(self, num_epochs):
        self.model.train()
        if self.Q_optimizer is None or self.enc_optimizer is None or self.dec_optimizer is None:
            raise ValueError('Learning rate is not set!')
        logging.info(f'Memory size: {len(self.memory)}. Traj memory size: {len(self.traj_memory)}')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        if self.traj_data_loader is None:
            self.traj_data_loader = DataLoader(self.traj_memory, self.batch_size, shuffle=True) 
        average_epoch_loss = 0
        for epoch in tqdm(range(num_epochs)):
            epoch_Q_loss = 0
            epoch_rep_loss = 0
            for data in self.data_loader:  #  states, values) 
                prev_traj, traj = next(iter(self.traj_data_loader))
                states, values = data
                
                prev_traj = prev_traj.to(self.device)
                traj = traj.to(self.device)
                states = states.to(self.device)
                values = values.to(self.device)

                
                target_states = traj[:, :, :self.model.num_humans*self.model.input_dim]
                target_rewards = traj[:, :, -2].unsqueeze(-1)
  
                target_traj = torch.reshape(torch.cat([target_states, target_rewards], dim=-1), (target_states.shape[0], -1))
            
                state_inputs = Variable(states)
                traj_inputs = Variable(prev_traj) # previous traj
                values = Variable(values)
                # logging.info(f'Check optim_epoch: state_inputs: {state_inputs.shape}, traj_inputs: {traj_inputs.shape}')
                self.Q_optimizer.zero_grad()
                self.enc_optimizer.zero_grad()
                self.dec_optimizer.zero_grad()
                min_size = min(state_inputs.shape[0], traj_inputs.shape[0])
                Q_hat, traj_hat = self.model(state_inputs[:min_size, :, :].to(self.device), traj_inputs[:min_size, :, :].to(self.device))
                Q_loss = self.Q_criterion(torch.amax(Q_hat,-1).unsqueeze(-1), values[:min_size])
                rep_loss = 0.05* self.rep_criterion(traj_hat[:min_size,:-self.model.hist], target_traj[:min_size,:-self.model.hist]) \
                           + self.rep_criterion(traj_hat[:min_size,-self.model.hist:], target_traj[:min_size,-self.model.hist:])
                # print(f'Q_loss: {Q_loss}')
                # print(f'Rep Loss: {rep_loss}')
                Q_loss.backward(retain_graph=True)
                rep_loss.backward()
           
                self.Q_optimizer.step()
                self.enc_optimizer.step()
                self.dec_optimizer.step()

                epoch_Q_loss += Q_loss.data.item()
                epoch_rep_loss += rep_loss.data.item()

            average_epoch_Q_loss = epoch_Q_loss / len(self.memory)
            average_epoch_rep_loss = epoch_rep_loss / len(self.memory)
            logging.info('Average Q loss in epoch %d: %.2E', epoch, average_epoch_Q_loss)
            logging.info('Average Rep loss in epoch %d: %.2E', epoch, average_epoch_rep_loss)
        return average_epoch_Q_loss, average_epoch_rep_loss,

    def optimize_batch(self, num_batches):
        self.model.train()
        if self.Q_optimizer is None or self.enc_optimizer is None or self.dec_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True, drop_last=True)
        if self.traj_data_loader is None:
            self.traj_data_loader = DataLoader(self.traj_memory, self.batch_size, shuffle=True, drop_last = True) 
        Q_losses = 0
        rep_losses = 0
        for idx in tqdm(range(num_batches)):
            prev_traj, traj = next(iter(self.traj_data_loader))
            
            states, values = next(iter(self.data_loader))

            prev_traj = prev_traj.to(self.device)
            traj = traj.to(self.device)
            states = states.to(self.device)
            values = values.to(self.device)

            target_states = traj[:, :, :self.model.num_humans*self.model.input_dim]
            target_rewards = traj[:, :, -2].unsqueeze(-1)
            target_traj = torch.reshape(torch.cat([target_states, target_rewards], dim=-1), (target_states.shape[0], -1))
            
            state_inputs = Variable(states)
            traj_inputs = Variable(prev_traj) # previous traj
            values = Variable(values)

            self.Q_optimizer.zero_grad()
            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            self.Q_optimizer.zero_grad()
            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()
            min_size = min(state_inputs.shape[0], traj_inputs.shape[0])
            if min_size<=1:
                    logging.info("min_size<= 1!")
            Q_hat, traj_hat = self.model(state_inputs[:min_size, :, :].to(self.device), traj_inputs[:min_size, :, :].to(self.device))
            Q_loss = self.Q_criterion(torch.amax(Q_hat,-1).unsqueeze(-1), values[:min_size])
            rep_loss = 0.05* self.rep_criterion(traj_hat[:min_size,:-self.model.hist], target_traj[:min_size,:-self.model.hist]) \
                           + self.rep_criterion(traj_hat[:min_size,-self.model.hist:], target_traj[:min_size,-self.model.hist:])
               
            Q_loss.backward(retain_graph=True)
            rep_loss.backward()
           
            self.Q_optimizer.step()
            self.enc_optimizer.step()
            self.dec_optimizer.step()
            
            Q_losses += Q_loss.data.item()
            rep_losses += rep_loss.data.item()

        average_Q_loss = Q_losses / num_batches
        average_rep_loss = rep_losses / num_batches
        logging.info('Average Q loss : %.2E', average_Q_loss)
        logging.info('Average Rep loss : %.2E', average_rep_loss)

        return average_Q_loss, average_rep_loss
