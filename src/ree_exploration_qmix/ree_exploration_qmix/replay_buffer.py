#!/usr/bin/env python3

import numpy as np
import torch
from collections import deque
import random

class QMIXReplayBuffer:
    """Replay buffer pour QMIX qui stocke des épisodes complets"""
    
    def __init__(self, buffer_size, batch_size, num_robots, device='cpu'):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_robots = num_robots
        self.device = device
        
        # Buffer circulaire d'épisodes
        self.episodes = deque(maxlen=buffer_size)
        
    def add_episode(self, episode_data):
        """
        Ajoute un épisode complet au buffer
        
        episode_data: dict avec
            - 'mineral_maps': list of arrays (T, C, H, W) pour chaque robot
            - 'positions': list of arrays (T, 2) pour chaque robot
            - 'actions': list of arrays (T,) pour chaque robot
            - 'rewards': list of arrays (T,) pour chaque robot
            - 'global_states': array (T, C, H, W)
            - 'dones': array (T,)
            - 'avail_actions': optional list of arrays (T, num_actions)
        """
        self.episodes.append(episode_data)
        
    def sample_batch(self):
        """Échantillonne un batch d'épisodes"""
        batch_size = min(self.batch_size, len(self.episodes))
        episodes = random.sample(self.episodes, batch_size)
        
        # Préparer les données batch
        batch_data = self._prepare_batch(episodes)
        return batch_data
    
    def _pad_to(self, arr, target_len, pad_value=0):
        """Pad ou tronque un numpy array à target_len sur l'axe 0"""
        arr = np.array(arr)
        cur_len = len(arr)
        if cur_len >= target_len:
            return arr[:target_len]
        pad_shape = [(0, target_len - cur_len)] + [(0, 0)] * (arr.ndim - 1)
        return np.pad(arr, pad_shape, mode='constant', constant_values=pad_value)

    def _prepare_batch(self, episodes):
        """Prépare un batch d'épisodes pour l'entraînement"""
        batch_size = len(episodes)

        # Longueur max sur TOUTES les données (global_states ET mineral_maps)
        max_len = max(
            max(len(np.array(ep['global_states'])) for ep in episodes),
            max(len(np.array(ep['mineral_maps'][i]))
                for ep in episodes for i in range(self.num_robots))
        )

        device = self.device

        mineral_maps = [[] for _ in range(self.num_robots)]
        positions = [[] for _ in range(self.num_robots)]
        actions = [[] for _ in range(self.num_robots)]
        rewards = [[] for _ in range(self.num_robots)]

        global_states = []
        next_global_states = []
        dones = []

        for ep in episodes:
            for i in range(self.num_robots):
                maps = self._pad_to(ep['mineral_maps'][i], max_len)
                pos  = self._pad_to(ep['positions'][i],    max_len)
                act  = self._pad_to(ep['actions'][i],      max_len, pad_value=-1)
                rew  = self._pad_to(ep['rewards'][i],      max_len)

                mineral_maps[i].append(torch.FloatTensor(maps).to(device))
                positions[i].append(torch.FloatTensor(pos).to(device))
                actions[i].append(torch.LongTensor(act).to(device))
                rewards[i].append(torch.FloatTensor(rew).to(device))

            states = np.array(ep['global_states'])
            if 'next_global_states' in ep:
                next_states = np.array(ep['next_global_states'])
            else:
                next_states = np.zeros_like(states)
                next_states[:-1] = states[1:]

            global_states.append(torch.FloatTensor(self._pad_to(states, max_len)).to(device))
            next_global_states.append(torch.FloatTensor(self._pad_to(next_states, max_len)).to(device))

            ep_dones = self._pad_to(ep['dones'], max_len, pad_value=1)
            dones.append(torch.BoolTensor(ep_dones).to(device))
        
        # Empiler
        batch_data = {
            'mineral_maps': [torch.stack(maps, dim=1) for maps in mineral_maps],  # (T, B, C, H, W)
            'positions': [torch.stack(pos, dim=1) for pos in positions],  # (T, B, 2)
            'actions': [torch.stack(act, dim=1) for act in actions],  # (T, B)
            'rewards': [torch.stack(rew, dim=1) for rew in rewards],  # (T, B)
            'global_states': torch.stack(global_states, dim=1),  # (T, B, C, H, W)
            'next_global_states': torch.stack(next_global_states, dim=1),  # (T, B, C, H, W)
            'dones': torch.stack(dones, dim=1)  # (T, B)
        }
        
        return batch_data
    
    def __len__(self):
        return len(self.episodes)