#!/usr/bin/env python3

import numpy as np
import torch
import json
import time

class ExperienceMsg:
    """Classe utilitaire pour encoder/décoder les messages d'expérience"""
    
    @staticmethod
    def encode_experience(robot_id, mineral_map, position, action, reward, 
                          done, global_state=None):
        """Encode une expérience en JSON"""
        data = {
            'robot_id': robot_id,
            'step_data': {
                'mineral_map': mineral_map.tolist() if isinstance(mineral_map, np.ndarray) else mineral_map,
                'position': position.tolist() if isinstance(position, np.ndarray) else position,
                'action': int(action),
                'reward': float(reward),
                'done': bool(done),
                'timestamp': time.time()
            }
        }
        if global_state is not None:
            data['step_data']['global_state'] = (
                global_state.tolist() if isinstance(global_state, np.ndarray) 
                else global_state
            )
        
        return json.dumps(data)
    
    @staticmethod
    def decode_experience(msg_str):
        """Décode une expérience depuis JSON"""
        return json.loads(msg_str)

def normalize_observation(obs, mean=None, std=None):
    """Normalise une observation"""
    if mean is None:
        mean = np.mean(obs)
    if std is None:
        std = np.std(obs) + 1e-8
    
    return (obs - mean) / std

def one_hot_encode(action, num_actions):
    """One-hot encode une action"""
    vec = np.zeros(num_actions)
    vec[action] = 1
    return vec

def compute_returns(rewards, gamma):
    """Calcule les retours cumulés"""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns