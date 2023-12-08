import torch
import torch.nn as nn
import numpy as np
from collections import deque
from .base import BaseBuffer
from src.common.train_utils import LinearScheduler
from einops import rearrange


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        # Initialize the buffer
        self.transitions = deque(maxlen=self.size)
        self.max = 1  # Initial max value to return (1 = 1^ω), default transition priority is set to max

     # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

     # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.transitions.append(data)
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    # Returns data given a data index
    def get(self, data_idxs):
        return [self.transitions[idx % self.size] for idx in data_idxs]

    def total(self):
        return self.sum_tree[0]


class PERBuffer(BaseBuffer):
    name = 'per_buffer'
    def __init__(self, size, n_step, gamma, prior_exp, prior_weight_scheduler, device):
        super().__init__()
        # Initialize
        self.size = size
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.prior_exp = prior_exp
        self.prior_weight_scheduler = LinearScheduler(**prior_weight_scheduler)

        self.num_in_buffer = 0
        self.n_step_transitions = deque(maxlen=self.n_step)
        self.transitions = SegmentTree(size)

    def _get_n_step_info(self):
        transitions = list(self.n_step_transitions)
        obs, action, _, _, _ = transitions[0]
        _, _, G, done, next_obs = transitions[-1]
        for _, _, _reward, _done, _next_obs in reversed(transitions[:-1]):
            G = _reward + self.gamma * G * (1-_done)    
            if _done:
                done, next_obs = _done, _next_obs

        action_list = []
        for _, act, _, _, _ in transitions:
            action_list.append(act)
        
        return (obs, action, action_list, G, done, next_obs)

    def store(self, obs, action, reward, done, next_obs):
        self.n_step_transitions.append((obs, action, reward, done, next_obs))
        if len(self.n_step_transitions) < self.n_step:
            return
        transition = self._get_n_step_info()
        # store new transition with maximum priority
        self.transitions.append(data=transition, value=self.transitions.max)
        self.num_in_buffer = min(self.num_in_buffer+1, self.size)

    # Returns a valid sample from each segment
    def _get_transitions_from_segments(self, batch_size):
        p_total = self.transitions.total() # sum of the priorities
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        # TODO: check whether loop hang in here
        while not valid:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts  # Uniformly sample from within all segments
            probs, idxs, tree_idxs = self.transitions.find(samples)  # Retrieve samples from tree with un-normalised probability
            if np.all(probs != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0
        # Retrieve all required transition data (from t - h to t + n)
        transitions = self.transitions.get(idxs)

        return tree_idxs, transitions, probs

    def sample(self, batch_size, mode='train'):
        if self.num_in_buffer < batch_size:
            assert('Replay buffer does not have enough transitions to sample')
        tree_idxs, transitions, probs = self._get_transitions_from_segments(batch_size)

        # encode transitions
        obs_batch, act_batch, act_list_batch, return_batch, done_batch, next_obs_batch = zip(*transitions)
        obs_batch = self.encode_obs(obs_batch)  
        act_batch = torch.LongTensor(act_batch).to(self.device)
        act_list_batch = torch.LongTensor(act_list_batch).to(self.device)
        return_batch = torch.FloatTensor(return_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        next_obs_batch = self.encode_obs(next_obs_batch)

        # compute importance weights
        p_total = self.transitions.total()
        N = self.num_in_buffer
        probs = probs / p_total

        # we don't need to update the priority weight in evaluation stage
        if mode == 'train':
            prior_weight = self.prior_weight_scheduler.get_value()
        else:
            prior_weight = 1.0

        weights = (1 / (probs * N) + 1e-5) ** prior_weight # importance sample weights
        # re-normalise by max weight (make update scale consistent w.r.t learning rate)
        weights = weights / max(weights)
        weights = torch.FloatTensor(weights).to(self.device)

        batch = {
            'idxs': tree_idxs,
            'obs': obs_batch,
            'act': act_batch,
            'act_list_batch': act_list_batch,
            'return': return_batch,
            'done': done_batch,
            'next_obs': next_obs_batch,
            'weights': weights,
            'prior_weight': prior_weight
        }
        
        return batch

    def encode_obs(self, obs, prediction=False):
        obs = np.array(obs).astype(np.float32)
        obs = obs / 255.0

        # prediction: batch-size: 1
        if prediction:
            obs = np.expand_dims(obs, 0)

        # in current form, time-step is fixed to 1
        obs = np.expand_dims(obs, 1)

        # n: batch_size
        # t: 1
        # f: frame_stack
        # c: channel (atari: 1, dmc: 3)
        # h: height
        # w: width
        n, t, f, c, h, w = obs.shape
        obs = torch.FloatTensor(obs).to(self.device)

        return obs

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.prior_exp)
        self.transitions.update(idxs, priorities)
