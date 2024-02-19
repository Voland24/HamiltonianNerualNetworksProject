import os, sys
from urllib.request import urlretrieve
import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

from helper_methods import read_lipson, str2array

def get_dataset(experiment_name, save_dir, test_split=0.8):
  if experiment_name == "pend-sim":
    dataset_name = "pendulum_h_1"
  elif experiment_name == "pend-real":
    dataset_name = "real_pend_h_1"
  else:
    assert experiment_name in ['sim', 'real']

  data_str = read_lipson(dataset_name, save_dir)
  state, names = str2array(data_str)

  # put data in a dictionary structure
  data = {k: state[:,i:i+1] for i, k in enumerate(names)}
  data['x'] = state[:,2:4]
  data['dx'] = (data['x'][1:] - data['x'][:-1]) / (data['t'][1:] - data['t'][:-1])
  data['x'] = data['x'][:-1]

  # make a train/test split while preserving order of data
  # there's no great way to do this.
  # here we just put the test set in the middle of the sequence
  train_set_size = int(len(data['x']) * test_split)
  test_set_size = int(len(data['x']) * (1-test_split))
  test_start_ix = train_set_size#int(train_set_size/2)
  a = test_start_ix
  b = test_start_ix + test_set_size

  split_data = {}
  for k, v in data.items():
    split_data[k] = np.concatenate([v[:a],v[b:]], axis=0)
    split_data['test_' + k] = v[a:b]
  data = split_data
  return data

def hamiltonian_fn(coords):
  k = 2.4  # this coefficient must be fit to the data
  q, p = np.split(coords,2)
  H = k*(1-np.cos(q)) + p**2 # pendulum hamiltonian
  return H

def dynamics_fn(t, coords):
  dcoords = autograd.grad(hamiltonian_fn)(coords)
  dqdt, dpdt = np.split(dcoords,2)
  S = -np.concatenate([dpdt, -dqdt], axis=-1)
  return S