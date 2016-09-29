import numpy as np
import cPickle
import heapq
import os
from IPython.core.debugger import Tracer
import scipy.io as scio
import time
import re


def tic_toc_print(interval, string):
  global tic_toc_print_time_old
  if 'tic_toc_print_time_old' not in globals():
    tic_toc_print_time_old = time.time()
    print string
  else:
    new_time = time.time()
    if new_time - tic_toc_print_time_old > interval:
      tic_toc_print_time_old = new_time;
      print string

def mkdir_if_missing(output_dir):
  """
  def mkdir_if_missing(output_dir)
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_variables(pickle_file_name, var, info, overwrite = False):
  """
    def save_variables(pickle_file_name, var, info, overwrite = False)
  """
  if os.path.exists(pickle_file_name) and overwrite == False:
    raise Exception('{:s} exists and over write is false.'.format(pickle_file_name))
  # Construct the dictionary
  assert(type(var) == list); assert(type(info) == list);
  d = {}
  for i in xrange(len(var)):
    d[info[i]] = var[i]
  with open(pickle_file_name, 'wb') as f:
    cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)

def load_variables(pickle_file_name):
  """
  d = load_variables(pickle_file_name)
  Output:
    d     is a dictionary of variables stored in the pickle file.
  """
  if os.path.exists(pickle_file_name):
    with open(pickle_file_name, 'rb') as f:
      d = cPickle.load(f)
    return d
  else:
    raise Exception('{:s} does not exists.'.format(pickle_file_name))

def det2cap(map_file, vocab):
    """Retrieve mapping of detection labels to caption labels.
    The probability of detection category is the max of the max probability
    of its caption labels.

    Args:
        map_file: manual mapping provided by Ishan Misra
        vocab: 1000 common concepts in MSCOCO caption_vocabs

    Returns:
        dict of (det_lab : cap index)
    """
    # construct dict of (vocab : indices)
    cap_ind_dict = {}
    for ind, word in enumerate(vocab['words']):
        cap_ind_dict[word] = ind
    # retrieve manual mapping of 1000 caption labels to 73 detection labels
    # mapping_file = open('dataset/mscoco_vc/coco2vocab_manual_mapping.txt', 'r')
    assert(os.path.isfile(map_file))
    mapping_file = open(map_file, 'r')
    mapping = mapping_file.read()
    # split by "\n"
    mapping = re.split('\n', mapping)[:-1]
    # mapping is formatted as 'a: a1, a2, a3\r', then we split it by ':,\r'
    mapping_dict = {}
    for vocab_map in mapping:
        vocab_map = vocab_map.lower()
        words = re.split('[:,\r]', vocab_map)

        words = [item for item in words if item not in ('', ' ')]
        if len(words) > 1:
            caption_vocabs = [i.replace(' ','') for i in words[1:]]
            cap_inds = [cap_ind_dict[word] for word in caption_vocabs]
            det_cat = words[0]
            mapping_dict.setdefault(det_cat, cap_inds)

    return mapping_dict