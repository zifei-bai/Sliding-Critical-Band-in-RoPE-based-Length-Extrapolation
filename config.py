# config.py
import torch


VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '&', '*', '@']

PADDING_TOKEN_INDEX = 12
END_TOKEN_INDEX = 11

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'