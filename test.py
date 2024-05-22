from huggingface_hub import login
from datasets import load_dataset,load_metric,Audio 
import requests
import json

login('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')

dataset=load_dataset('mapau7/parlaSmall',data_dir='data')
print(dataset['train'])