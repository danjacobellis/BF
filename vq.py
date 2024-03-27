#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dac
import torch
from torchaudio.transforms import Resample
from datasets import load_dataset
from audiotools import AudioSignal
from IPython.display import Audio


# In[2]:


dataset = load_dataset("danjacobellis/audio_har",split='semi_natural')
dataset = dataset.with_format('torch')
fs_orig = dataset[0]['path']['sampling_rate']
fs = 24000
resampler = Resample(fs_orig,fs)


# In[3]:


model_path = dac.utils.download(model_type="24khz")
model = dac.DAC.load(model_path)
model.to('cuda');


# In[4]:


def audio_to_code(sample):
    with torch.no_grad():
        signal = AudioSignal(sample['path']['array'],sample_rate=fs)
        signal.to(model.device)
        x = model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = model.encode(x)
        sample['codes'] = codes.detach().cpu()
        return sample


# In[5]:


dataset = dataset.map(audio_to_code)


# In[13]:


dataset = dataset.remove_columns('path')


# In[16]:


dataset.push_to_hub('danjacobellis/audio_har_descript_24kHz',split='train')

