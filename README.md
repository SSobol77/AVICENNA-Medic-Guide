
![aimed2](https://github.com/SSobol77/AI_myTransformer_Project/assets/108773983/fff1b39c-e0e2-4d2e-a477-5d7582dc5c7e)

# AVICENNA Medic Guide

### Transformer Project

![421223173_122095481072200040_3579657397230548228_n](https://github.com/SSobol77/AI_myTransformer_Project/assets/108773983/e25650b8-5576-44d0-b1c9-dfca238ddbae)

https://chat.openai.com/g/g-UrRHZyFQW-avicenna-medic-guide

### Step 1.
> We demonstrating a first implementation of the basic transformer engine **CyberAI** *(sorry, this closed repository for project development)*in Python using the PyTorch library.
> This example will be simplified and intended to demonstrate basic concepts, rather than for practical use in complex natural language processing tasks.
> Creating a simple transformer includes several key components: **attention layers**, **encoder layers**, and **decoder layers**. 

### Here is the basic structure of the code:

1. **Attention Layer**: The main block that allows the model to focus on different parts of the input data.

2. **Encoder Layer**: Processes the input data and transmits the information to the decoder.

3. **Decoder Layer**: Accepts the output of the encoder and generates the final output.

### First, let's write the code for the attention layer:

This code is an implementation of the attention layer (SelfAttention), which is a key part of the transformer. This layer uses the attention mechanism to calculate the weighted sum of values based on the similarity of queries and keys. This allows the model to dynamically focus on the most important parts of the input data.

Next, we can add layers of encoder and decoder to create a complete transformer. However, keep in mind that a full-fledged transformer is a rather complex model, and its full implementation can be quite voluminous. The above code is intended to illustrate the basic concepts. 


## Transformer Project
```
TransformerProject
│
├── data                 # A folder for storing data
│
├── models               # Folder with models
│   ├── encoder.py       # The code for the encoder layer
│   ├── decoder.py       # Code for the decoder layer
│   └── transformer.py   # Common transformer code, including encoder and decoder
│
├── utils                # Auxiliary utilities and scripts
│   ├── __init__.py      # Initializing the module
│   ├── tokenization.py  # Scripts for text tokenization
│   └── dataset.py       # Scripts for processing and loading data
│
├── training             # Scripts for training and testing
│   ├── train.py         # A script for training a model
│   └── test.py          # The script for testing the model
│
├── requirements.txt     # List of dependencies to install
└── README.md            # Description of the project


```
