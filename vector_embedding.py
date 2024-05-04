import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import cosine

def cosine_similarity(embeddings1, embeddings2):
    # Calculate cosine similarities for each corresponding layer's embeddings
    cosine_similarity = [1 - cosine(embedding1, embedding2) for embedding1, embedding2 in zip(embeddings1, embeddings2)]
    return cosine_similarity



def euclidean_distances(embeddings1, embeddings2):

    if len(embeddings1) != len(embeddings2):
        raise ValueError("Both embeddings lists should have the same number of layers.")

    # Calculate Euclidean distance for each layer
    euclidean_distances = [np.linalg.norm(embedding1 - embedding2) for embedding1, embedding2 in zip(embeddings1, embeddings2)]
    return euclidean_distances


def get_word_embeddings_without_subword(sentence, word1, word2, tokenizer, model):
    # Tokenize input text and convert to IDs
    inputs = tokenizer(sentence, return_tensors="pt")
    token_ids = inputs['input_ids'].squeeze().tolist()  # Convert tensor to list of token IDs
    tokens = tokenizer.convert_ids_to_tokens(token_ids)  # Convert token IDs to tokens

    # Find indices of tokens corresponding to the target words
    indices_word1 = [i for i, token in enumerate(tokens) if word1 in token]
    indices_word2 = [i for i, token in enumerate(tokens) if word2 in token]

    # Get model output
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # Tuple of hidden states from each layer

    # Extract embeddings for each target word from each layer
    embeddings_word1 = [hidden_states[layer][0, idx].detach().numpy() for layer in range(len(hidden_states)) for idx in indices_word1]
    embeddings_word2 = [hidden_states[layer][0, idx].detach().numpy() for layer in range(len(hidden_states)) for idx in indices_word2]

    return embeddings_word1, embeddings_word2


def get_word_embeddings(sentence, word1, word2, tokenizer, model):
    # Tokenize input text and convert to IDs
    inputs = tokenizer(sentence, return_tensors="pt")
    token_ids = inputs['input_ids'].squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Prepare model and get output
    model_output = model(**inputs, output_hidden_states=True)
    hidden_states = model_output.hidden_states

    # Function to find and aggregate subword embeddings
    def aggregate_embeddings(word):
        # Include the 'Ġ' prefix for matching the start of words
        word_tokens = tokenizer.tokenize(' ' + word)  # Added space to ensure prefix
        indices = [i for i in range(len(tokens)) if tokens[i:i+len(word_tokens)] == word_tokens]

        # Flatten the list to capture all indices of subword tokens in range
        flattened_indices = [idx for i in indices for idx in range(i, i + len(word_tokens))]
        if not flattened_indices:
            return []
        # Average embeddings across subword tokens for each layer
        return [hidden_states[layer][0, flattened_indices].mean(dim=0).detach().numpy() 
                for layer in range(len(hidden_states))]

    embeddings_word1 = aggregate_embeddings(word1)
    embeddings_word2 = aggregate_embeddings(word2)

    return embeddings_word1, embeddings_word2


def calculate_similarity(embedding1, embedding2, similarity_function = "cosine", layers = 8):
    # Get word embeddings
    embedding1 = embedding1[:layers]
    embedding2 = embedding2[:layers]
    # Calculate similarity
    if similarity_function == "cosine":
        similarity = cosine_similarity(embedding1, embedding2)
    elif similarity_function == "euclidean":
        similarity = euclidean_distances(embedding1, embedding2)
    else:
        raise ValueError("Invalid similarity function specified.")

    return np.mean(similarity)