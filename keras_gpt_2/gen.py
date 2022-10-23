import numpy as np


def generate(model,
             bpe,
             texts,
             length=100,
             top_k=1,
             temperature=1.0):
    """Generate text after the given contexts.

    :param model: The trained model.
    :param bpe: Byte pair encoding object.
    :param texts: A list of texts.
    :param length: The length of following texts to be generated.
    :param top_k: Choose the next token from top K.
    :param temperature: Randomness in boltzmann distribution.
    :return: A list of generated texts.
    """
    batch_size = len(texts) # if bsz==64
    encodes = [bpe.encode(text) for text in texts] # len 64, different length of token lists form this list
    text_lens = [len(encode) for encode in encodes] #[len1, len2, .. len54]
    max_len = max(text_lens) # get maximum of all lens
    input_data = [encode + [0] * (max_len - len(encode)) for encode in encodes] #padding to make align
    for shift in range(length): # from 0 - 99th adding tokens (add one, shift one to deal with next position)
        output_data = model.predict(np.array(input_data)) # predict a matrix with shape [64, 50257]
        for index in range(batch_size): # every sample
            probs = [(prob, i) for i, prob in enumerate(output_data[index, text_lens[index] + shift - 1])] # get the last token output with shape [1,50257]
            probs.sort(reverse=True) # sort with largest first
            probs = probs[:top_k] # get top k choices
            indices, probs = list(map(lambda x: x[1], probs)), list(map(lambda x: x[0], probs)) #seperate indice and the corresponding prob into 2 diff lists
            probs = np.array(probs) / temperature # deal 1 with new probs list, smoothing
            probs = probs - np.max(probs) # deal 2 with new probs list, get rid of potential too-large exponential
            probs = np.exp(probs) # deal 3 with new probs list
            probs = probs / np.sum(probs) # deal 4 with new probs list， with deals above is doing softmax
            next_token = np.random.choice(indices, p=probs) # base on probability distribution to choose the token
            input_data[index].append(0) # allocate a space in python
            input_data[index][text_lens[index] + shift] = next_token # set the new space with the chosen token
    outputs = [bpe.decode(input_data[index][:text_lens[index] + length]) for index in range(batch_size)] # get all result in str
    return outputs
