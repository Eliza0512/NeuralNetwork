import tensorflow as tf
import numpy as np
import collections

data_index = 0

def generate_batch(data, batch_size, skip_window):
    """
    Generates a mini-batch of training data for the training CBOW
    embedding model.
    :param data (numpy.ndarray(dtype=int, shape=(corpus_size,)): holds the
        training corpus, with words encoded as an integer
    :param batch_size (int): size of the batch to generate
    :param skip_window (int): number of words to both left and right that form
        the context window for the target word.
    Batch is a vector of shape (batch_size, 2*skip_window), with each entry for the batch containing all the context words, with the corresponding label being the word in the middle of the context
    """
    global data_index
    batch = np.ndarray(shape=(batch_size,2*skip_window), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    #buffer = collections.deque(maxlen=span)
    #if data_index + span > len(data):
    #    data_index = 0
    #buffer.extend(data[data_index:data_index + span])
    data_index =0
    for i in range(batch_size):
        buffer = data[data_index:data_index+span]
        target = buffer[skip_window]  # target label at the center of the buffer
        for j in range(span):
            #print(j," : ",buffer[j])
            if j < skip_window:
                batch[i,j] = buffer[j]
            elif j > skip_window:
                batch[i,j-1] = buffer[j]
        labels[i,0] = target
        #if data_index == len(data)-span:
            # reached the end of the data, start again
            #data_index = 0
        #else:
            #data_index += 1
        data_index += 1
        data_index = data_index%len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    #data_index = (data_index - span) % len(data)
    return batch, labels

def get_mean_context_embeds(embeddings, train_inputs):
    """
    :param embeddings (tf.Variable(shape=(vocabulary_size, embedding_size))
    :param train_inputs (tf.placeholder(shape=(batch_size, 2*skip_window))
    returns:
        `mean_context_embeds`: the mean of the embeddings for all context words
        for each entry in the batch, should have shape (batch_size,
        embedding_size)
    """
    # cpu is recommended to avoid out of memory errors, if you don't
    # have a high capacity GPU
    with tf.device('/cpu:0'):
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        mean_context_embeds  = tf.reduce_mean(embed,1)     
    return mean_context_embeds
