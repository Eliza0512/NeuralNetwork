import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile


batch_size = 50

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    filename = "reviews.tar.gz"
    #Extract data from tarball and store as list of strings
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data1/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'data1/'))
    #print("----------READING DATA--------------")
    data = np.ndarray(shape=(25000,40), dtype=np.int32)
    dir = os.path.dirname(__file__)
    pos_file_list = glob.glob(os.path.join(dir,
                                        'data1/pos/*'))
    neg_file_list = glob.glob(os.path.join(dir,
                                        'data1/neg/*'))
    #print("-----------Parsing files------------")
    count = 0
    for f in pos_file_list:
        with open(f,"r",encoding="utf-8") as openf:
            s = openf.read()
            v = preprocessing(s)
            for i in range(40):
                if v[i] in glove_dict.keys():
                    #print("v[]  :  ",v[i])
                    data[count,i]=glove_dict[v[i]]
                else:
                    #print("Error-------------",v[i])
                    data[count,i]=0
        #print("array data-------------",data[count])
        count += 1
        if(count==12500):
            break
    
    for f in neg_file_list:
        with open(f,"r",encoding="utf-8") as openf:
            s = openf.read()
            v = preprocessing(s)
            for i in range(40):
                
                if v[i] in glove_dict.keys():
                    data[count,i]=glove_dict[v[i]]
                else:
                    data[count,i]=0
        count += 1
        if(count==25000):
            break       
    return data

def preprocessing(s):
    s = s.replace("<br />","")
    s = s.replace("<br/>","")
    raw_data = s.split()
    result=[]
    for word in raw_data:
        word=word.strip("[.]")
        word=word.strip("[,]")
        word=word.strip("[!]")
        word=word.strip("[?]")
        word=word.strip("[:]")
        word=word.strip('"')
        word=word.strip("'")
        word=word.strip("[(]")
        word=word.strip("[)]")
        word=word.lower()
        result.append(word)
    if len(result)>40:
        while "the" in result:
            result.remove("the")
            if len(result) ==40:
                break
    if len(result)>40:
        while "a" in result:
            result.remove("a")
            if len(result) ==40:
                break
    if len(result)>40:
        while "of" in result:
            result.remove("of")
            if len(result) ==40:
                break
    if len(result)<40:
        while len(result)<40:
            result.append("UNK")
    #print("after-preprocessing-----------",result[:40])
    return result[:40]
        
        

def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    word_index_dict = dict()
    word_index_dict["UNK"]=0
    embed_array = list()
    zero_vector = [0,]*50
    embed_array.append(zero_vector)
    key = 1
    for line in data:
        raw = line.split()
        raw[-1]=raw[-1]
        word = raw[0]
        word_index_dict[word]=key
        #if key<=50:
            #print("------key and value----------",word," ",key)
            #print("array:  ",raw[1:])
        embed_array.append(raw[1:])
        key += 1
    embeddings = np.ndarray(shape=(len(embed_array)+1,50), dtype=np.float32)
    for i in range(len(embed_array)):
        embeddings[i] = np.array(embed_array[i])
        #print(embeddings[i])       
    return embeddings,word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""
    n_inputs = 40
    n_classes = 2
    n_hidden_units = 25
    lr = 0.001
    #n_layers = 2
    # x y placeholder
    input_data= tf.placeholder(tf.int32, [batch_size,n_inputs],name="input_data")
    labels = tf.placeholder(tf.float32, [batch_size, n_classes],name="labels")
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())
    
    # Use GRU Cell.
    LSTM_cell = tf.contrib.rnn.LSTMCell(n_hidden_units)
    LSTM_cell = tf.contrib.rnn.DropoutWrapper(cell=LSTM_cell, output_keep_prob=dropout_keep_prob)
    #cell_stack = []
    #for i in range(0,n_layers):
    #    cell_stack.append(tf.contrib.rnn.DropoutWrapper(cell=tf.contrib.rnn.LSTMCell(n_hidden_units), output_keep_prob=dropout_keep_prob))
    #GRU_cell = tf.contrib.rnn.MultiRNNCell(cell_stack)
    input_vector = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)
    #get outputs
    outputs, final_state = tf.nn.dynamic_rnn(LSTM_cell, input_vector,dtype = tf.float32)
    rnn_outputs =tf.layers.dense(outputs[:,-1,:],2)
    
    #print("out-------------------------",outputs)
    #print("final_state------------------",final_state)

    # loss and accuracy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rnn_outputs, labels=labels),name="loss")
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
    equality = tf.equal(tf.argmax(rnn_outputs,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32),name="accuracy")
    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss

#e,d = load_glove_embeddings()
#load_data(d)
