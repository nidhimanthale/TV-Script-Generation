
# coding: utf-8

# In[ ]:


import os
import pickle
import copy
import numpy as np 


# In[ ]:


CODES={'<PAD>':0, '<EOS>':1, '<UNK>':2, '<GO>':3}


# In[ ]:


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


# In[ ]:


def preprocess_and_save_data(source_path, target_path, text_to_ids):
    """
    Preprocess Text Data.  Save to to file.
    """
    # Preprocess
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    source_text = source_text.lower()
    target_text = target_text.lower()

    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    # Save Data
    with open('preprocess.p', 'wb') as out_file:
        pickle.dump((
            (source_text, target_text),
            (source_vocab_to_int, target_vocab_to_int),
            (source_int_to_vocab, target_int_to_vocab)), out_file)


# In[ ]:


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    with open('preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


# In[ ]:


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    """
    vocab = set(text.split())
    vocab_to_int = copy.copy(CODES)

    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i

    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


# In[ ]:


def save_params(params):
    """
    Save parameters to file
    """
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


# In[ ]:


def load_params():
    """
    Load parameters from file
    """
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)


# In[ ]:


def batch_data(source, target, batch_size):
    """
    Batch source and target together
    """
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield np.array(pad_sentence_batch(source_batch)), np.array(pad_sentence_batch(target_batch))


# In[ ]:


def pad_sentence_batch(sentence_batch):
    """
    Pad sentence with <PAD> id
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [CODES['<PAD>']] * (max_sentence - len(sentence))
            for sentence in sentence_batch]


# In[ ]:


import numpy as np
import tensorflow as tf
import itertools
import collections


# In[ ]:


def _print_success_message():
    print('Tests Passed')


# In[ ]:


def test_text_to_ids(text_to_ids):
    test_source_text = 'new jersey is sometimes quiet during autumn , and it is snowy in april .\nthe united states is usually chilly during july , and it is usually freezing in november .\ncalifornia is usually quiet during march , and it is usually hot in june .\nthe united states is sometimes mild during june , and it is cold in september .'
    test_target_text = 'new jersey est parfois calme pendant l\' automne , et il est neigeux en avril .\nles états-unis est généralement froid en juillet , et il gèle habituellement en novembre .\ncalifornia est généralement calme en mars , et il est généralement chaud en juin .\nles états-unis est parfois légère en juin , et il fait froid en septembre .'

    test_source_text = test_source_text.lower()
    test_target_text = test_target_text.lower()

    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(test_source_text)
    target_vocab_to_int, target_int_to_vocab =create_lookup_tables(test_target_text)

    test_source_id_seq, test_target_id_seq = text_to_ids(test_source_text, test_target_text, source_vocab_to_int, target_vocab_to_int)

    assert len(test_source_id_seq) == len(test_source_text.split('\n')),        'source_id_text has wrong length, it should be {}.'.format(len(test_source_text.split('\n')))
    assert len(test_target_id_seq) == len(test_target_text.split('\n')),         'target_id_text has wrong length, it should be {}.'.format(len(test_target_text.split('\n')))

    target_not_iter = [type(x) for x in test_source_id_seq if not isinstance(x, collections.Iterable)]
    assert not target_not_iter,        'Element in source_id_text is not iteratable.  Found type {}'.format(target_not_iter[0])
    target_not_iter = [type(x) for x in test_target_id_seq if not isinstance(x, collections.Iterable)]
    assert not target_not_iter,         'Element in target_id_text is not iteratable.  Found type {}'.format(target_not_iter[0])

    source_changed_length = [(words, word_ids)
                             for words, word_ids in zip(test_source_text.split('\n'), test_source_id_seq)
                             if len(words.split()) != len(word_ids)]
    assert not source_changed_length,        'Source text changed in size from {} word(s) to {} id(s): {}'.format(
            len(source_changed_length[0][0].split()), len(source_changed_length[0][1]), source_changed_length[0][1])

    target_missing_end = [word_ids for word_ids in test_target_id_seq if word_ids[-1] != target_vocab_to_int['<EOS>']]
    assert not target_missing_end,        'Missing <EOS> id at the end of {}'.format(target_missing_end[0])

    target_bad_size = [(words.split(), word_ids)
                       for words, word_ids in zip(test_target_text.split('\n'), test_target_id_seq)
                       if len(word_ids) != len(words.split()) + 1]
    assert not target_bad_size,        'Target text incorrect size.  {} should be length {}'.format(
            target_bad_size[0][1], len(target_bad_size[0][0]) + 1)

    source_bad_id = [(word, word_id)
                     for word, word_id in zip(
                        [word for sentence in test_source_text.split('\n') for word in sentence.split()],
                        itertools.chain.from_iterable(test_source_id_seq))
                     if source_vocab_to_int[word] != word_id]
    assert not source_bad_id,        'Source word incorrectly converted from {} to id {}.'.format(source_bad_id[0][0], source_bad_id[0][1])

    target_bad_id = [(word, word_id)
                     for word, word_id in zip(
                        [word for sentence in test_target_text.split('\n') for word in sentence.split()],
                        [word_id for word_ids in test_target_id_seq for word_id in word_ids[:-1]])
                     if target_vocab_to_int[word] != word_id]
    assert not target_bad_id,        'Target word incorrectly converted from {} to id {}.'.format(target_bad_id[0][0], target_bad_id[0][1])

    _print_success_message()


# In[ ]:


def test_model_inputs(model_inputs):
    with tf.Graph().as_default():
        input_data, targets, lr, keep_prob = model_inputs()

        # Check type
        assert input_data.op.type == 'Placeholder',            'Input is not a Placeholder.'
        assert targets.op.type == 'Placeholder',            'Targets is not a Placeholder.'
        assert lr.op.type == 'Placeholder',            'Learning Rate is not a Placeholder.'
        assert keep_prob.op.type == 'Placeholder',             'Keep Probability is not a Placeholder.'

        # Check name
        assert input_data.name == 'input:0',            'Input has bad name.  Found name {}'.format(input_data.name)
        assert keep_prob.name == 'keep_prob:0',             'Keep Probability has bad name.  Found name {}'.format(keep_prob.name)

        assert tf.assert_rank(input_data, 2, message='Input data has wrong rank')
        assert tf.assert_rank(targets, 2, message='Targets has wrong rank')
        assert tf.assert_rank(lr, 0, message='Learning Rate has wrong rank')
        assert tf.assert_rank(keep_prob, 0, message='Keep Probability has wrong rank')

    _print_success_message()


# In[ ]:


def test_encoding_layer(encoding_layer):
    rnn_size = 512
    batch_size = 64
    num_layers = 3

    with tf.Graph().as_default():
        rnn_inputs = tf.placeholder(tf.float32, [batch_size, 22, 1000])
        keep_prob = tf.placeholder(tf.float32)
        states = encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob)

        assert len(states) == num_layers,            'Found {} state(s). It should be {} states.'.format(len(states), num_layers)

        bad_types = [type(state) for state in states if not isinstance(state, tf.contrib.rnn.LSTMStateTuple)]
        assert not bad_types,            'Found wrong type: {}'.format(bad_types[0])

        bad_shapes = [state_tensor.get_shape()
                      for state in states
                      for state_tensor in state
                      if state_tensor.get_shape().as_list() not in [[None, rnn_size], [batch_size, rnn_size]]]
        assert not bad_shapes,            'Found wrong shape: {}'.format(bad_shapes[0])

    _print_success_message()


# In[ ]:


def test_decoding_layer(decoding_layer):
    batch_size = 64
    vocab_size = 1000
    embedding_size = 200
    sequence_length = 22
    rnn_size = 512
    num_layers = 3
    target_vocab_to_int = {'<EOS>': 1, '<GO>': 3}

    with tf.Graph().as_default():
        dec_embed_input = tf.placeholder(tf.float32, [batch_size, 22, embedding_size])
        dec_embeddings = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        keep_prob = tf.placeholder(tf.float32)
        state = tf.contrib.rnn.LSTMStateTuple(
            tf.placeholder(tf.float32, [None, rnn_size]),
            tf.placeholder(tf.float32, [None, rnn_size]))
        encoder_state = (state, state, state)

        train_output, inf_output = decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size,
                                                  sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)

        assert isinstance(train_output, tf.Tensor),            'Train Logits is wrong type: {}'.format(type(train_output))
        assert isinstance(inf_output, tf.Tensor),             'Inference Logits is wrong type: {}'.format(type(inf_output))

        assert train_output.get_shape().as_list() == [batch_size, None, vocab_size],            'Train Logits is the wrong shape: {}'.format(train_output.get_shape())
        assert inf_output.get_shape().as_list() == [None, None, vocab_size],             'Inference Logits is the wrong shape: {}'.format(inf_output.get_shape())

    _print_success_message()


# In[ ]:


def test_seq2seq_model(seq2seq_model):
    batch_size = 64
    target_vocab_size = 300
    sequence_length = 22
    rnn_size = 512
    num_layers = 3
    target_vocab_to_int = {'<EOS>': 1, '<GO>': 3}

    with tf.Graph().as_default():
        input_data = tf.placeholder(tf.int32, [64, 22])
        target_data = tf.placeholder(tf.int32, [64, 22])
        keep_prob = tf.placeholder(tf.float32)
        train_output, inf_output = seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length,
                                                 200, target_vocab_size, 64, 80, rnn_size, num_layers, target_vocab_to_int)

        assert isinstance(train_output, tf.Tensor),            'Train Logits is wrong type: {}'.format(type(train_output))
        assert isinstance(inf_output, tf.Tensor),             'Inference Logits is wrong type: {}'.format(type(inf_output))

        assert train_output.get_shape().as_list() == [batch_size, None, target_vocab_size],            'Train Logits is the wrong shape: {}'.format(train_output.get_shape())
        assert inf_output.get_shape().as_list() == [None, None, target_vocab_size],             'Inference Logits is the wrong shape: {}'.format(inf_output.get_shape())


    _print_success_message()


# In[ ]:


def test_sentence_to_seq(sentence_to_seq):
    sentence = 'this is a test sentence'
    vocab_to_int = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, 'this': 3, 'is': 6, 'a': 5, 'sentence': 4}

    output = sentence_to_seq(sentence, vocab_to_int)

    assert len(output) == 5,        'Wrong length. Found a length of {}'.format(len(output))

    assert output[3] == 2,        'Missing <UNK> id.'

    assert np.array_equal(output, [3, 6, 5, 2, 4]),        'Incorrect ouput. Found {}'.format(output)

    _print_success_message()


# In[ ]:


def test_process_decoding_input(process_decoding_input):
    batch_size = 2
    seq_length = 3
    target_vocab_to_int = {'<GO>': 3}
    with tf.Graph().as_default():
        target_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        dec_input = process_decoding_input(target_data, target_vocab_to_int, batch_size)

        assert dec_input.get_shape() == (batch_size, seq_length),            'Wrong shape returned.  Found {}'.format(dec_input.get_shape())

        test_target_data = [[10, 20, 30], [40, 18, 23]]
        with tf.Session() as sess:
            test_dec_input = sess.run(dec_input, {target_data: test_target_data})

        assert test_dec_input[0][0] == target_vocab_to_int['<GO>'] and               test_dec_input[1][0] == target_vocab_to_int['<GO>'],            'Missing GO Id.'

    _print_success_message()


# In[ ]:


def test_decoding_layer_train(decoding_layer_train):
    batch_size = 64
    vocab_size = 1000
    embedding_size = 200
    sequence_length = 22
    rnn_size = 512
    num_layers = 3

    with tf.Graph().as_default():
        with tf.variable_scope("decoding") as decoding_scope:
            dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
            output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
            dec_embed_input = tf.placeholder(tf.float32, [batch_size, 22, embedding_size])
            keep_prob = tf.placeholder(tf.float32)
            state = tf.contrib.rnn.LSTMStateTuple(
                tf.placeholder(tf.float32, [None, rnn_size]),
                tf.placeholder(tf.float32, [None, rnn_size]))
            encoder_state = (state, state, state)

            train_logits = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length,
                                 decoding_scope, output_fn, keep_prob)

            assert train_logits.get_shape().as_list() == [batch_size, None, vocab_size],                 'Wrong shape returned.  Found {}'.format(train_logits.get_shape())

    _print_success_message()


# In[ ]:


def test_decoding_layer_infer(decoding_layer_infer):
    vocab_size = 1000
    sequence_length = 22
    embedding_size = 200
    rnn_size = 512
    num_layers = 3

    with tf.Graph().as_default():
        with tf.variable_scope("decoding") as decoding_scope:
            dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
            output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
            dec_embeddings = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            keep_prob = tf.placeholder(tf.float32)
            state = tf.contrib.rnn.LSTMStateTuple(
                tf.placeholder(tf.float32, [None, rnn_size]),
                tf.placeholder(tf.float32, [None, rnn_size]))
            encoder_state = (state, state, state)

            infer_logits = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, 10, 20,
                                sequence_length, vocab_size, decoding_scope, output_fn, keep_prob)

            assert infer_logits.get_shape().as_list() == [None, None, vocab_size],                  'Wrong shape returned.  Found {}'.format(infer_logits.get_shape())

    _print_success_message()


# In[ ]:


# Load Data

source_path = 'data/small_vocab_en.txt'
target_path = 'data/small_vocab_fr.txt'
source_text = load_data(source_path)
target_text = load_data(target_path)


# In[ ]:


# Explore the Data

view_sentence_range = (0, 10)

import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


# In[ ]:


def sentence_to_ids(sentence, vocab_to_int, is_target=True):
    words = sentence.split(' ')
    words = list(filter(None, words))
    tmp = [vocab_to_int[x] for x in words]
    if(is_target):
        tmp = tmp + [vocab_to_int['<EOS>']]
    return tmp

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    source_texts = source_text.split("\n")
    target_texts = target_text.split("\n")
    source_list = []
    target_list = []
    for s in source_texts:
        source_list.append(sentence_to_ids(s,source_vocab_to_int, is_target=False))
    for s in target_texts:
        target_list.append(sentence_to_ids(s,target_vocab_to_int, is_target=True))    
    
    return source_list, target_list

test_text_to_ids(text_to_ids)


# In[ ]:


preprocess_and_save_data(source_path, target_path, text_to_ids)


# In[ ]:


# Check Point

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = load_preprocess()


# In[ ]:


# Build the Network


# In[ ]:


# Input

def model_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """
    # TODO: Implement Function
    text_int_input = tf.placeholder(tf.int32, shape=(None,None), name="input")
    text_int_target = tf.placeholder(tf.int32, shape=(None,None), name="output")
    learning_rate = tf.placeholder(tf.float32, shape=None)
    keep_prob = tf.placeholder(tf.float32, shape=None, name="keep_prob")
    return text_int_input, text_int_target, learning_rate, keep_prob

test_model_inputs(model_inputs)


# In[ ]:


# Process Decoding Input

def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for dencoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    target_data = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)
    return target_data

test_process_decoding_input(process_decoding_input)


# In[ ]:


# Encoding

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """
    # TODO: Implement Function
    # Encoder
    enc_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
    enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=keep_prob)
    rnn, state = tf.nn.dynamic_rnn(enc_cell, rnn_inputs, dtype=tf.float32)
    return state

test_encoding_layer(encoding_layer)


# In[ ]:


# Decoding - Training

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits
    """
    # TODO: Implement Function
    train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        dec_cell, train_decoder_fn, dec_embed_input, sequence_length, scope=decoding_scope)
    
    # Apply output function
    train_logits =  output_fn(train_pred)
    train_logits = tf.nn.dropout(train_logits, keep_prob)
    return train_logits

test_decoding_layer_train(decoding_layer_train)


# In[ ]:


# Decoding - Inference

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: The maximum allowed time steps to decode
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """
    # TODO: Implement Function
    infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
        output_fn, encoder_state, dec_embeddings, start_of_sequence_id, end_of_sequence_id, 
        maximum_length, vocab_size)
    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, infer_decoder_fn, scope=decoding_scope)
    inference_logits = tf.nn.dropout(inference_logits, keep_prob)
    return inference_logits

test_decoding_layer_infer(decoding_layer_infer)


# In[ ]:


# Decoding Layer

def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # Decoder RNNs
    dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)

    with tf.variable_scope("decoding") as decoding_scope:
        # Output Layer
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
        
        train_logits = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob)    
    with tf.variable_scope("decoding", reuse=True) as decoding_scope:    
        infer_logits = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'],
                         sequence_length, vocab_size, decoding_scope, output_fn, keep_prob)
    return train_logits, infer_logits

test_decoding_layer(decoding_layer)


# In[ ]:


# Neural Network

def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, enc_embedding_size)
    encoder_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob)
    target_data = process_decoding_input(target_data, target_vocab_to_int, batch_size)
    #target_embed = tf.contrib.layers.embed_sequence(target_data, target_vocab_size, dec_embedding_size)    
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, dec_embedding_size]))
    target_embed = tf.nn.embedding_lookup(dec_embeddings, target_data)
    return decoding_layer(target_embed, dec_embeddings, encoder_state, target_vocab_size, sequence_length, rnn_size,                    num_layers, target_vocab_to_int, keep_prob)

test_seq2seq_model(seq2seq_model)


# In[ ]:


# Neural Network Training


# In[ ]:


# Hyperparameters

# Number of Epochs
epochs = 4
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 512

# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 256
decoding_embedding_size = 256
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.5


# In[ ]:


# Graph

save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_source_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob = model_inputs()
    sequence_length = tf.placeholder_with_default(max_source_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)
    
    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
        encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)

    tf.identity(inference_logits, 'logits')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


# In[ ]:


# Train

import time

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1]), (0,0)],
            'constant')

    return np.mean(np.equal(target, np.argmax(logits, 2)))

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]

valid_source = pad_sentence_batch(source_int_text[:batch_size])
valid_target = pad_sentence_batch(target_int_text[:batch_size])

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch) in enumerate(
                batch_data(train_source, train_target, batch_size)):
            start_time = time.time()
            
            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})
            
            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})
                
            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(np.array(valid_target), batch_valid_logits)
            end_time = time.time()
            print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
                  .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')


# In[ ]:


# Save parameters for checkpoint
helper.save_params(save_path)


# In[ ]:


#Sentence to Sequence

def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    # TODO: Implement Function
    sentence = sentence.lower()
    words = sentence.split(' ')
    ids = []
    for w in words:
        if(w in vocab_to_int):
            ids.append(vocab_to_int[w])
        else:
            ids.append(vocab_to_int['<UNK>'])
    return ids

test_sentence_to_seq(sentence_to_seq)


# In[ ]:


#Translate

translate_sentence = 'he saw a old yellow truck .'

translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))

