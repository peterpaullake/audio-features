'''
  This file contains helper utilities to make it easy to access
  DeepSpeech from Jupyter Notebook, as well as the code behind
  the three main experiments of this project, which are
  - learning inputs to maximize certain neurons,
  - finding examples in the training set that maximize certain neurons,
  - and building importance masks.
  The functions that do these jobs are run_train_inputs_experiment,
  run_search_dataset_experiment, and run_mask_experiment.

  This file is used to generate the raw results of the experiments,
  which is compute-intensive work and requires a GPU. The results
  are explained and presented in deepspeech-audio-features.ipynb.
'''

import app
from ds_ctcdecoder import Alphabet, ctc_beam_search_decoder, Scorer
import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# from matplotlib import cm
import numpy as np
import os
import pickle
import random
from scipy.fftpack import idct
import shutil
import soundfile as sf
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.python.ops import gen_audio_ops as contrib_audio

from DeepSpeech.training.deepspeech_training.train \
    import create_model, create_overlapping_windows, \
    do_single_file_inference, load_graph_for_evaluation, \
    rnn_impl_lstmblockfusedcell, rnn_impl_static_rnn
from DeepSpeech.training.deepspeech_training.util.config \
    import Config, initialize_globals
from DeepSpeech.training.deepspeech_training.util.feeding \
    import audio_to_features, audiofile_to_features, \
    create_dataset
from DeepSpeech.training.deepspeech_training.util.flags \
    import create_flags, FLAGS

def create_inference_graph(input_var, batch_size=1, n_steps=16, tflite=False):
    batch_size = batch_size if batch_size > 0 else None

    # Create feature computation graph
    input_samples = tfv1.placeholder(tf.float32,
                                     [Config.audio_window_samples],
                                     'input_samples')
    samples = tf.expand_dims(input_samples, -1)
    mfccs, _ = audio_to_features(samples, FLAGS.audio_sample_rate)
    mfccs = tf.identity(mfccs, name='mfccs')

    # Input tensor will be of shape [batch_size, n_steps, 2*n_context+1, n_input]
    # This shape is read by the native_client in DS_CreateModel to know the
    # value of n_steps, n_context and n_input. Make sure you update the code
    # there if this shape is changed.
    '''
    input_tensor = tfv1.placeholder(tf.float32,
                                    [batch_size,
                                     n_steps if n_steps > 0 else None,
                                     2 * Config.n_context + 1,
                                     Config.n_input],
                                    name='input_node')
    '''
    # input_tensor = input_var
    input_tensor = create_overlapping_windows(tf.expand_dims(input_var, 0))
    seq_length = tfv1.placeholder(tf.int32, [batch_size], name='input_lengths')

    if batch_size <= 0:
        # no state management since n_step is
        # expected to be dynamic too (see below)
        previous_state = None
    else:
        previous_state_c = tfv1.placeholder(tf.float32,
                                            [batch_size,
                                             Config.n_cell_dim],
                                            name='previous_state_c')
        previous_state_h = tfv1.placeholder(tf.float32,
                                            [batch_size,
                                             Config.n_cell_dim],
                                            name='previous_state_h')

        previous_state = tf.nn.rnn_cell.LSTMStateTuple(previous_state_c,
                                                       previous_state_h)

    # One rate per layer
    no_dropout = [None] * 6

    if tflite:
        rnn_impl = rnn_impl_static_rnn
    else:
        rnn_impl = rnn_impl_lstmblockfusedcell

    logits, layers = create_model(batch_x=input_tensor,
                                  batch_size=batch_size,
                                  seq_length=seq_length \
                                  if not FLAGS.export_tflite else None,
                                  dropout=no_dropout,
                                  previous_state=previous_state,
                                  overlap=False,
                                  rnn_impl=rnn_impl)

    # TF Lite runtime will check that input dimensions are 1, 2 or 4
    # by default we get 3, the middle one being batch_size which is forced to
    # one on inference graph, so remove that dimension
    if tflite:
        logits = tf.squeeze(logits, [1])

    # Apply softmax for CTC decoder
    logits = tf.nn.softmax(logits, name='logits')

    if batch_size <= 0:
        if tflite:
            raise NotImplementedError(('dynamic batch_size does not '
                                       'support tflite nor streaming'))
        if n_steps > 0:
            raise NotImplementedError(('dynamic batch_size expect '
                                       'n_steps to be dynamic too'))
        return (
            {
                'input': input_tensor,
                'input_lengths': seq_length,
            },
            {
                'outputs': logits,
            },
            layers
        )

    new_state_c, new_state_h = layers['rnn_output_state']
    new_state_c = tf.identity(new_state_c, name='new_state_c')
    new_state_h = tf.identity(new_state_h, name='new_state_h')

    inputs = {
        'input': input_tensor,
        'previous_state_c': previous_state_c,
        'previous_state_h': previous_state_h,
        'input_samples': input_samples,
    }

    if not FLAGS.export_tflite:
        inputs['input_lengths'] = seq_length

    outputs = {
        'outputs': logits,
        'new_state_c': new_state_c,
        'new_state_h': new_state_h,
        'mfccs': mfccs,
    }

    return inputs, outputs, layers

def encode_text(text, alphabet, blank_char='_'):
    """Convert text to a list of indicies."""
    indices = [28] * len(text)
    for i, char in zip(range(len(text)), text):
        if char != blank_char:
            indices[i] = alphabet.EncodeSingle(char)
    return indices

def probs_to_text(probs, blank_char='_'):
    """Convert the raw probabilities returned
    by the DeepSpeech model to text."""
    alphabet = Alphabet('DeepSpeech/data/alphabet.txt')
    ids = np.argmax(probs, axis=1).astype(np.uint32)
    text = ''
    for id in ids:
        id = int(id)
        if id == 28:
            char = blank_char
        else:
            char = alphabet.DecodeSingle(id)
        text += char
    return text

def compute_classification_loss(raw_logits, labels):
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=raw_logits)
    return tf.reduce_mean(losses)

def compute_text_classification_loss(raw_logits, shift=0, text=TEXT):
    alphabet = Alphabet('DeepSpeech/data/alphabet.txt')
    alphabet_size = alphabet.GetSize() + 1 # add 1 to include the blank char
    indices = np.asarray(encode_text(text, alphabet))
    indices = np.roll(indices, shift=shift, axis=0)
    labels = np.zeros((len(text), alphabet_size))
    for i in range(len(labels)):
        labels[i][indices[i]] = 1
    return compute_classification_loss(raw_logits, labels)

def search_dir(path, pred=lambda path : False):
    """Recursively search a directory for files
    whose filenames satisfy the given predicate.
    This is useful for finding all the files in
    a directory with a specific file extension."""
    if len(path) == 0:
        return []
    elif path[-1] == '/':
        return search_dir(path[:-1], pred)
    elif os.path.isfile(path):
        return [path] if pred(path) else []
    elif os.path.isdir(path):
        paths = []
        for p in os.listdir(path):
            paths.extend(search_dir(path + '/' + p, pred))
        return paths

def make_list_fn(fn):
    return lambda *args : list(fn(*args))

lfilter = make_list_fn(filter)
lmap = make_list_fn(map)
lreversed = make_list_fn(reversed)

def audio_to_mfccs(audio, sample_rate):
    """Compute the mfccs for an m x n matrix of m audio clips of length n."""
    # audio_to_features expects a dimension for the channels,
    # so we add an empty dimension to the end of the audio array
    audio = np.expand_dims(audio, axis=2)
    input_var = tfv1.placeholder(tf.float32, audio.shape[1:], 'input_var')
    mfccs = audio_to_features(input_var, sample_rate)[0]
    init_op = tfv1.global_variables_initializer()
    with tfv1.Session() as session:
        session.run(init_op)
        mfccs_evaled = np.zeros((len(audio), *mfccs.shape))
        for i, y in zip(range(len(audio)), audio):
            mfccs_evaled[i] = session.run(mfccs, feed_dict={input_var: y})
    return mfccs_evaled

def audio_file_to_mfccs(path):
    y, sr = librosa.load(path)
    return audio_to_mfccs(np.expand_dims(y, axis=0), sr)[0]

def eval_graph_on_audio_files(in_dir_path,
                              out_dir_path,
                              extension='.flac',
                              make_loss_dict=None,
                              batch_size=128):
    all_paths = search_dir(in_dir_path, lambda path : path.endswith(extension))
    def arrays_to_array(arrays):
        """Convert a list of m 1D arrays of arbitrary lengths
        to an an m x n array where n is the length of the longest
        array. Arrays shorter than n are padded with zeros."""
        max_len = max(lmap(len, arrays))
        array = np.zeros((len(arrays), max_len))
        for i, a in zip(range(len(arrays)), arrays):
            array[i][:len(a)] = a
        return array
    num_batches = int(np.ceil(len(all_paths) / batch_size))
    # all_outs = []
    reset_dir(out_dir_path)
    for i in range(0, len(all_paths), batch_size):
        print('starting batch %d/%d' % (i / batch_size + 1, num_batches))
        paths = all_paths[i:i+batch_size]
        print('loading %d audio files' % len(paths))
        audio = arrays_to_array(lmap(lambda path : librosa.load(path)[0], paths))
        sample_rate = librosa.load(paths[0])[1]
        print('computing mfccs')
        mfccs = audio_to_mfccs(audio, sample_rate)
        print('evaluating batch results')
        evaled = eval_graph(mfccs, make_loss_dict=make_loss_dict, batch=True)
        print('saving batch results')
        with open('%s/batch_%d.pickle' % (out_dir_path, i), 'wb') as f:
            pickle.dump(evaled, f)
    with open('%s/all_paths.pickle' % out_dir_path, 'wb') as f:
        pickle.dump(all_paths, f)

def mfccs_to_audio_file(input_path,
                        output_path='out.wav',
                        n_mels=257,
                        sample_rate=22050):
    mfccs = np.load(input_path)
    mfccs = np.swapaxes(mfccs, 0, 1)
    y = librosa.feature.inverse.mfcc_to_audio(mfccs, n_mels=n_mels)
    sf.write(output_path, y, sample_rate)

def reset_dir(path):
    """If the directory specified by path doesn't exist,
    create it. Otherwise, delete it, then create it."""
    if os.path.isfile(path):
        os.remove(path)
        reset_dir(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
        reset_dir(path)
    else:
        os.mkdir(path)

def get_loss_dict_term_keys(loss_dict):
    return list(set(loss_dict.keys()) - set(['coefs']))

def loss_dict_to_final_loss(loss_dict):
    """Convert a loss_dict to a loss function. A loss_dict
    is a dictionary that contains the terms and associated
    coefficients of a loss function. Since the terms are kept
    separate from each other and their coefficients, it's
    easy to evaluate and inspect the terms individually."""
    term_keys = get_loss_dict_term_keys(loss_dict)
    get_final_term = lambda key : loss_dict['coefs'][key] * loss_dict[key]
    final_terms = lmap(get_final_term, term_keys)
    return tf.math.add_n(final_terms)

def eval_loss_dict(loss_dict, session, feed_dict):
    evaled_loss_dict = {}
    term_keys = get_loss_dict_term_keys(loss_dict)
    for key in term_keys:
        evaled_loss_dict[key] = session.run(loss_dict[key],
                                            feed_dict=feed_dict)
    coefs = loss_dict['coefs']
    get_final_evaled_term = lambda key : coefs[key] * evaled_loss_dict[key]
    evaled_loss_dict['final_loss'] = sum(lmap(get_final_evaled_term, term_keys))
    return evaled_loss_dict

def eval_graph(input_array, make_loss_dict=None, batch=False):
    """Evaluate the DeepSpeech model on a batch of mfcc matricies.
    If make_loss_dict is None, return the model's output. Otherwise,
    use make_loss_dict to create a loss function, then return the
    evaluated final loss."""
    if not batch:
        # If input_array is not a batch,
        # expand it to be a batch of size 1.
        input_array = np.expand_dims(input_array, axis=0)
    input_var = tfv1.placeholder(tf.float32, input_array.shape[1:], 'input_var')
    # input_var = tf.Variable(input_array, name='input_var', dtype=tf.float32)
    with tfv1.variable_scope('', reuse=tf.AUTO_REUSE):
        cig = create_inference_graph
        input_dict, output_dict, layer_dict = cig(input_var,
                                                  batch_size=1,
                                                  n_steps=-1)
    if make_loss_dict is None:
        out = tf.squeeze(output_dict['outputs'])
    else:
        loss_dict = make_loss_dict(input_var, layer_dict)
    with tfv1.Session(config=Config.session_config) as session:
        # Restore variables from training checkpoint
        load_graph_for_evaluation(session)
        n_steps = int(input_var.shape[0])
        previous_state_c = np.zeros([1, Config.n_cell_dim])
        previous_state_h = np.zeros([1, Config.n_cell_dim])
        feed_dict = {input_dict['input_lengths']: [n_steps],
                     input_dict['previous_state_c']: previous_state_c,
                     input_dict['previous_state_h']: previous_state_h}
        # out_evaled = np.zeros((len(input_array), *out.shape))
        out_evaled = []
        for mfccs in input_array:
            feed_dict[input_var] = mfccs
            if make_loss_dict is None:
                out_evaled.append(session.run(out, feed_dict=feed_dict))
            else:
                out_evaled.append(eval_loss_dict(loss_dict, session, feed_dict))
    return out_evaled if batch else out_evaled[0]

def report_progress(iter_id, size, period=10):
    if iter_id % period == 0:
        print('progres: %.3f%%' % ((iter_id + 1) / size * 100))

def gen_binary_importance_mask(path, name):
    """Generate a binary mask with the same shape as the input file's
    mfcc matrix whose entries denote whether the corresponding mfcc
    entry is important (affects the classification of the audio clip)."""
    mfccs = audio_file_to_mfccs(path)
    mask = np.ones(mfccs.shape, dtype=int)
    input_tensor = tfv1.placeholder(tf.float32, mfccs.shape, 'input_samples')
    with tfv1.variable_scope('', reuse=tf.AUTO_REUSE):
        cig = create_inference_graph
        input_dict, output_dict, layer_dict = cig(input_tensor,
                                                  batch_size=1,
                                                  n_steps=-1)
    output = tf.squeeze(output_dict['outputs'])
    correct_text = probs_to_text(eval_graph(mfccs))
    with tfv1.Session(config=Config.session_config) as session:
        # Restore variables from training checkpoint
        load_graph_for_evaluation(session)
        n_steps = int(input_tensor.shape[0])
        previous_state_c = np.zeros([1, Config.n_cell_dim])
        previous_state_h = np.zeros([1, Config.n_cell_dim])
        feed_dict = {input_dict['input_lengths']: [n_steps],
                     input_dict['previous_state_c']: previous_state_c,
                     input_dict['previous_state_h']: previous_state_h}
        mask_size = mask.shape[0] * mask.shape[1]
        mask_ids = np.asarray(range(mask_size))
        np.random.seed(0)
        np.random.shuffle(mask_ids)
        i_to_row_col = lambda i : (i // mask.shape[1], i % mask.shape[1])
        reset_dir(name)
        for iter_id, mask_id in zip(range(mask_size), mask_ids):
            row, col = i_to_row_col(mask_id)
            mask[row][col] = 0
            feed_dict[input_tensor] = mfccs * mask
            text = probs_to_text(session.run(output, feed_dict=feed_dict))
            if text != correct_text:
                mask[row][col] = 1
            np.save(name + '/%d.npy' % iter_id, mask)
            report_progress(iter_id, mask_size)

def minimize_losses(names=[], input_vars=[],
                    make_loss_dicts=[], period=1):
    """Minimize a list of loss functions with respect
    to a list of corresponding input variables."""
    input_dicts = []
    # output_dicts = []
    layer_dicts = []
    # outputs = []
    with tfv1.variable_scope('', reuse=tf.AUTO_REUSE):
        for input_var in input_vars:
            cig = create_inference_graph
            input_dict, output_dict, layer_dict = cig(input_var,
                                                      batch_size=1,
                                                      n_steps=-1)
            input_dicts.append(input_dict)
            # output_dicts.append(output_dict)
            layer_dicts.append(layer_dict)
            # outputs.append(tf.squeeze(output_dict['outputs']))
    for key in ['layer_1', 'layer_2', 'layer_3',
                'rnn_output', 'layer_5', 'layer_6']:
        print(key, layer_dicts[0][key].shape)
    loss_dicts = []
    final_losses = []
    optimizer_ops = []
    optimizer = tfv1.train.AdamOptimizer()
    for make_loss_dict, input_var, layer_dict in zip(make_loss_dicts,
                                                     input_vars,
                                                     layer_dicts):
        loss_dict = make_loss_dict(input_var, layer_dict)
        final_loss = loss_dict_to_final_loss(loss_dict)
        optimizer_op = optimizer.minimize(final_loss, var_list=[input_var])
        loss_dicts.append(loss_dict)
        final_losses.append(final_loss)
        optimizer_ops.append(optimizer_op)
    with tfv1.Session(config=Config.session_config) as session:
        # Restore variables from training checkpoint
        load_graph_for_evaluation(session)
        previous_state_c = np.zeros([1, Config.n_cell_dim])
        previous_state_h = np.zeros([1, Config.n_cell_dim])
        feed_dicts = []
        for input_var, input_dict in zip(input_vars, input_dicts):
            n_steps = int(input_var.shape[0])
            feed_dict = {input_dict['input_lengths']: [n_steps],
                         input_dict['previous_state_c']: previous_state_c,
                         input_dict['previous_state_h']: previous_state_h}
            feed_dicts.append(feed_dict)
        # Delete files saved to the disk by
        # previous runs of minimize_losses
        lmap(reset_dir, names)
        iter_id = 0
        while True:
            for optimizer_op, feed_dict in zip(optimizer_ops, feed_dicts):
                session.run(optimizer_op, feed_dict=feed_dict)
            if iter_id % period == 0:
                # Every few iterations, print out the evaluated loss
                # functions, then save the training progress to the disk
                print('iter_id: %d' % iter_id)
                for name, input_var, loss_dict, \
                    final_loss, feed_dict in zip(names, input_vars, loss_dicts,
                                                 final_losses, feed_dicts):
                    print('objective: %s' % name)
                    evaled_loss_dict = eval_loss_dict(loss_dict,
                                                      session,
                                                      feed_dict)
                    keys = get_loss_dict_term_keys(loss_dict) + ['final_loss']
                    for key in keys:
                        path = name + '/%s_%d.npy' % (key, iter_id)
                        print('saving %s (%f)' % (path, evaled_loss_dict[key]))
                        np.save(path, evaled_loss_dict[key])
                    input_var_evaled = session.run(input_var)
                    path = name + '/input_var_%d.npy' % iter_id
                    print('saving ' + path)
                    np.save(path, input_var_evaled)
            iter_id += 1
            if iter_id >= 1:
                pass
                # return

def compute_activation_loss(layer, seed, n_features=3):
    # Randomly choose features to maximize
    n_audio_features = int(layer.shape[1])
    '''
    feature_ids = np.asarray(range(n_audio_features))
    np.random.seed(seed)
    np.random.shuffle(feature_ids)
    feature_ids = feature_ids[:n_features]
    '''
    random.seed(seed)
    feature_ids = random.sample(range(n_audio_features), n_features)
    features = lmap(lambda feature_id : layer[:,feature_id], feature_ids)
    activation = tf.reduce_mean(tf.stack(features))
    return -activation

def compute_total_var(x, per_px=True):
    """Return the total variation of a 2D tensor.
    https://en.wikipedia.org/wiki/Total_variation_denoising#2D_signal_images"""
    # return (x[1:,:] - x[:-1,:]).abs().sum() + (x[:,1:] - x[:,:-1]).abs().sum()
    vertical_var = tf.reduce_sum(tf.abs(x[1:,:] - x[:-1,:]))
    horizontal_var = tf.reduce_sum(tf.abs(x[:,1:] - x[:,:-1]))
    var = vertical_var + horizontal_var
    if per_px:
        return var / tf.cast(x.shape[0] * x.shape[1], tf.float32)
    else:
        return var

def run_mask_experiment(n_clips=3):
    """Generate importance masks for a few audio clips."""
    # randomly choose a few audio clips
    all_paths = search_dir('audio', lambda path : path.endswith('.wav'))
    path_to_name = lambda path : 'run_mask_experiment/' + str.replace(path,
                                                                      '/',
                                                                      '-')
    random.seed(0)
    paths = random.sample(all_paths, n_clips)
    names = lmap(path_to_name, paths)
    reset_dir('run_mask_experiment')
    lmap(gen_binary_importance_mask, paths, names)

def run_search_dataset_experiment():
    """Evaluate the graph on all the audio files in the
    training set, and save the results in .pickle files."""
    def make_loss_dict(input_var, layer_dict):
        layer_keys = ['layer_1', 'layer_2', 'layer_3',
                      'rnn_output', 'layer_5', 'layer_6']
        loss_dict = {'coefs': {}}
        for layer_key, seed in zip(layer_keys, range(len(layer_keys))):
            activation_loss = compute_activation_loss(layer_dict[layer_key],
                                                      seed=seed)
            key = 'activation_loss_%s' % layer_key
            loss_dict[key] = activation_loss
            loss_dict['coefs'][key] = 1
        return loss_dict
    eval_graph_on_audio_files('LibriSpeech',
                              'run_search_dataset_experiment',
                              make_loss_dict=make_loss_dict)
    '''
    make_loss_dicts = gen_make_loss_dicts(layer_keys)
    reset_dir('run_search_dataset_experiment')
    for layer_key, make_loss_dict in zip(layer_keys, make_loss_dicts):
        print('starting layer %s' % layer_key)
        eval_graph_on_audio_files('LibriSpeech', make_loss_dict=make_loss_dict)
        result = eval_graph_on_audio_files('LibriSpeech',
                                           make_loss_dict=make_loss_dict)
        path = 'run_search_dataset_experiment/%s.pickle' % layer_key
        with open(path, 'wb') as f:
            pickle.dump(result, f)
    '''

def run_train_inputs_experiment():
    """Train inputs to maximize the activations of various
    neurons of hidden layers inside the DeepSpeech model."""
    layer_keys = ['layer_1', 'layer_2', 'layer_3',
                  'rnn_output', 'layer_5', 'layer_6']
    names = lmap(lambda layer_key : 'run_train_inputs_experiment/%s' % layer_key,
                 layer_keys)
    input_vars = []
    # make_loss_dicts = gen_make_loss_dicts(layer_keys)
    def make_make_loss_dict(layer_key, seed):
        def make_loss_dict(input_var, layer_dict):
            activation_loss = compute_activation_loss(layer_dict[layer_key],
                                                      seed=seed)
            total_var_loss = compute_total_var(input_var)
            return {'activation_loss': activation_loss,
                    'total_var_loss': total_var_loss,
                    # 'coefs': {'activation_loss': 1, 'total_var_loss': 1}}
                    'coefs': {'activation_loss': 1, 'total_var_loss': 2.808}}
        return make_loss_dict
    make_loss_dicts = lmap(make_make_loss_dict,
                           layer_keys,
                           range(len(layer_keys)))
    for layer_key in layer_keys:
        shape = (135, 26)
        # shape = (256, 26)
        input_var = tf.Variable(np.random.randn(*shape),
                                name='input_var',
                                dtype=tf.float32)
        input_vars.append(input_var)
    reset_dir('run_train_inputs_experiment')
    minimize_losses(names, input_vars, make_loss_dicts, period=10)

def main(argv):
    if Config._config is None:
        initialize_globals()

    run_mask_experiment()
    run_search_dataset_experiment()
    run_train_inputs_experiment()

def deepspeech_call(fn):
    """Call a function that uses DeepSpeech. This
    function makes sure that DeepSpeech's command
    line arguments and global variables are set up
    before calling fn."""
    if not FLAGS.is_parsed():
        create_flags()
    def main(argv):
        if Config._config is None:
            initialize_globals()
        fn()
    app.run(main, argv=['deepspeech-audio-features',
                        '--log_level', '2',
                        '--n_hidden', '2048',
                        '--checkpoint_dir', 'deepspeech-0.8.0-checkpoint',
                        '--alphabet_config_path',
                        'DeepSpeech/data/alphabet.txt'])

def run_script():
    if not FLAGS.is_parsed():
        create_flags()
    app.run(main, argv=['deepspeech-audio-features',
                        '--n_hidden', '2048',
                        '--checkpoint_dir', 'deepspeech-0.8.0-checkpoint',
                        '--alphabet_config_path',
                        'DeepSpeech/data/alphabet.txt'])
    # '--scorer', 'deepspeech-0.7.4-models.scorer'])

def report_which_neurons():
    """Report out which neurons we maximized."""
    n_features = 3
    ns = [2048, 2048, 2048, 2048, 2048, 29]
    seeds = range(len(ns))
    for n_audio_features, seed in zip(ns, seeds):
        feature_ids = random.sample(range(n_audio_features), n_features)
        print(feature_ids)
        

run_script()
