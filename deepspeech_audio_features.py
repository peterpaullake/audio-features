import absl.app as app
import contextlib
import DeepSpeech.training.deepspeech_training.train as train
import DeepSpeech.training.deepspeech_training.util.config as config
import DeepSpeech.training.deepspeech_training.util.feeding as feeding
import DeepSpeech.training.deepspeech_training.util.flags as flags
import ds_ctcdecoder
import librosa
import matplotlib.pyplot as plt
import numpy as np
import obspy.signal.filter
import os
import shutil
import soundfile as sf
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

# Indicates whether the code is running
# on a Google Compute Engine instance.
ON_GCE = False

class DeepSpeech:
    """Context manager used to wrap code that uses DeepSpeech.

    This class is useful for making sure that the DeepSpeech
    global configuration is set and the TensorFlow graph is
    reset before running code that uses DeepSpeech.

    Example:
    >>> with DeepSpeech():
    >>>     train.do_single_file_inference('audio/4507-16021-0012.wav')
    """

    def __enter__(self):
        if not flags.FLAGS.is_parsed():
            flags.create_flags()
        argv = ['deepspeech_audio_features.py',
                '--alphabet_config_path=DeepSpeech/data/alphabet.txt',
                '--checkpoint_dir=deepspeech-0.8.2-checkpoint']
        args = app._run_init(argv, app.parse_flags_with_usage)
        while app._init_callbacks:
            callback = app._init_callbacks.popleft()
            callback()
        if config.Config._config is None:
            config.initialize_globals()
        tfv1.reset_default_graph()

    def __exit__(self, type, value, traceback):
        return None

def purge(path):
    """Completely remove a file or directory.

    If path points to a file, delete it. If path points
    to a directory, recursively delete it. Otherwise, raise
    a ValueError."""

    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError('%s is not a file or directory' % path)

class Experiment:
    """Base class for defining a DeepSpeech experiment.

    run() is used to produce the experiment's results and save
    them to the disk. present() reads the experiment's results
    from the disk and returns html that presents them. The html
    returned by present() is intended to be displayed in Jupyter
    Notebook, and it is completely self-contained: it doesn't
    refer to any external files. All multimedia is base64-encoded
    and included inline in the html."""

    def __init__(self, name, prefix='experiments/', overwrite=False):
        self.path = prefix + name
        if os.path.exists(self.path):
            if overwrite:
                purge(self.path)
            else:
                raise ValueError('Path %s already exists' % self.path)
        os.makedirs(self.path)

    def run(self):
        pass

    def present(self):
        return ''

def audiofile_to_mfccs(filename):
    """Convert an audiofile to mfccs."""

    # with DeepSpeech():
    mfccs = feeding.audiofile_to_features(filename)
    with tfv1.Session() as session:
        return session.run(mfccs)[0]

def probs_to_text(probs, blank_char='_'):
    """Convert the raw probabilities returned
    by the DeepSpeech model to text."""

    alphabet = config.Config.alphabet
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

def get_cl_loss(raw_logits_tensor, labels, isolate_ids):
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=raw_logits_tensor)
    if isolate_ids is not None:
        mask = np.zeros(losses.shape)
        for isolate_id in isolate_ids:
            mask[isolate_id] = 1
        losses = mask * losses
    return tf.reduce_mean(losses)

def build_deepspeech_graph(session, mfccs_tensor, ignore_var_names=[]):
    expanded_mfccs = tf.expand_dims(mfccs_tensor, 0)
    input_tensor = train.create_overlapping_windows(expanded_mfccs)
    batch_size = 1
    input_lengths = tfv1.placeholder(tf.int32, [batch_size],
                                     name='input_lengths')
    ps_shape = [batch_size, config.Config.n_cell_dim]
    previous_state_c = tfv1.placeholder(tf.float32,
                                        ps_shape,
                                        name='previous_state_c')
    previous_state_h = tfv1.placeholder(tf.float32,
                                        ps_shape,
                                        name='previous_state_h')
    ps = tf.nn.rnn_cell.LSTMStateTuple(previous_state_c,
                                       previous_state_h)
    rnn_impl = train.rnn_impl_lstmblockfusedcell
    logits, layers = train.create_model(batch_x=input_tensor,
                                        batch_size=batch_size,
                                        seq_length=input_lengths,
                                        dropout=[None] * 6,
                                        previous_state=ps,
                                        overlap=False,
                                        rnn_impl=rnn_impl)
    train.load_graph_for_evaluation(session, ignore_var_names)
    zeros = np.zeros([1, config.Config.n_cell_dim])
    feed_dict = {input_lengths: [mfccs_tensor.shape[0]],
                 previous_state_c: zeros,
                 previous_state_h: zeros}
    return logits, layers, feed_dict

def forever():
    """Convenience generator for iterating over integers from 0 onwards."""

    i = 0
    while True:
        yield i
        i += 1

class SigmoidMaskExperiment(Experiment):
    def __init__(self, filename, name='sigmoid-mask',
                 isolate_ids=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.filename = filename
        self.isolate_ids = isolate_ids

    def run(self):
        with DeepSpeech():
            with tfv1.Session(config=config.Config.session_config) as session:
                mfccs = audiofile_to_mfccs(self.filename)
                mfccs_tensor = tf.placeholder(tf.float32, mfccs.shape)
                logits, layers, feed_dict = build_deepspeech_graph(session,
                                                                   mfccs_tensor)

                probs = tf.nn.softmax(tf.squeeze(logits))
                feed_dict[mfccs_tensor] = mfccs
                probs_evaled = probs.eval(feed_dict=feed_dict, session=session)
        with DeepSpeech():
            with tfv1.Session(config=config.Config.session_config) as session:
                mfccs = audiofile_to_mfccs(self.filename)
                mask_var = tf.Variable(# np.random.randn(*mfccs.shape),
                                       5 * np.ones(mfccs.shape),
                                       dtype=tf.float32, name='mask_var')
                session.run(tf.variables_initializer([mask_var]))
                mask_sigmoid = tf.math.sigmoid(mask_var)
                mfccs_tensor = tf.placeholder(tf.float32, mfccs.shape)
                masked_mfccs = mask_sigmoid * mfccs_tensor
                logits, layers, feed_dict = build_deepspeech_graph(session,
                                                                   masked_mfccs,
                                                                   ['mask_var'])
                logits = tf.squeeze(logits)
                cl_loss = get_cl_loss(logits, probs_evaled, self.isolate_ids)
                lightness = tf.norm(mask_sigmoid)
                loss = cl_loss + lightness
                optimizer = tf.train.AdamOptimizer()
                optimizer_op = optimizer.minimize(loss, var_list=[mask_var])
                session.run(tf.variables_initializer(optimizer.variables()))
                probs = tf.nn.softmax(logits)
                feed_dict[mfccs_tensor] = mfccs
                for i in forever():
                    n = 10 if ON_GCE else 2
                    if i % n == 0:
                        cl_loss_evaled = cl_loss.eval(feed_dict=feed_dict,
                                                      session=session)
                        lightness_evaled = lightness.eval(feed_dict=feed_dict,
                                                          session=session)
                        loss_evaled = cl_loss_evaled + lightness_evaled
                        probs_evaled = probs.eval(feed_dict=feed_dict,
                                                  session=session)
                        ms_evaled = mask_sigmoid.eval(feed_dict=feed_dict,
                                                      session=session)
                        decode = ds_ctcdecoder.ctc_beam_search_decoder
                        decoded = decode(probs_evaled,
                                         config.Config.alphabet,
                                         flags.FLAGS.beam_width,
                                         cutoff_prob=flags.FLAGS.cutoff_prob,
                                         cutoff_top_n=flags.FLAGS.cutoff_top_n)
                        print('cl_loss_evaled = %f' % cl_loss_evaled)
                        print('lightness_evaled = %f' % lightness_evaled)
                        print('loss_evaled = %f' % loss_evaled)
                        print('raw text = "' + probs_to_text(probs_evaled) + '"')
                        print('decoded text = "' + decoded[0][1] + '"')
                        print('Saving mask to disk')
                        np.save(self.path + '/%d.npy' % (i / n), ms_evaled)
                        print('=' * 64)
                        print('Running optimization step %d' % i)
                    return
                    session.run(optimizer_op, feed_dict=feed_dict)

def demo(current_experiment_name='stew-word'):
    def do_experiment(name, type, isolate_ids=None):
        experiment_name = '%s-%s' % (name, type)
        if experiment_name != current_experiment_name:
            return
        print('Running experiment "%s"' % experiment_name)
        e = SigmoidMaskExperiment('meeting-3-audio/%s.wav' % name,
                                  experiment_name,
                                  isolate_ids=isolate_ids,
                                  overwrite=False)
        e.run()
    # Word: danes
    # Vowel: a
    do_experiment('danes', 'sentence')
    do_experiment('danes', 'word', range(147, 170))
    do_experiment('danes', 'vowel', [155])
    # Word: deepspeech
    # Vowel: second ee
    do_experiment('deepspeech', 'sentence')
    do_experiment('deepspeech', 'word', range(23, 53))
    do_experiment('deepspeech', 'vowel', range(45, 49))
    # Word: flowers
    # Vowel: o
    do_experiment('flowers', 'sentence')
    do_experiment('flowers', 'word', range(267, 299))
    do_experiment('flowers', 'vowel', [281])
    # Word: four
    # Vowel: o
    do_experiment('four', 'sentence')
    do_experiment('four', 'word', range(148, 156))
    do_experiment('four', 'vowel', [151])
    # Word: stew
    # Vowel: e
    do_experiment('stew', 'sentence')
    do_experiment('stew', 'word', range(89, 102))
    do_experiment('stew', 'vowel', [99])

def mel_to_f(mel):
    return 700 * (np.exp(mel / 1127) - 1)

def f_to_mel(f):
    return 1127 * np.log(1 + f / 700)

with DeepSpeech():
    # These numbers come from the audio_to_features function in
    # DeepSpeech/training/deepspeech_training/util/feeding.py
    UPPER_FREQUENCY_LIMIT = flags.FLAGS.audio_sample_rate / 2
    LOWER_FREQUENCY_LIMIT = 20

def config_labels_and_ticks(ax, num_mfccs, num_features, duration):
    upper_mel = f_to_mel(UPPER_FREQUENCY_LIMIT)
    lower_mel = f_to_mel(LOWER_FREQUENCY_LIMIT)
    def xformat(value, pos):
        return '%.2f' % (value * duration / num_mfccs)
    def yformat(value, pos):
        value = num_features - 1 - value
        step = (upper_mel - lower_mel) / (num_features - 1)
        mel = lower_mel + value * step
        return '%.2f' % mel_to_f(mel)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(xformat))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(yformat))
    ax.set_yticks(np.arange(0, num_features, 5))
    # ax.set_yticks([0, 2, 4])
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')

def draw_mfccs(fig, ax, mfccs, duration, title='MFCCs', vmin=None, vmax=None):
    if vmin is None:
        vmin = mfccs.min()
    if vmax is None:
        vmax = mfccs.max()
    im = ax.imshow(np.flip(np.swapaxes(mfccs, 0, 1), 0),
                   aspect='auto',
                   interpolation='nearest',
                   cmap='coolwarm',
                   vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, orientation='vertical')
    config_labels_and_ticks(ax, *mfccs.shape, duration)
    ax.set_title(title)

def draw_mask(fig, ax, mask, duration, title='Mask'):
    im = ax.imshow(np.flip(np.swapaxes(mask, 0, 1), 0),
                   aspect='auto',
                   interpolation='nearest',
                   cmap='binary_r', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, orientation='vertical')
    config_labels_and_ticks(ax, *mask.shape, duration)
    ax.set_title(title)

def make_audiofile_figure(filename, filename_filtered, mask):
    y, sr = librosa.load(filename)
    y_filtered = librosa.load(filename_filtered)[0]
    duration = len(y) / sr
    mfccs = audiofile_to_mfccs(filename)
    mfccs_filtered = audiofile_to_mfccs(filename_filtered)
    vmin = min(mfccs.min(), mfccs_filtered.min())
    vmax = max(mfccs.max(), mfccs_filtered.max())
    # vmin = mfccs.min()
    # vmax = mfccs.max()
    size = 8
    fig, axes = plt.subplots(4, figsize=(size, size), dpi=100)
    draw_mfccs(fig, axes[0], mfccs, duration,
               vmin=vmin, vmax=vmax)
    draw_mask(fig, axes[1], mask, duration)
    draw_mfccs(fig, axes[2], mask * mfccs, duration,
               'MFCCs with mask applied',
               vmin=vmin, vmax=vmax)
    draw_mfccs(fig, axes[3], mfccs_filtered, duration,
               'MFCCs of filtered audio',
               vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.show()

def apply_mask_to_audio(mask, y, sr):
    y = np.copy(y)
    if len(mask.shape) == 1:
        num_features = len(mask)
        freqs = librosa.mel_frequencies(num_features + 1,
                                        LOWER_FREQUENCY_LIMIT,
                                        UPPER_FREQUENCY_LIMIT)
        bandstop = obspy.signal.filter.bandstop
        # rng = list(range(num_features))
        # np.random.shuffle(rng)
        for i in range(num_features):
            lower_freq = freqs[i]
            upper_freq = freqs[i + 1]
            mask_value = mask[i]
            filtered = bandstop(y, lower_freq, upper_freq, sr)
            y = mask_value * y + (1 - mask_value) * filtered
    elif len(mask.shape) == 2:
        num_windows = len(mask)
        num_samples = len(y)
        step = int(np.round(num_samples / len(mask)))
        for window_id, sample_id in zip(range(num_windows),
                                        range(0, num_samples, step)):
            start, end = sample_id, sample_id + step
            y[start:end] = apply_mask_to_audio(mask[window_id],
                                               y[start:end],
                                               sr)
    return y

def transcribe_mfccs(mfccs):
    with open('stdout', 'w') as f:
        with contextlib.redirect_stdout(f):
            with DeepSpeech():
                with tfv1.Session(config=config.Config.session_config) as session:
                    # mfccs = audiofile_to_mfccs(filename)
                    mfccs_tensor = tf.placeholder(tf.float32, mfccs.shape)
                    logits, layers, feed_dict = build_deepspeech_graph(session,
                                                                    mfccs_tensor)

                    probs = tf.nn.softmax(tf.squeeze(logits))
                    feed_dict[mfccs_tensor] = mfccs
                    probs_evaled = probs.eval(feed_dict=feed_dict,
                                              session=session)
                    decode = ds_ctcdecoder.ctc_beam_search_decoder
                    decoded = decode(probs_evaled,
                                    config.Config.alphabet,
                                    flags.FLAGS.beam_width,
                                    cutoff_prob=flags.FLAGS.cutoff_prob,
                                    cutoff_top_n=flags.FLAGS.cutoff_top_n)
                    return decoded[0][1]

def transcribe_audiofile(filename):
    return transcribe_mfccs(audiofile_to_mfccs(filename))
