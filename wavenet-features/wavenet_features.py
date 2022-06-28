import numpy as np
import os
from shutil import rmtree
from soundfile import write as write_audio
import torch
from tqdm import tqdm

cwd = os.getcwd()
os.chdir(os.path.join(cwd, 'pytorch-wavenet'))
import audio_data
import wavenet_model
import wavenet_training
os.chdir(cwd)

def generate_triangle(n_classes, n_samples, period_multiplier):
    assert(type(period_multiplier) == int)
    assert(period_multiplier >= 1)

    natural_triangle = np.concatenate(
        [np.arange(n_classes - 1, dtype=np.uint8),
         n_classes - 1 - np.arange(n_classes - 1, dtype=np.uint8)])
    natural_period = len(natural_triangle)

    sample_ids = np.arange(n_samples, dtype=float)

    # Scale the frequency
    sample_ids /= period_multiplier

    # Add a random phase shift
    sample_ids += np.random.randint(period_multiplier * natural_period)

    # Make sure the ids are in [0, natural_period)
    sample_ids %= natural_period

    # Round down to the nearest integer
    sample_ids = sample_ids.astype(int)

    return natural_triangle[sample_ids]

def generate_triangle_dataset(npz_path, n_classes, n_samples, period_multiplier, n_examples=400):
    try:
        os.remove(npz_path)
    except FileNotFoundError:
        pass

    examples = []
    for i in tqdm(range(n_examples)):
        examples.append(generate_triangle(n_classes, n_samples, period_multiplier))
    np.savez(npz_path, *examples)

def generate_datasets():
    duration_secs = 4
    sample_rate = 16000
    n_samples = duration_secs * sample_rate

    for period_multiplier in [1, 2, 4, 8]:
        generate_triangle_dataset(
            f'triangle-{period_multiplier}.npz',
            n_classes=256, n_samples=n_samples,
            period_multiplier=period_multiplier,
            n_examples=400)

def create_model():
    return wavenet_model.WaveNetModel(
        layers=10,
        blocks=3,
        dilation_channels=32,
        residual_channels=32,
        skip_channels=1024,
        end_channels=512, 
        classes=256,
        output_length=16,
        dtype=torch.FloatTensor,
        bias=True)

def train(use_cuda=True):
    dtype = torch.FloatTensor # data type
    ltype = torch.LongTensor # label type

    if use_cuda:
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor

    for dataset_file in [f for f in os.listdir() if f.startswith('triangle')]:
        model = create_model()
        data = wavenet_model.WavenetDataset(
            dataset_file=dataset_file,
            item_length=model.receptive_field + model.output_length - 1,
            target_length=model.output_length,
            test_stride=100)

        logger = wavenet_training.Logger()

        trainer = wavenet_training.WavenetTrainer(
            model=model.cuda() if use_cuda else model,
            dataset=data,
            lr=0.001,
            snapshot_path='snapshots',
            snapshot_name=dataset_file.split('.')[0],
            snapshot_interval=1000,
            logger=logger,
            dtype=dtype,
            ltype=ltype)

        trainer.train(batch_size=8, epochs=1) # batch_size=4, epochs=2)

def generate(use_trained=True):
    if use_trained:
        model = wavenet_model.load_latest_model_from('snapshots', use_cuda=False)
    else:
        model = create_model()

    def prog_callback(step, total_steps):
        print(str(100 * step // total_steps) + '% generated')

    # Use a triangle wave prefix
    first_samples = generate_triangle(256, 2 * (256 - 1), period_multiplier=1)
    first_samples = first_samples[:len(first_samples) // 4]
    first_samples = torch.LongTensor(first_samples)

    result = model.generate_fast(
        num_samples=round(1024),
        first_samples=first_samples,
        temperature=0.1,
        regularize=0.0,
        progress_callback=prog_callback,
        progress_interval=100)

    return result
