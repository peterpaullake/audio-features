# Audio Features
Do feature visualization on [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) and [Deep Speech](https://arxiv.org/pdf/1412.5567.pdf).

# Quick links
[Intermediate outputs](https://peterpaullake.github.io/audio-features/table/index.html)

[Deterministic inference](https://peterpaullake.github.io/audio-features/deterministic-inference.html)

[Editing wave input and mel spectrogram input simultaneously](https://peterpaullake.github.io/audio-features/editing-wave-input-and-mel-spec-simultaneously.html)

[More high variance units](https://peterpaullake.github.io/audio-features/more-high-variance-units.html)

[High variance units](https://peterpaullake.github.io/audio-features/high-variance-units.html)

[Minimizing distance from expectation values](https://peterpaullake.github.io/audio-features/minimizing-distance-from-expectation-values.html)

[More WaveNet features (local conditioning features)](https://peterpaullake.github.io/audio-features/more-wavenet-features.html)

[WaveNet features](https://peterpaullake.github.io/audio-features/wavenet-features.html)

[Looking for interesting neurons 1](https://peterpaullake.github.io/audio-features/looking-for-interesting-neurons-1.html), [Looking for interesting neurons 2](https://peterpaullake.github.io/audio-features/looking-for-interesting-neurons-2.html)

[Higher contrast masks](https://peterpaullake.github.io/audio-features/higher-contrast-masks.html)

[Better importance masks and applying importance masks to audio](https://peterpaullake.github.io/audio-features/importance-masks.html)

[Original notebook](https://peterpaullake.github.io/audio-features/audio-features.html)

# Installation
Clone audio-features
```
git clone https://github.com/peterpaullake/audio-features
cd audio-features
```
Clone DeepSpeech and set up a virtual environment according to the [DeepSpeech training installation instructions](https://deepspeech.readthedocs.io/en/latest/TRAINING.html)
```
git clone -b v0.8.2 https://github.com/mozilla/DeepSpeech
python3.6 -m venv deepspeech-train-venv
source deepspeech-train-venv/bin/activate
cd DeepSpeech
pip3 install --upgrade pip==20.0.2 wheel==0.34.2 setuptools==46.1.3
pip3 install --upgrade -e .
```
To use a GPU, install tensorflow-gpu instead
```
pip3 uninstall tensorflow
pip3 install 'tensorflow-gpu==1.15.2'
```
Run `replace-checkpoints.py.sh` to replace the default DeepSpeech `checkpoints.py` with a custom version
```
cd ..
source replace-checkpoints.py.sh
```
