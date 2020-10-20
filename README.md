# deepspeech-audio-features
Do feature visualization on [Mozilla's DeepSpeech](https://github.com/mozilla/DeepSpeech/).
Requires Python 3.6 and a Mac or Linux environment.

# Quick links
[Original notebook](https://peterpaullake.github.io/deepspeech-audio-features/deepspeech-audio-features.html)

[Better importance masks and applying importance masks to audio](https://peterpaullake.github.io/deepspeech-audio-features/importance-masks.html)

# Installation
Clone deepspeech-audio-features
```
git clone https://github.com/peterpaullake/deepspeech-audio-features
cd deepspeech-audio-features
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