#!/usr/bin/env bash

# We need to replace the DeepSpeech checkpoints.py file with our
# custom checkpoints.py file. Our checkpoints.py file is the same
# as the DeepSpeech checkpoints.py file except that our version
# doesn't try to load certain specific variables which we've added
# to the graph in order to do feature visualization.
cp checkpoints.py DeepSpeech/training/deepspeech_training/util/checkpoints.py
