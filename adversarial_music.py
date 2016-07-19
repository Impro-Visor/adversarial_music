# adversarial_music.py
# GANs with LSTM nodes and music data
# This file contains a bunch of useful things, actual neural net implementations are in adversarial_music_sampling and nosampling

from os import listdir
from theano_lstm import *
import theano
import theano.tensor as T
import numpy as np
import datetime
import os
import pickle

import constants
import leadsheet as ls

TRAINING_DIR = "/home/sam/misc_code/adversarial_music/data/ii-V-I_leadsheets"
OUTPUT_DIR = "/home/sam/misc_code/adversarial_music/output/" + str(datetime.datetime.now())
NUM_SAMPLES_PER_TIMESTEP = 20
MAX_NUM_BATCHES = 250
VERBOSE = False
TRAINING_METHOD = 'adagrad'

UPBOUND = .65
LOWBOUND = .35

circleofthirds_bits = 13

# converts a midi number and duration to an array of generator outputs formatted in circle of thirds encoding
# first four bits correspond to minor thirds, next three correspond to major thirds - a note is specified by exactly two of these. Next three bits are octave bits, low to high
# There is also a rest bit, a sustain bit and an articulate bit (total of 13 bits for output per timestep and 48 timesteps per measure)
# if the sustain bit is on, the melody sustains. If the rest bit is on, it rests, if the articulate bit is on, we look at the other bits to figure out what to do
def pitchduration_to_circleofthirds(pitch, duration):
	note = np.zeros([duration, 13]) # our granularity is 48 timesteps/measure of 4/4, not 480
	if pitch == None:
		note[:, 10] = 1
	else:
		minor_third = pitch % 4
		major_third = pitch % 3
		octave = pitch//12 - 4 # (60 is middle C, 12 notes per octave)
		if octave > 2: 
			if VERBOSE: print "Warning, note was rounded down"
			octave = 2
		note[:, minor_third] = 1
		note[:, 4 + major_third] = 1
		note[:, 7 + octave] = 1
		note[0, 12] = 1 # articulate the first timestep
		note[1:,11] = 1 # sustain for all of the note except the first timestep
	return note

# converts a list of timesteps with the 13 bit circle of thirds encoding to a list of duration/pitch tuples
def circleofthirds_to_pitchduration(notes):
	note_splits = np.arange(len(notes))[notes[:,11] == 0] # all of the timesteps where we either rest or articulate
	note_splits = np.append(note_splits, len(notes))
	durations = np.array([])
	pitches = np.array([])
	i = 0
	while i < len(note_splits)-1:
		note = notes[note_splits[i]]
		if note[10] == 1: # we have a rest, we can skip a bunch of values of i now
			pitches = np.append(pitches, None)
			durations = np.append(durations, 0)
			while notes[note_splits[i]][10] == 1:
				i+=1
				durations[-1]+=1
				if note_splits[i] == len(notes): break
		else: # we have a real note, we need to do some manipulation
			durations = np.append(durations, note_splits[i+1] - note_splits[i])
			pitch_categories = np.int32(np.arange(10)[note[:10] == 1])
			pitches = np.append(pitches, int(48 + 12 * (pitch_categories[2]-7) + circleofthirds_helper(pitch_categories[0], pitch_categories[1])))
			if pitches[-1] is float: print "WTF????"
		i+=1
	return [(pitch, duration) for pitch, duration in zip(pitches.tolist(), np.int32(durations))]

# I can't find a good mathematical way to do this, so I hardcoded it
def circleofthirds_helper(minor, major):
	if   minor == 0 and major == 4: return 0
	elif minor == 1 and major == 5: return 1
	elif minor == 2 and major == 6: return 2
	elif minor == 3 and major == 4: return 3
	elif minor == 0 and major == 5: return 4
	elif minor == 1 and major == 6: return 5
	elif minor == 2 and major == 4: return 6
	elif minor == 3 and major == 5: return 7
	elif minor == 0 and major == 6: return 8
	elif minor == 1 and major == 4: return 9
	elif minor == 2 and major == 5: return 10
	elif minor == 3 and major == 6: return 11

def softmax_circleofthirds(arr):
	return T.concatenate((T.nnet.softmax(arr[:4]), 
									   T.nnet.softmax(arr[4:7]), 
									   T.nnet.softmax(arr[7:10]), 
									   T.nnet.softmax(arr[10:13])), axis=1)[0]

# symbolically sample from a circle of thirds representation. 
#If num_samples=1 it will return a single circle of thirds encoding, otherwise it will return an array of shape [num_samples, 13]
def sample_from_circleofthirds_probabilities(dist, rng, num_samples=1):
	sample = [T.zeros(dist.shape)]*num_samples
	minor = rng.choice(size=[num_samples], a=T.arange(0,4), p=dist[:4])
	major = rng.choice(size=[num_samples], a=T.arange(4,7), p=dist[4:7])
	octave = rng.choice(size=[num_samples], a=T.arange(7,10), p=dist[7:10])
	state = rng.choice(size=[num_samples], a=T.arange(10,13), p=dist[10:])
	for i in range(num_samples):
		sample[i] = T.set_subtensor(sample[i][minor[i]], 1)
		sample[i] = T.set_subtensor(sample[i][major[i]], 1)
		sample[i] = T.set_subtensor(sample[i][octave[i]], 1)
		sample[i] = T.set_subtensor(sample[i][state[i]], 1)
	return sample

def load_data(directory=TRAINING_DIR):
	chords = np.array([])
	melodies = np.array([])
	for file in listdir(directory):
		c, m = ls.parse_leadsheet(directory + "/" + file)
		for i in range(len(c)):
			c[i] = c[i][1] + [c[i][0]] # deal with the root of the chord
		if len(chords) == 0:
			chords = np.array([c])
		else:
			chords = np.append(chords, [c], axis=0)
		temp_melodies = np.array([])
		for note in m:
			if len(temp_melodies) == 0:
				temp_melodies = np.array(pitchduration_to_circleofthirds(note[0], note[1]))
			else:
				temp_melodies = np.append(temp_melodies, pitchduration_to_circleofthirds(note[0], note[1]), axis=0)
		if len(melodies) == 0:
			melodies = np.array([temp_melodies])
		else:
			melodies = np.append(melodies, [temp_melodies], axis=0)
	return chords, melodies



# save the weights from a model to a filepath
def save_weights(model, filepath):
	params = []
	for p in model.params:
		params += [p.get_value()]
	pickle.dump(params, open(filepath, 'w'))

# load weights saved with save_weights and assign them to the given model
# currently no way to save the network shape
def load_model_from_weights(model, filepath):
	print "loading from ", filepath
	new_params = pickle.load(open(filepath))
	for i in range(0, len(model.params)):
		assert len(new_params[i]) == model.params[i].get_value().shape[0]
		model.params[i].set_value(new_params[i])

