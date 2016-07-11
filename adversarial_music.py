# adversarial_music.py
# GANs with LSTM nodes and music data
# LSTM implementation from github.com/JonathanRaiman/theano_lstm

from os import listdir
from theano_lstm import *
import theano
import theano.tensor as T
import numpy as np
import datetime
import os

import constants
import leadsheet as ls

TRAINING_DIR = "/home/sam/misc_code/adversarial_music/data/ii-V-I_leadsheets"
OUTPUT_DIR = "/home/sam/misc_code/adversarial_music/output/" + str(datetime.datetime.now())
NUM_SAMPLES_PER_TIMESTEP = 20
MAX_NUM_BATCHES = 250

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
		octave = pitch//12 - 5 # (60 is middle C, 12 notes per octave)
		note[:, minor_third] = 1
		note[:, 4 + major_third] = 1
		note[:, 7 + octave] = 1
		note[0, 12] = 1 # articulate the first timestep
		note[1:,11] = 1 # sustain for all of the note except the first timestep
	return note

# converts a list of timesteps with the 13 bit circle of thirds encoding to a list of duration/pitch tuples
def circleofthirds_to_pitchduration(notes):
	note_splits = np.argmin(notes[:,11]) # all of the timesteps where we either rest or articulate
	durations = np.array([])
	pitches = np.array([])
	for i in range(len(note_splits)-1):
		note = notes[note_splits[i]]
		if note[10] == 1: # we have a rest, we can skip a bunch of values of i now
			np.append(pitches, None)
			np.append(durations, 10)
			while note_splits[i][10] == 1:
				i+=1
				durations[-1]+=10
		else: # we have a real note, we need to do some manipulation
			durations[i] = (note_splits[i+1] - note_splits[i]) * 10
			pitch_categories = np.argmax(note[:10])
			np.append(pitches, 60 + 12 * pitch_categories[2] + circle_of_thirds_helper(pitch_categories[0], pitch_categories[1]))
	return durations,pitches

# I can't find a good mathematical way to do this, so I hardcoded it
def circleofthirds_helper(minor, major):
	if   minor == 0 and major == 0: return 0
	elif minor == 1 and major == 1: return 1
	elif minor == 2 and major == 2: return 2
	elif minor == 3 and major == 0: return 3
	elif minor == 0 and major == 1: return 4
	elif minor == 1 and major == 2: return 5
	elif minor == 2 and major == 0: return 6
	elif minor == 3 and major == 1: return 7
	elif minor == 0 and major == 2: return 8
	elif minor == 1 and major == 0: return 9
	elif minor == 2 and major == 1: return 10
	elif minor == 3 and major == 2: return 11

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

def load_data():
	chords = []
	melodies = []
	for file in listdir(TRAINING_DIR):
		c, m = ls.parse_leadsheet(TRAINING_DIR + "/" + file)
		for i in range(len(c)):
			c[i] = c[i][1] + [c[i][0]]
		chords+= [c]
		for note in m:
			melodies += [pitchduration_to_circleofthirds(note[0], note[1])]
	return chords, melodies



# I want to start with a generator which takes a random prior and a chord and feeds it through LSTM layers to get a distribution over possible output notes for the timestep
# Then, I sample from that distribution, get notes and save the gradients to increase the probability of those notes somewhere. Then, I feed that note into the discriminator, and if it fools it, I update according to that gradient
# the issue is that a single note will always fool the discriminator. Maybe I should scale the adjustment with the number of timesteps it fools the discriminator by?

# apologies for the nonsense with lists, numpy arrays and theano tensors. Be careful about your types if modifying!
def build_generator(layer_sizes):
	rng = T.shared_randomstreams.RandomStreams()

	prior = rng.uniform(size=[13]) # the random prior to keep things interesting, let's say it's a vector of 13 numbers
	chord = T.vector('chord') # each chord is a 13 element array with one hot for notes and the root as a midi number
	

	# model - for now just one hidden layer and an output layer
	model = StackedCells(26, celltype=LSTM, layers=layer_sizes, activation=T.tanh)
	model.layers[0].in_gate2.activation = lambda x: x # we don't want to tanh the input, also in_gate2 is the layer within an LSTM layer that actually accepts input from the previous layer

	# I have no idea how to pass None into forward the first time, so I copied the code from theano_lstm to intialize the hiddens. If you call init_gen_pass, it'll use these
	init_hiddens = [(T.repeat(T.shape_padleft(layer.initial_hidden_state),
                                      T.concatenate((prior, chord)).shape[0], axis=0)
                             if T.concatenate((prior, chord)).ndim > 1 else layer.initial_hidden_state)
                            if hasattr(layer, 'initial_hidden_state') else None
                            for layer in model.layers]

	prev_hiddens = [T.vector() for layer in model.layers]

	
	# I'm trying to use this without scan. We'll see if it works
	new_hiddens = model.forward(T.concatenate((prior, chord)), prev_hiddens)

	result = softmax_circleofthirds(new_hiddens[-1][-layer_sizes[-1]:])
	# grads are a bit weird, we need to take samples first, then save the gradients with respect to the cost of the samples, but we wait to see which we apply and by how much
	# samples is a python list of theano vectors here for the sake of readability
	samples = sample_from_circleofthirds_probabilities(result, rng, NUM_SAMPLES_PER_TIMESTEP)


	# calculate the gradient with respect to each sample
	grads = [T.grad(T.sum(-T.log(result[sample==1])), model.params) for sample in samples]



	rewards = [T.scalar() for grad in grads] # the output from the discriminator: how certain the discriminator was that the generated timestep was real
	# there should be one reward per sample
	log_rewards = [T.log(reward) for reward in rewards]

	weighted_grads = []
	# I wanted to take the product of a matrix and a vector here, but grads shouldn't be a matrix, it should be a list of lists of vectors :(
	for i in range(len(grads[0])):
		weighted_grads += [sum(grads[j][i] * rewards[j] for j in range(len(grads)))]


	# copied from Jonathan Raiman's sample code, I don't need to use gsums, xsums, lr or max_norm
	# I don't need to put in costs because I have my custom gradient set up
	updates, gsums, xsums, lr, max_norm = create_optimization_updates(None, model.params, method='sgd', gradients=weighted_grads)

	
	
	# function for a generative pass - returns list of samples for the next timestep
	hiddens_len = len(new_hiddens)
	samples_len = len(samples)

	generative_pass = theano.function([chord] + prev_hiddens, samples + new_hiddens + [item for sublist in grads for item in sublist]) # some wizardry from stackoverflow
	init_gen_pass = theano.function([chord], samples + new_hiddens + [item for sublist in grads for item in sublist], 
									givens={prev_hidden:init_hidden for prev_hidden, init_hidden in zip(prev_hiddens, init_hiddens)})

	# theano doesn't like returning lists from functions, so this wrapper makes generative_pass work the way I want it to
	def generative_pass_wrapper(chord, prev_hiddens):
		if prev_hiddens != None: 
			raw_in = [chord, prev_hiddens[0]]
			for hid in prev_hiddens[1:]:
				raw_in += [hid]
		raw_output = np.array(apply(generative_pass, raw_in)) if prev_hiddens != None else np.array(init_gen_pass(chord))
		samples = raw_output[:samples_len]
		new_hiddens = raw_output[samples_len:samples_len+hiddens_len]
		grads = raw_output[samples_len+hiddens_len:]
		return samples, new_hiddens, grads


	# function for updating the generator, doesn't return anything, but applies gradients. Separate from generative_pass so that I can regulate timesteps between updates
	update_pass = theano.function([item for sublist in grads for item in sublist] + rewards, updates=updates)

	def update_pass_wrapper(grads, rewards):
		apply(update_pass, np.append(grads, rewards))

	return generative_pass_wrapper, update_pass_wrapper

# the discriminator should take in a melody timestep, a chord and its internal state and get a certainty of it being real
def build_discriminator(layer_sizes):
	chord = T.vector('chord') # should be a 13 bit vector
	melody = T.vector('melody') # 13 bits for one hot and root
	isreal = T.iscalar('isreal?') # 1 if real, 0 if fake - necessary for training

	

	model = StackedCells(26, celltype=LSTM, layers=layer_sizes, activation=T.tanh) # outputs real or fake, 2 bits
	model.layers[0].in_gate2.activation = lambda x: x

	init_hiddens = [(T.repeat(T.shape_padleft(layer.initial_hidden_state),
                                      T.concatenate((chord, melody)).shape[0], axis=0)
                             if T.concatenate((chord, melody)).ndim > 1 else layer.initial_hidden_state)
                            if hasattr(layer, 'initial_hidden_state') else None
                            for layer in model.layers]

	prev_hiddens = [T.vector() for layer in model.layers]

	new_hiddens = model.forward(T.concatenate((chord, melody)), prev_hiddens)
	raw_result = new_hiddens[-1]
	result = T.nnet.softmax(raw_result[-layer_sizes[-1]:])

	# cost is the negative log liklihood that the correct answer (real or not) was chosen, I have no idea why result is 2d
	cost = -T.mean(T.log(result[0][isreal]))

	theano.grad(cost, model.params)

	updates, gsums, xsums, lr, max_norm = create_optimization_updates(cost, model.params, method='sgd')
	forward_pass = theano.function([chord, melody] + prev_hiddens, [result] + new_hiddens)
	init_fd_pass = theano.function([chord, melody], [result] + new_hiddens, 
									givens={prev_hidden:init_hidden for prev_hidden, init_hidden in zip(prev_hiddens, init_hiddens)})

	def forward_pass_wrapper(chord, melody, prev_hiddens):
		if prev_hiddens != None: 
			raw_in = [chord, melody, prev_hiddens[0]]
			for hid in prev_hiddens[1:]:
				raw_in += [hid]
		raw_output = np.array(apply(forward_pass, raw_in)) if prev_hiddens != None else np.array(init_fd_pass(chord, melody))
		result = raw_output[0][0]
		new_hiddens = raw_output[1:]
		return result, new_hiddens

	training_pass = theano.function([chord, melody, isreal] + prev_hiddens, [cost] + new_hiddens, updates=updates, allow_input_downcast=True)
	init_tr_pass = theano.function([chord, melody, isreal], [cost] + new_hiddens, updates=updates, 
									givens={prev_hidden:init_hidden for prev_hidden, init_hidden in zip(prev_hiddens, init_hiddens)}, allow_input_downcast=True)

	def training_pass_wrapper(chord, melody, isreal, prev_hiddens):
		if prev_hiddens != None: 
			raw_in = [chord, melody, isreal, prev_hiddens[0]]
			for hid in prev_hiddens[1:]:
				raw_in += [hid]
			raw_output = apply(training_pass, raw_in)
			
		else:
			raw_output = init_tr_pass(chord, melody, isreal)
		cost = raw_output[0]
		new_hiddens = raw_output[1:]
		return cost, new_hiddens
	return forward_pass_wrapper, training_pass_wrapper


def build_and_train_GAN():
	os.mkdir(OUTPUT_DIR)
	print "Loading data..."
	chords, melodies = load_data()
	print "Building the model..."
	ggen, gupd = build_generator([50, 50, 50, circleofthirds_bits])
	dpass, dtrain = build_discriminator([50,2])

	print "Training"

	for batch in range(1, MAX_NUM_BATCHES):
		# for each piece
		print "Batch ", batch
		pause_d_training, pause_g_training = False, False
		j = 0
		for c,m in zip(chords, melodies):
			ghidden_state = None
			dhidden_state = None
			best_g = []
			worst_g = []
			j+=1
			# for each timestep
			for cstep, mstep in zip(c,m):
				# generate samples
				samples, ghidden_state, grads = ggen(cstep, ghidden_state)
				# pass them through the discriminator
				results = np.zeros(len(samples))
				for i in range(len(samples)): 
					r, _ = dpass(cstep, samples[i], dhidden_state)
					results[i] = r[1] # certainty that it is real
				# update generator based on results
				if not pause_g_training: gupd(grads, results)
				# train discriminator on the best generated timestep

				best = np.argmax(results)
				worst = np.argmin(results)
				best_g += [results[best]]
				worst_g += [results[worst]]
				dtrain(cstep, samples[best], 0, dhidden_state)
				# train discriminator on the correct timestep
				if pause_d_training: _, dhidden_state = dpass(cstep, mstep, 1, dhidden_state)
				else: dcost, dhidden_state = dtrain(cstep, mstep, 1, dhidden_state)
			#check whether we should pause training for one network based on how good/bad the generated results are relative
			avg_best = np.mean(np.array(best_g))
			avg_worst = np.mean(np.array(worst_g))
			pause_g_training = avg_worst > UPBOUND
			pause_d_training = avg_best < LOWBOUND
			if j % 100 == 0:
				print "Piece ", j
				print "Average best generated timestep: ", avg_best
				print "Average worst generated timestep: ", avg_worst


		# Every 10 batches, output a piece
		if batch % 10 == 0:
			gen_melody = []
			ghidden_state = None
			dhidden_state = None
			# for each timestep, generate a bunch of melody timesteps
			for cstep in chords[0]:
				samples, ghidden_state, _ = ggen(cstep, ghidden_state)
				# take the best one, according to the discriminator
				for i in range(len(samples)):
					r, _ = dpass(cstep, samples[i], dhidden_state)
					results[i] = r[1]
				best = np.argmax(results)
				gen_melody += samples[best]
				_, dhidden_state = dpass(cstep, samples[best], dhidden_state)
			pitchduration_melody = circleofthirds_to_pitchduration(np.array(gen_melody))
			ls.write_leadsheet(chords[0], pitchduration_melody, OUTPUT_DIR + "Batch_" + str(batch))

build_and_train_GAN()