# adversarial_music_minibatched.py
# GANs with LSTM nodes and music data
# LSTM implementation from github.com/JonathanRaiman/theano_lstm

# In progress, not nearly finished making things vectorized

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
from adversarial_music import *

from theano.compile.nanguardmode import NanGuardMode

TRAINING_DIR = "/home/sam/misc_code/adversarial_music/data/ii-V-I_leadsheets"
OUTPUT_DIR = "/home/sam/misc_code/adversarial_music/output/nosamplingminibatched" + str(datetime.datetime.now())
MAX_NUM_BATCHES = 250
VERBOSE = False
TRAINING_METHOD = 'adadelta'
MINIBATCH_SIZE = 50

UPBOUND = .65
LOWBOUND = .35

circleofthirds_bits = 13
epsilon = 10e-10


# implementation using scan, only needs to be called once per piece rather than once per timestep

def build_generator(layer_sizes, rng):
	
	chords = T.matrix('chord') # each chord is a 25 element bit vector
	prior = rng.uniform(size=chords.shape) # the random prior to keep things interesting, let's say it's a vector of 25 numbers
	
	

	# model - for now just one hidden layer and an output layer
	model = StackedCells(50, celltype=LSTM, layers=layer_sizes, activation=T.tanh)
	model.layers[0].in_gate2.activation = lambda x: x # we don't want to tanh the input, also in_gate2 is the layer within an LSTM layer that actually accepts input from the previous layer
	model.layers.append(Layer(layer_sizes[-1], 13, lambda x: softmax_circleofthirds(x)))
	
	def step(chord, prior, *prev_hiddens):
		new_states = model.forward(T.concatenate((prior,chord)), prev_hiddens)
		return new_states

	results, updates = theano.scan(step, n_steps=chords.shape[0], 
		outputs_info=[dict(initial=layer.initial_hidden_state, taps=[-1]) if hasattr(layer, 'initial_hidden_state') else None for layer in model.layers],
		sequences=[chords, prior])

	generative_pass = theano.function([chords], [results[-1]], allow_input_downcast=True)

	a = T.vector()
	b = sample_from_circleofthirds_probabilities(a, rng, 1)[0]
	sample_from_cot = theano.function([a], [b])

	# this one is wrapped in order to avoid sampling nonsense within theano scan
	def generative_pass_wrapper(chords):
		generated_m = np.stack(map(sample_from_cot, generative_pass(chords)[0]))
		generated_m = generated_m.reshape(generated_m.shape[0], generated_m.shape[2])
		return generated_m


	quality_of_each_output_for_each_timestep = T.matrix('quality')

	log_results = theano.printing.Print(results[-1])(T.log(results[-1] + epsilon))

	log_probability_of_each_output_for_each_timestep = T.dot(log_results, np.swapaxes(gen_possible_circleofthirds_notes(), 0,1))

	cost = -T.prod(T.sum(log_probability_of_each_output_for_each_timestep * quality_of_each_output_for_each_timestep, axis=1)) # * is elementwise product

	updates, gsums, xsums, lr, max_norm = create_optimization_updates(cost, model.params, method=TRAINING_METHOD)

	training_pass = theano.function([chords, quality_of_each_output_for_each_timestep], [cost], updates=updates, 
		mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True), allow_input_downcast=True)

	return model, generative_pass_wrapper, training_pass

# the discriminator should take in a melody timestep, a chord and its internal state and get a certainty of it being real
def build_discriminator(layer_sizes):
	chord = T.matrix('chord') # should be a 25 bit vector for each timestep
	melody = T.matrix('melody') # 12 bits for one hot and root for each timestep
	isreal = T.iscalar('isreal?') # 1 if real, 0 if fake - necessary for training


	possible_circleofthirds_notes = gen_possible_circleofthirds_notes()

	model = StackedCells(38, celltype=LSTM, layers=layer_sizes, activation=T.tanh) # outputs real or fake, 2 bits
	model.layers[0].in_gate2.activation = lambda x: x
	model.layers.append(Layer(layer_sizes[-1], 2, lambda x: T.nnet.softmax(x)))

	def step(chord, melody, *prev_hiddens):
		new_hiddens = model.forward(T.concatenate((melody,chord)), prev_hiddens)
		quality_of_each_output = T.zeros(possible_circleofthirds_notes.shape[0])
		for i in range(len(possible_circleofthirds_notes)):
			quality_of_each_output = T.set_subtensor(quality_of_each_output[i], model.forward(T.concatenate((possible_circleofthirds_notes[i], chord)), new_hiddens)[-1][0][1])
		return [quality_of_each_output] + new_hiddens 

	results, updates = theano.scan(step, n_steps=chord.shape[0],
		outputs_info=[None] + [dict(initial=layer.initial_hidden_state, taps=[-1]) if hasattr(layer, 'initial_hidden_state') else None for layer in model.layers],
		sequences=[chord, melody])  
	# function that returns the quality of each possible output in the next timestep
	predictive_pass = theano.function([chord, melody], [results[0]], allow_input_downcast=True)
	forward_pass = theano.function([chord, melody], [results[-1][:][1]], allow_input_downcast=True)

	temp = results[-1][isreal]
	cost = -T.sum(T.log(temp))

	updates, gsums, xsums, lr, max_norm = create_optimization_updates(cost, model.params, method=TRAINING_METHOD)
	
	training_pass = theano.function([chord, melody, isreal], [cost], updates=updates, allow_input_downcast=True)

	return model, predictive_pass, forward_pass, training_pass


# generate output from the network: chords are the chords to solo over, ggen is a function like generative_pass_wrapper above, dpass is like forward_pass_wrapper above
def generate_sample_output(chords, ggen, dpass, batch, output_directory=OUTPUT_DIR):

	# for each timestep, generate a bunch of melody timesteps
	ghidden_state = None
	dhidden_state = None
	gen_melody= ggen(chords[0])
	print gen_melody
	pitchduration_melody = circleofthirds_to_pitchduration(np.int32(np.array(gen_melody)))
	for i in range(len(pitchduration_melody)):
		if pitchduration_melody[i][0] is not None:
			pitchduration_melody[i] = (int(pitchduration_melody[i][0]), pitchduration_melody[i][1])
	chords_for_ls = [(chord[-1], list(chord[:-1])) for chord in chords[0]]
	ls.write_leadsheet(chords_for_ls, pitchduration_melody, output_directory + "/Batch_" + str(batch) + ".ls")

def build_and_train_GAN(training_data_directory=TRAINING_DIR, gweight_file=None, dweight_file=None, start_batch=1):
	
	print "Loading data..."
	chords, melodies = load_data(training_data_directory)
	print "Building the model..."
	rng = T.shared_randomstreams.RandomStreams()
	gmodel, ggen, gupd = build_generator([100, 200, 100, circleofthirds_bits], rng)
	dmodel, dpred, dpass, dtrain = build_discriminator([100,50])

	a = T.vector()
	b = sample_from_circleofthirds_probabilities(a, rng, 1)[0]
	sample_from_cot = theano.function([a], [b])

	if gweight_file is not None:
		load_model_from_weights(gmodel, gweight_file)
	if dweight_file is not None:
		load_model_from_weights(dmodel, dweight_file)

	batch = 0

	gen_melody = []
	ghidden_state = None
	dhidden_state = None
	os.mkdir(OUTPUT_DIR)

	print "Training"
	for batch in range(start_batch, MAX_NUM_BATCHES):

		# Every 10 batches, store weights and output a piece
		if batch %10 == 0:
			save_weights(gmodel, OUTPUT_DIR + "/Gweights_Batch_" + str(batch) + ".p")
			save_weights(dmodel, OUTPUT_DIR + "/Dweights_Batch_" + str(batch) + ".p")
			generate_sample_output(chords, ggen, dpass, batch)

		print "Batch ", batch
		pause_d_training, pause_g_training = False, False
		j = 0
		# for each piece
		for c,m in zip(chords, melodies):
			ghidden_state = None
			dhidden_state = None
			best_g = []
			worst_g = []
			j+=1
			
			# expected results at each timestep for discriminator
			quality_of_each_output_for_each_timestep = dpred(c, m)[0]
			generated_m = ggen(c)
			# train the discriminator
			dcost_real = dtrain(c, m, 1)
			dcost_generated = dtrain(c, generated_m, 0)
			# train the generator
			gcost = gupd(c, quality_of_each_output_for_each_timestep)


			if j % 100 == 0:
				print "Piece ", j
				print "Gcost ", gcost
				print "Dcost: ", dcost_generated


		
			

if __name__=='__main__':
	build_and_train_GAN()
