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
	real_melody = T.matrix('melody')

	# model - for now just one hidden layer and an output layer
	model = StackedCells(63, celltype=LSTM, layers=layer_sizes, activation=T.tanh)
	model.layers[0].in_gate2.activation = lambda x: x # we don't want to tanh the input, also in_gate2 is the layer within an LSTM layer that actually accepts input from the previous layer
	model.layers.append(Layer(layer_sizes[-1], 13, lambda x: softmax_circleofthirds(x)))
	
	def step(chord, prior, melody, *prev_hiddens):
		new_states = model.forward(T.concatenate((prior,melody,chord)), prev_hiddens)
		next_timestep = sample_from_circleofthirds_probabilities(new_states[-1], rng, 1)
		return next_timestep + new_states

	gen_results, gen_updates = theano.scan(step, n_steps=chords.shape[0], 
		outputs_info=[T.zeros(13)] +[dict(initial=layer.initial_hidden_state, taps=[-1]) if hasattr(layer, 'initial_hidden_state') else None for layer in model.layers],
		sequences=[chords, prior])

	results, updates1 = theano.scan(step, n_steps=chords.shape[0], 
		outputs_info=[None] +[dict(initial=layer.initial_hidden_state, taps=[-1]) if hasattr(layer, 'initial_hidden_state') else None for layer in model.layers],
		sequences=[chords, prior, real_melody])

	generative_pass = theano.function([chords], [gen_results[0]], updates=gen_updates, allow_input_downcast=True)

	# this one is wrapped in order to avoid sampling nonsense within theano scan
	def generative_pass_wrapper(chords):
		generated_m = generative_pass(chords)[0]
		#generated_m = np.stack(map(sample_from_cot, probs))
		#generated_m = generated_m.reshape(generated_m.shape[0], generated_m.shape[2])
		return generated_m


	quality_of_each_output_for_each_timestep = T.matrix('quality')

	log_results = T.log(results[-1])

	log_probability_of_each_output_for_each_timestep = T.dot(log_results, np.swapaxes(gen_possible_circleofthirds_notes(), 0,1))

	ev_of_each_timestep = T.sum(log_probability_of_each_output_for_each_timestep * quality_of_each_output_for_each_timestep, axis=1)

	cost = -T.sum(ev_of_each_timestep)

	updates, gsums, xsums, lr, max_norm = create_optimization_updates(cost, model.params, method=TRAINING_METHOD)


	training_pass = theano.function([chords, real_melody, quality_of_each_output_for_each_timestep], [cost], updates=updates + updates1, 
		mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True), allow_input_downcast=True)

	transparent_tr_pass = theano.function([chords, real_melody, quality_of_each_output_for_each_timestep], 
		[cost, log_probability_of_each_output_for_each_timestep], updates=updates + updates1, allow_input_downcast=True)

	def training_pass_wrapper(chords, real_melody, quality_of_each_output_for_each_timestep, transparent=False):
		return transparent_tr_pass(chords, real_melody, quality_of_each_output_for_each_timestep) if transparent else training_pass(chords, real_melody, quality_of_each_output_for_each_timestep)

	return model, generative_pass_wrapper, training_pass_wrapper

# the discriminator should take in a melody timestep, a chord and its internal state and get a certainty of it being real
def build_discriminator(layer_sizes):
	chord = T.matrix('chord') # should be a 25 bit vector for each timestep
	melody = T.matrix('melody') # 13 bits for circleofthirds
	isreal = T.iscalar('isreal?') # 1 if real, 0 if fake - necessary for training

	paddedmelody = T.concatenate((T.zeros([1,13]), melody))

	model = StackedCells(38, celltype=LSTM, layers=layer_sizes, activation=T.tanh) # outputs real or fake, 2 bits
	model.layers[0].in_gate2.activation = lambda x: x
	model.layers.append(Layer(layer_sizes[-1], 2, lambda x: T.nnet.softmax(x)))

	def step(c, prev_m, m, possible_circleofthirds_notes, *prev_hiddens):
		possible_circleofthirds_notes = T.set_subtensor(possible_circleofthirds_notes[-1,:10], prev_m[:10])
		new_hiddens = model.forward(T.concatenate((m,c)), prev_hiddens)
		quality_of_each_output = T.zeros(38) # Magic Number, beware changing note encoding
		for i in range(38):
			# measure the quality of each possible output
			quality_of_each_output = T.set_subtensor(quality_of_each_output[i], 
				model.forward(T.concatenate((possible_circleofthirds_notes[i], c)), prev_hiddens)[-1][0][1])
		return [quality_of_each_output, possible_circleofthirds_notes] + new_hiddens

	results, updates = theano.scan(step, n_steps=chord.shape[0],
		outputs_info=[None, gen_possible_circleofthirds_notes()] + [dict(initial=layer.initial_hidden_state, taps=[-1]) if hasattr(layer, 'initial_hidden_state') else None for layer in model.layers],
		sequences=[T.concatenate((chord, T.zeros([1,25]))), dict(input=paddedmelody, taps=[0,1])])
	# function that returns the quality of each possible output in the next timestep
	predictive_pass = theano.function([chord, melody], [results[0]], allow_input_downcast=True)
	forward_pass = theano.function([chord, melody], [results[-1][:,0,1]], allow_input_downcast=True)

	correct_answer_probs = results[-1][:,0,isreal]
	cost = -T.sum(T.log(correct_answer_probs))

	updates, gsums, xsums, lr, max_norm = create_optimization_updates(cost, model.params, method=TRAINING_METHOD)
	
	training_pass = theano.function([chord, melody, isreal], [cost], updates=updates, allow_input_downcast=True)

	return model, predictive_pass, forward_pass, training_pass


# generate output from the network: chords are the chords to solo over, ggen is a function like generative_pass_wrapper above, dpass is like forward_pass_wrapper above
def generate_sample_output(chords, ggen, dpass, batch, output_directory=OUTPUT_DIR):

	# for each timestep, generate a bunch of melody timesteps
	ghidden_state = None
	dhidden_state = None
	gen_melody= ggen(chords[0])
	pitchduration_melody = circleofthirds_to_pitchduration(np.int32(np.array(gen_melody)))
	for i in range(len(pitchduration_melody)):
		if pitchduration_melody[i][0] is not None:
			pitchduration_melody[i] = (int(pitchduration_melody[i][0]), pitchduration_melody[i][1])
	chords_for_ls = [(circleofthirds_to_pitchduration(np.array([chord[12:]]))[0][0]-60, list(chord[:12])) for chord in chords[0]]
	ls.write_leadsheet(chords_for_ls, pitchduration_melody, output_directory + "/Batch_" + str(batch) + ".ls")

def build_and_train_GAN(training_data_directory=TRAINING_DIR, gweight_file=None, dweight_file=None, start_batch=1):
	
	print "Loading data..."
	chords, melodies = load_data(training_data_directory)
	print "Building the model..."
	rng = T.shared_randomstreams.RandomStreams()
	gmodel, ggen, gupd = build_generator([100, 200, 100], rng)
	dmodel, dpred, dpass, dtrain = build_discriminator([50,100,50])

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
	gcost_init = 0
	dcost_init = 0
	piece_length = 12
	print "Training"
	for batch in range(start_batch, MAX_NUM_BATCHES):

		# Every 10 batches, store weights and output a piece
		if batch %10 == 0:
			save_weights(gmodel, OUTPUT_DIR + "/Gweights_Batch_" + str(batch) + ".p")
			save_weights(dmodel, OUTPUT_DIR + "/Dweights_Batch_" + str(batch) + ".p")
			generate_sample_output(chords, ggen, dpass, batch)

		print "Batch ", batch
		dpause, gpause = False, False
		j = 0
		p = True
		# for each piece
		for c,m in zip(chords, melodies):
			c = c[:piece_length]
			m = m[:piece_length]
			ghidden_state = None
			dhidden_state = None
			best_g = []
			worst_g = []
			j+=1
			
			# expected results at each timestep for discriminator

			# train the discriminator
			if not dpause:
				generated_m = ggen(c)
				dcost_real = dtrain(c, m, 1)[0]
				dcost_generated = dtrain(c, generated_m, 0)[0]
				dcost_avg = (dcost_real + dcost_generated) / 2
				if dcost_init == 0: dcost_init = dcost_avg
				
			# train the generator
			if not gpause:
				quality_of_each_output_for_each_timestep = dpred(c, m)[0]
				gcost = gupd(c, m, quality_of_each_output_for_each_timestep)[0]
				if gcost_init == 0: gcost_init = gcost
			# pause if either one is super far ahead
			dpause = dcost_avg/ dcost_init < 0.75 * gcost / gcost_init
			gpause = gcost/ gcost_init < 0.75 * dcost_avg / dcost_init

			if gcost < -np.log(0.5) * piece_length and piece_length < len(chords[0]): piece_length+=12

			
			if j % 300 == 0:
				_, prob = gupd(c, m, quality_of_each_output_for_each_timestep, True)
				plot_pitches_over_time(quality_of_each_output_for_each_timestep, "batch "+ str(batch)+ "discriminator output: real" + str(j), OUTPUT_DIR)
				quality_of_each_output_for_each_timestep = dpred(c, generated_m)[0]
				plot_pitches_over_time(quality_of_each_output_for_each_timestep, "batch "+ str(batch)+ "discriminator output: generated" + str(j), OUTPUT_DIR)
				plot_pitches_over_time(np.exp(prob), "batch "+ str(batch)+ "generator output" + str(j), OUTPUT_DIR)
				print "Piece ", j
				print "Gcost ", gcost
				print "Dcost (real): ", dcost_real
				print "Dcost (generated): ", dcost_generated
		
			

if __name__=='__main__':
	build_and_train_GAN()
