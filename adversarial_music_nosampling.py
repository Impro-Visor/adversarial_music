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
import pickle
import matplotlib.pyplot as plt

import constants
import leadsheet as ls
from adversarial_music import * 

TRAINING_DIR = "/home/sam/misc_code/adversarial_music/data/ii-V-I_leadsheets"
OUTPUT_DIR = "/home/sam/misc_code/adversarial_music/output/nosampling" + str(datetime.datetime.now())
NUM_SAMPLES_PER_TIMESTEP = 1 # when 
MAX_NUM_BATCHES = 250 # the program will halt training after this number of passes through the data
VERBOSE = True # whether or not plots will appear every 10 epochs
OUTPUT = True # whether or not weights/outputs will be saved every 10 epochs
TRAINING_METHOD = 'adadelta' # optimizer to use when training

DIFF_BOUND = .3

circleofthirds_bits = 13


def gen_possible_circleofthirds_notes():
	note_possibilities = np.zeros([4*3*3+2, 13])
	n = 0
	for i in range(4):
		for j in range(4,7):
			for k in range(7,10):
				note_possibilities[n][i] = 1
				note_possibilities[n][j] = 1
				note_possibilities[n][k] = 1
				note_possibilities[n][12] = 1
				n+=1
	note_possibilities[n][10] = 1
	n += 1
	note_possibilities[n][11] = 1
	return note_possibilities


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

	samples = sample_from_circleofthirds_probabilities(result, rng, NUM_SAMPLES_PER_TIMESTEP)

	generative_pass = theano.function([chord] + prev_hiddens, samples + new_hiddens, allow_input_downcast=True) # no gradients returned 
	init_gen_pass = theano.function([chord], samples + new_hiddens, 
									givens={prev_hidden:init_hidden for prev_hidden, init_hidden in zip(prev_hiddens, init_hiddens)}, allow_input_downcast=True)

	samples_len = len(samples)

	def generative_pass_wrapper(chord, prev_hiddens=None):
		if prev_hiddens is not None: 
			raw_in = [chord, prev_hiddens[0]]
			for hid in prev_hiddens[1:]:
				raw_in += [hid]
		raw_output = np.array(apply(generative_pass, raw_in)) if prev_hiddens is not None else np.array(init_gen_pass(chord))
		samples = raw_output[:samples_len]
		new_hiddens = raw_output[samples_len:]
		return samples, new_hiddens

	# now we want to do gradient with respect to some given desired output
	quality_of_each_output = T.vector('quality')

	log_probability_of_each_output = T.dot(T.log(result), np.swapaxes(gen_possible_circleofthirds_notes(), 0,1)) # matrix times a vector is elegant solution here

	cost = -T.dot(quality_of_each_output, log_probability_of_each_output)

	updates, gsums, xsums, lr, max_norm = create_optimization_updates(cost, model.params, method=TRAINING_METHOD)

	training_pass = theano.function([chord, quality_of_each_output] + prev_hiddens, [cost] + new_hiddens, updates=updates, allow_input_downcast=True)
	init_tr_pass = theano.function([chord, quality_of_each_output], [cost] + new_hiddens, updates=updates, 
		givens={prev_hidden:init_hidden for prev_hidden, init_hidden in zip(prev_hiddens, init_hiddens)}, allow_input_downcast=True)
	# returns the log probability of each output as well, used for making graphs
	transparent_tr_pass = theano.function([chord, quality_of_each_output] + prev_hiddens, 
		[cost, log_probability_of_each_output] + new_hiddens, updates=updates, allow_input_downcast=True)
	transparent_init_tr_pass = theano.function([chord, quality_of_each_output], [cost, log_probability_of_each_output] + new_hiddens, updates=updates, 
		givens={prev_hidden:init_hidden for prev_hidden, init_hidden in zip(prev_hiddens, init_hiddens)}, allow_input_downcast=True)

	def training_pass_wrapper(chord, quality_of_each_output, prev_hiddens=None, transparent=False):
		if prev_hiddens is not None: 
			raw_in = [chord, quality_of_each_output, prev_hiddens[0]]
			for hid in prev_hiddens[1:]:
				raw_in += [hid]
		if transparent:
			raw_output = apply(transparent_tr_pass, raw_in) if prev_hiddens is not None else transparent_init_tr_pass(chord, quality_of_each_output)
			cost = raw_output[0]
			log_probability_of_each_output = raw_output[1]
			new_hiddens = raw_output[2:]
			return cost, log_probability_of_each_output, new_hiddens
		else:
			raw_output = apply(training_pass, raw_in) if prev_hiddens is not None else init_tr_pass(chord, quality_of_each_output)
			cost = raw_output[0]
			new_hiddens = raw_output[1:]
			return cost, new_hiddens
	return model, generative_pass_wrapper, training_pass_wrapper

	

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
	cost = -T.log(result[0][isreal])

	theano.grad(cost, model.params)

	updates, gsums, xsums, lr, max_norm = create_optimization_updates(cost, model.params, method=TRAINING_METHOD)
	forward_pass = theano.function([chord, melody] + prev_hiddens, [result] + new_hiddens, allow_input_downcast=True)
	init_fd_pass = theano.function([chord, melody], [result] + new_hiddens, 
									givens={prev_hidden:init_hidden for prev_hidden, init_hidden in zip(prev_hiddens, init_hiddens)}, allow_input_downcast=True)

	def forward_pass_wrapper(chord, melody, prev_hiddens):
		if prev_hiddens is not None: 
			raw_in = [chord, melody, prev_hiddens[0]]
			for hid in prev_hiddens[1:]:
				raw_in += [hid]
		raw_output = np.array(apply(forward_pass, raw_in)) if prev_hiddens is not None else np.array(init_fd_pass(chord, melody))
		result = raw_output[0][0]
		new_hiddens = raw_output[1:]
		return result, new_hiddens

	training_pass = theano.function([chord, melody, isreal] + prev_hiddens, [cost] + new_hiddens, updates=updates, allow_input_downcast=True)
	init_tr_pass = theano.function([chord, melody, isreal], [cost] + new_hiddens, updates=updates, 
									givens={prev_hidden:init_hidden for prev_hidden, init_hidden in zip(prev_hiddens, init_hiddens)}, allow_input_downcast=True)

	def training_pass_wrapper(chord, melody, isreal, prev_hiddens):
		if prev_hiddens is not None: 
			raw_in = [chord, melody, isreal, prev_hiddens[0]]
			for hid in prev_hiddens[1:]:
				raw_in += [hid]
			raw_output = apply(training_pass, raw_in)
			
		else:
			raw_output = init_tr_pass(chord, melody, isreal)
		cost = raw_output[0]
		new_hiddens = raw_output[1:]
		return cost, new_hiddens
	return model, forward_pass_wrapper, training_pass_wrapper



# generate output from the network: chords are the chords to solo over, ggen is a function like generative_pass_wrapper above, dpass is like forward_pass_wrapper above
def generate_sample_output(chords, ggen, dpass, batch, output_directory=OUTPUT_DIR):
	# for each timestep, generate a bunch of melody timesteps
	ghidden_state = None
	dhidden_state = None
	gen_melody = []
	for cstep in chords:
		samples, ghidden_state = ggen(cstep, ghidden_state)
		# take the best one, according to the discriminator
		results = np.zeros(len(samples))
		for i in range(len(samples)):
			r, _ = dpass(cstep, samples[i], dhidden_state)
			results[i] = r[1]
		best = np.argmax(results)
		gen_melody += [samples[best].tolist()]
		_, dhidden_state = dpass(cstep, samples[best], dhidden_state)
	pitchduration_melody = circleofthirds_to_pitchduration(np.int32(np.array(gen_melody)))
	for i in range(len(pitchduration_melody)):
		if pitchduration_melody[i][0] is not None:
			pitchduration_melody[i] = (int(pitchduration_melody[i][0]), pitchduration_melody[i][1])
	chords_for_ls = [(chord[-1], list(chord[:-1])) for chord in chords]
	ls.write_leadsheet(chords_for_ls, pitchduration_melody, output_directory + "/Batch_" + str(batch) + ".ls")



def plot_pitches_over_time(data, label):
	fig, ax = plt.subplots()
	heatmap = ax.pcolor(data.T, cmap=plt.cm.Blues_r)
	plt.title(label)
	plt.xlabel('timesteps')
	plt.ylabel('pitches')
	if OUTPUT: plt.savefig(OUTPUT_DIR + '/' + label)
	else: plt.show()


def build_and_train_GAN(training_data_directory=TRAINING_DIR, gweight_file=None, dweight_file=None, start_batch=1):
	
	print "Loading data..."
	chords, melodies = load_data(training_data_directory)
	print "Building the model..."
	gmodel, gpass, gtrain = build_generator([100, 200, 100, circleofthirds_bits])
	dmodel, dpass, dtrain = build_discriminator([100,50, 2])

	if gweight_file is not None:
		load_model_from_weights(gmodel, gweight_file)
	if dweight_file is not None:
		load_model_from_weights(dmodel, dweight_file)

	batch = 0

	gen_melody = []
	ghidden_state = None
	dhidden_state_real = None
	dhidden_state_generated = None
	if OUTPUT: os.mkdir(OUTPUT_DIR)

	print "Training"

	every_possible_output = gen_possible_circleofthirds_notes() # NOTE: set every_possible_output[-1]'s first 10 bits to the previous timestep's

	for batch in range(start_batch, MAX_NUM_BATCHES):
		# Every 10 batches, store weights and output a piece
		if batch %10 == 0 and OUTPUT:
			save_weights(gmodel, OUTPUT_DIR + "/Gweights_Batch_" + str(batch) + ".p")
			save_weights(dmodel, OUTPUT_DIR + "/Dweights_Batch_" + str(batch) + ".p")
			generate_sample_output(chords[0], gpass, dpass, batch)
		print "Batch ", batch
		pause_d_training, pause_g_training = False, False
		j = 0
		p = True
		# for each piece
		for c,m in zip(chords, melodies):
			ghidden_state = None
			dhidden_state_real = None
			dhidden_state_generated = None
			best_g = []
			worst_g = []
			j+=1

			if VERBOSE and batch % 10 == 0 and p:
				quality = np.zeros([len(c), len(every_possible_output)])
				log_probability_across_time = np.zeros([len(c), len(every_possible_output)])
				k = 0
			# for each timestep
			for cstep, mstep in zip(c,m):
				# so we're not generating samples from the generator, so we train the discriminator first
				results = np.zeros(every_possible_output.shape[0])
				for i in range(len(every_possible_output)):
					r, _ = dpass(cstep, every_possible_output[i], dhidden_state_generated)
					results[i] = r[1]
				if not pause_g_training:
					if VERBOSE and batch % 10 == 0 and p:
						gcost, log_probability, _ = gtrain(cstep, results, ghidden_state, transparent=True)
						quality[k] = results
						log_probability_across_time[k] = log_probability
						k+=1
					else: 
						gcost, _ = gtrain(cstep, results, ghidden_state)
				sample, ghidden_state = gpass(cstep, ghidden_state)
				if not pause_d_training:
					dcost = [0,0]
					dcost[0], dhidden_state_generated = dtrain(cstep, sample[0], 0, dhidden_state_generated)
					dcost[1], dhidden_state_real = dtrain(cstep, mstep, 1, dhidden_state_real)

				
				every_possible_output[-1][:10] = sample[0][:10] 
				
			if VERBOSE and batch % 10 == 0 and p:
				plot_pitches_over_time(quality, 'discriminator_output_epoch' + str(batch))
				plot_pitches_over_time(log_probability_across_time, 'generator_probs_epoch' + str(batch))
				p = False


			#check whether we should pause training for one network based on how good/bad the generated results are relative
			#pause_g_training = dcost - gcost > DIFF_BOUND
			#pause_d_training = gcost - dcost > DIFF_BOUND
			if j % 100 == 0 and VERBOSE:
				print "Piece ", j
				if not pause_g_training: print "Gcost: ", gcost
				if not pause_d_training: print "Dcost: ", dcost


		
			

if __name__=='__main__':
	build_and_train_GAN()#gweight_file='/home/sam/misc_code/adversarial_music/output/2016-07-14 16:11:37.997571/Dweights_Batch_10.p', 
						#dweight_file='/home/sam/misc_code/adversarial_music/output/2016-07-14 16:11:37.997571/Gweights_Batch_10.p')
