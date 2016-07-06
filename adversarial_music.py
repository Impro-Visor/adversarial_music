# adversarial_music.py
# GANs with LSTM nodes and music data
# LSTM implementation from github.com/JonathanRaiman/theano_lstm

from os import listdir
from theano_lstm import *
import theano
import theano.tensor as T

import constants

TRAINING_DIR = "todo"
NUM_SAMPLES_PER_TIMESTEP = 20

# converts a midi number and duration to an array of generator outputs formatted in circle of thirds encoding
# first four bits correspond to minor thirds, next three correspond to major thirds - a note is specified by exactly two of these. Next three bits are octave bits, low to high
# There is also a rest bit, a sustain bit and an articulate bit (total of 13 bits for output per timestep and 48 timesteps per measure)
# if the sustain bit is on, the melody sustains. If the rest bit is on, it rests, if the articulate bit is on, we look at the other bits to figure out what to do
def pitchduration_to_circleofthirds(duration, pitch):
	note = np.zeros([duration//10, 13])
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

# converts a list of timesteps with the 12 bit circle of thirds encoding to a list of duration/pitch tuples
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
	return T.append((T.nnet.softmax(arr[:4]), 
									   T.nnet.softmax(arr[4:7]), 
									   T.nnet.softmax(arr[7:10]), 
									   T.nnet.softmax(arr[10:])))

# symbolically sample from a circle of thirds representation. 
#If num_samples=1 it will return a single circle of thirds encoding, otherwise it will return an array of shape [num_samples, 13]
def sample_from_circleofthirds_probabilities(dist, rng, num_samples=1): 
	sample = T.zeros([num_samples, dist.shape])
	ones = ones_like(dist)
	minor = rng.choice(size=[num_samples], a=T.arange(0,4), dist[:4])
	major = rng.choice(size=[num_samples], a=T.arange(4,7), p=dist[4:7])
	octave = rng.choice(size=[num_samples], a=T.arange(7,10), p=dist[7:10])
	state = rng.choice(size=[num_samples], a=T.arange(10,13), p=dist[10:])
	sample[:,minor] = ones[minor]
	sample[:,major] = ones[major]
	sample[:,octave] = ones[octave]
	sample[:,state] = ones[state]
	return sample if sample.shape()[0] > 1 else sample[0]

def load_data():
	for file in listdir(TRAINING_DIR):
		pass # TODO



# I want to start with a generator which takes a random prior and a chord and feeds it through LSTM layers to get a distribution over possible output notes for the timestep
# Then, I sample from that distribution, get notes and save the gradients to increase the probability of those notes somewhere. Then, I feed that note into the discriminator, and if it fools it, I update according to that gradient
# the issue is that a single note will always fool the discriminator. Maybe I should scale the adjustment with the number of timesteps it fools the discriminator by?

def build_generator():
	rng = T.shared_randomstreams.RandomStreams()

	prior = rng.uniform(size=(12)) # the random prior to keep things interesting, let's say it's a vector of 12 numbers
	chord = T.matrix('chord') # each chord is a 12 element array with one hot for notes

	# model - for now just one hidden layer and an output layer
	model = StackedCells(24, celltype=LSTM, layers=[50, 13], activation=T.tanh)
	model.layers[0].in_gate2.activation = lambda x: x # we don't want to tanh the input, also in_gate2 is the layer within an LSTM layer that actually accepts input from the previous layer

	timesteps = chord.shape[0]
	outputs_info = [dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in model.layers if hasattr(layer, 'initial_hidden_state')] + [None]

	# using scan, we need a function to execute each timestep
	def step(input, *prev_hiddens):
		new_states = model.forward(x, prev_hiddens)
		# we output the new state of the network and the real output at each timestep to be fed back into the network
		return new_states + [softmax_circle_of_thirds(new_states[-1])]

	results, updates = theano.scan(step, n_steps=timesteps, outputs_info=outputs_info)
	# cost is a bit weird
	

	samples = sample_from_circleofthirds_probabilities(results[-1][1], rng, num_samples=NUM_SAMPLES_PER_TIMESTEP) # this should be the softmaxes from step
	costs = -T.log(T.sum(results[-1][1][samples == 1]))

	grads = [T.grad(cost, params)] for cost in costs

	rewards = T.vector('rewards') # the output from the discriminator: how certain the discriminator was that the generated timestep was real
	# there should be one reward per sample
	log_rewards = T.log(rewards)

	# dot should take the n x m matrix of gradients and multiply it by the length n vector of log rewards, then sum up the gradients with respect to each parameter 
	# scaled by the rewards, giving us a length m vector of updates
	weighted_grads = T.dot(grads, log_rewards)

	# copied from Jonathan Raiman's sample code, I don't need to use gsums, xsums, lr or max_norm
	# I don't need to put in costs because I have my custom gradient set up
	updates, gsums, xsums, lr, max_norm = create_optimization_updates(None, model.params, method='sgd', grads=weighted_grads)

	# function for a generative pass - returns list of samples for the next timestep
	# Problem: what about internal state? Do I even want to be using scan? Or should I be calling the discriminator within step? I'm currently doing this all wrong
	generative_pass = theano.function([chord], [samples,grads])

	update_pass = theano.function([grads, rewards], update=updates)