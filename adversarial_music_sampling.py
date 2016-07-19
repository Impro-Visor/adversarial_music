# adversarial_music_sampling


from adversarial_music import *

TRAINING_DIR = "/home/sam/misc_code/adversarial_music/data/ii-V-I_leadsheets"
OUTPUT_DIR = "/home/sam/misc_code/adversarial_music/output/" + str(datetime.datetime.now())
NUM_SAMPLES_PER_TIMESTEP = 20
MAX_NUM_BATCHES = 250
VERBOSE = False
TRAINING_METHOD = 'adadelta'

UPBOUND = .65
LOWBOUND = .35

circleofthirds_bits = 13

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

	mean = T.mean(T.stack(rewards))

	weighted_grads = []
	# I wanted to take the product of a matrix and a vector here, but grads shouldn't be a matrix, it should be a list of lists of vectors :(
	for i in range(len(grads[0])):
		weighted_grads += [sum(grads[j][i] * rewards[j] for j in range(len(grads)))]


	# copied from Jonathan Raiman's sample code, I don't need to use gsums, xsums, lr or max_norm
	# I don't need to put in costs because I have my custom gradient set up
	updates, gsums, xsums, lr, max_norm = create_optimization_updates(None, model.params, method=TRAINING_METHOD, gradients=weighted_grads)

	
	
	# function for a generative pass - returns list of samples for the next timestep
	hiddens_len = len(new_hiddens)
	samples_len = len(samples)

	generative_pass = theano.function([chord] + prev_hiddens, samples + new_hiddens + [item for sublist in grads for item in sublist], allow_input_downcast=True) # some wizardry from stackoverflow
	init_gen_pass = theano.function([chord], samples + new_hiddens + [item for sublist in grads for item in sublist], 
									givens={prev_hidden:init_hidden for prev_hidden, init_hidden in zip(prev_hiddens, init_hiddens)}, allow_input_downcast=True)

	# theano doesn't like returning lists from functions, so this wrapper makes generative_pass work the way I want it to
	def generative_pass_wrapper(chord, prev_hiddens):
		if prev_hiddens is not None: 
			raw_in = [chord, prev_hiddens[0]]
			for hid in prev_hiddens[1:]:
				raw_in += [hid]
		raw_output = np.array(apply(generative_pass, raw_in)) if prev_hiddens is not None else np.array(init_gen_pass(chord))
		samples = raw_output[:samples_len]
		new_hiddens = raw_output[samples_len:samples_len+hiddens_len]
		grads = raw_output[samples_len+hiddens_len:]
		return samples, new_hiddens, grads


	# function for updating the generator, doesn't return anything, but applies gradients. Separate from generative_pass so that I can regulate timesteps between updates
	update_pass = theano.function([item for sublist in grads for item in sublist] + rewards, updates=updates)

	def update_pass_wrapper(grads, rewards):
		apply(update_pass, np.append(grads, rewards))

	return model, generative_pass_wrapper, update_pass_wrapper

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
		samples, ghidden_state, _ = ggen(cstep, ghidden_state)
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




def build_and_train_GAN(training_data_directory=TRAINING_DIR, gweight_file=None, dweight_file=None, start_batch=1):
	
	print "Loading data..."
	chords, melodies = load_data(training_data_directory)
	print "Building the model..."
	gmodel, ggen, gupd = build_generator([100, 200, 100, circleofthirds_bits])
	dmodel, dpass, dtrain = build_discriminator([100,50, 2])

	if gweight_file is not None:
		load_model_from_weights(gmodel, gweight_file)
	if dweight_file is not None:
		load_model_from_weights(dmodel, dweight_file)


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
			generate_sample_output(chords[0], ggen, dpass, batch)
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
				if not pause_d_training: dtrain(cstep, samples[best], 0, dhidden_state)
				# train discriminator on the correct timestep
				if pause_d_training: _, dhidden_state = dpass(cstep, mstep, dhidden_state)
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
				if not pause_d_training: print "Current dcost: ", dcost


		
			

if __name__=='__main__':
	build_and_train_GAN(start_batch=10)#gweight_file='/home/sam/misc_code/adversarial_music/output/2016-07-15 09:26:49.172497/Gweights_Batch_10.p',
						#dweight_file='/home/sam/misc_code/adversarial_music/output/2016-07-15 09:26:49.172497/Dweights_Batch_10.p', start_batch=10)