Adversarial Music
=================

Uses Generative Adversarial Networks (as described by Goodfellow et al) to generate novel jazz solos based on Robert Keller's iiVI leadsheets. Implementation is in theano, using Jonathan Raiman's theano_lstm library.

adversarial_music.py - miscellaneous functions used by other programs

adversarial\_music\_sampling.py - trains using an adaptation of reinforcement learning

adversarial\_music\_nosampling.py - trains using an estimation of expected value, without ever sampling from output distributions

adversarial\_music\_minibatched.py - nosampling method, done in minibatches (piece by piece) instead of timestep by timestep

Written for the Intelligent Music Software Project at Harvey Mudd College, summer 2016
