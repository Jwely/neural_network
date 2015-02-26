# neural_network
This is a python implimentation of neural networks with back propogation 
and experimental stability metrics to speed convergence of cortex training

This was created as my own learning excersize for neural networks, but has
evolved to be of moderate complexity and high quality. It manages the 
creation and training of neural networks or "cortex's" with some diagnostic
and tracking ability. It has thus far been tested to train cortex's on fairly
simple training datasets, but has performed extremely reliably.

Users wishing to learn a little bit about neural networks can browse through
the methods of the "neuron" class first, and create a script or two to manually
pass input and output out of a single neuron with simple learning before moving 
up to use the "cortex" class to manage groups of neurons with more complex learning
by back propogation.

What this package NEEDS is testing on more complex datasets, and a new method or
two for loading lots of inputs into a trained cortex and saving the outputs with
relevant statistical information. It is presently built to import training data 
from a text file with the following format with a header

in_name1, in_name2, ... , in_nameN ; out_name1, out_name2, ... , out_nameN

in1, in2, ..., inN; out1, out2, ..., outN 

in1, in2, ..., inN; out1, out2, ..., outN

in1, in2, ..., inN; out1, out2, ..., outN 
etc...


hope you find it informative, enjoy!
