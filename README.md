# neural_network
This is a python implimentation of neural networks with back propogation 
and experimental stability metrics to speed convergence of cortex training

This was created as my own learning excersize for neural networks. It manages the 
creation and training of neural networks or "cortex's" with some diagnostic
and tracking ability. It has thus far been tested to train cortex's on fairly
simple training datasets, but has performed extremely reliably.

Users wishing to learn a little bit about neural networks can browse through
the methods of the "neuron" class first, and create a script or two to manually
pass input and output out of a single neuron with simple learning before moving 
up to use the "cortex" class to manage groups of neurons with more complex learning
by back propogation.

hope you find it informative, enjoy!

### Installation
Does not require installation, presently set up only to work in local directory

Requires python 2.7

### Input/Output
What this package NEEDS is testing on more complex datasets, and a new method or
two for loading lots of inputs into a trained cortex and saving the outputs with
relevant statistical information. It is presently built to import training data
via one of two methods.

1) from a text file with the following format with a header (note the semicolons) 

in_name1, in_name2, ... , in_nameN ; out_name1, out_name2, ... , out_nameN

in1, in2, ..., inN; out1, out2, ..., outN 

in1, in2, ..., inN; out1, out2, ..., outN

in1, in2, ..., inN; out1, out2, ..., outN 
etc...

2) from two separate csv files (inputs and targets) with an equal number of rows.

### Dependencies
Python 2.7



