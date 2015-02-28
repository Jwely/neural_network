from random import random, randint, sample, randrange
import numpy

__author__ = "Jeffry Ely, jeff.ely.08@gmal.com"

class neuron:
    """
    Creates a single neuron object

    A neurons input values are stored on "dendrites" on self.dendrite
    A neurons output values are stored on the "axon" or self.axon

    the "load" method must be passed before a "fire" can occur.
    Once a firing has occured, a "learn" method may be executed
    to have the neuron adjust its weights to make future outputs
    closer to a target value.

    Children and parent neurons may be added and removed, but these
    methods should likely be called by the greater cortex object to which
    an individual neuron belongs. 
    """
    
    def __init__(self, name):
        """Initializes a single neuron"""
        
        # basic default attributes
        self.name       = name
        self.haschildren= False
        self.children   = []
        self.hasparents = False
        self.parents    = []
        
        self.t_function = "Sigmoid"     # Supports 'Sigmoid' or 'Threshold'
        self.size       = 0             # number of dendrites (input values)
        self.dec        = 2             # number of decimal digits in output
        
        self.f_age      = 0             # forward age, increased by firing
        self.r_age      = 0             # reverse age, increased by learning
        
        self.reset()                    # sets evolution attributes
        return


    def reset(self):
        """ Sets/resets the evolving attributes of the neuron"""
        
        # evolving attributes
        self.bias           = float(randrange(0,10,1))/10
        
        self.dendrite       = []        # array of inputs "dendrite"
        self.weights        = []        # array of weights
        self.pretrans       = 0         # output value before transfer fn
        self.axon           = 0         # output value after transfer fn   

        self.instability    = 0         # initial instability of neuron
        self.delta          = 0         # initial "error" value
        self.del_delta      = 0         # rate of change of delta value
        self.learning_rate  = 1         # initial learning rate of neuron
        return

        
    def load(self,cortex_inputs = False):

        """
        prepares the neuron to fire by loading dendrites.

        either reads input values from the children neurons
        or recieves direct cortex_inputs if the neuron is on the
        first layer.

        populates random weights if the weights list is shorter than
        the dendrite list. This should only happen if the number
        of cortex inputs has changed or it is the first time
        the neuron has loaded input to the cortex.

        reading from a "child" neuron does NOT require that
        the child neuron has this neuron listed as "parent".
        those connections should be managed at the cortex level.
        """
        
        del self.dendrite[:]

        for i,child in enumerate(self.children):
            self.dendrite.append(child.axon)

        if not isinstance(cortex_inputs,bool):
            self.dendrite = self.dendrite + cortex_inputs
            
        self.size = len(self.dendrite)

        # make corrections for infrequent scenarios (such as first time loadings)
        if len(self.weights) < self.size:
            self.weights = [float(randrange(-9,9,1))/10 for _ in range(self.size)]

        if len(self.weights) > self.size:
            raise AttributeError('''Too many weights,
                                    this should never happen! you suck! fix it!''')

        return self.dendrite, self.weights

            
    def fire(self):
        """ Calculates new output and places it on the nurons "axon" """
        
        self.f_age = self.f_age + 1     # forward age the neuron

        self.pretrans = self.bias
        for i,weight in enumerate(self.weights):
            self.pretrans +=  weight * self.dendrite[i]
        
        self.axon = self.forward_transfer(self.pretrans)
        
        return self.axon


    def forward_transfer(self,z):
        """Contains all transfer functions that are supported"""

        def Sig(self, z):
            """simple Sigmoid function for continous output"""
            e = 2.71828
            return round(1/(1 + e**(-z)), self.dec)

        def Thresh(self, z):
            """True/False style threshold function for categorical output"""
            if z > 0:
                return 1.0
            else:
                return 0.0
        
        if   self.t_function == "Sigmoid":
            return Sig(self, z)
        elif self.t_function == "Threshold":
            return Thresh(self, z)


    def back_transfer(self, target):
        """
        Handles back-propogation for each transfer function type

        Tracks changes in delta values to gauge instability. The sum of
        squares for the delta value and the rate of change of that
        delta value is a decent 2nd order estimate of instability.

        @param target will be "False" for neurons on hidden layers, and have
        a float value for output layer neurons.
        """

        def Sig(self, target):
            """ back propogation based on a sigmoid function"""

            Dd_Dz = self.axon * (1 - self.axon)
            
            if target:  # for neurons on the output layer
                new_delta       = Dd_Dz * (self.axon - target)

            else:       # for neurons in hidden layers
                temp_delta = 0
                for parent in self.parents:
                    parent_weight = parent.weights[parent.children.index(self)]
                    temp_delta   += parent.delta * parent_weight

                new_delta = temp_delta * Dd_Dz


            # set new delta values and calculate instability
            self.del_delta   = new_delta - self.delta
            self.delta       = new_delta
            self.instability = self.delta**2 + self.del_delta**2

            # speed the learning rate BEYOND what is typical, but cap it at 0.5. 
            self.learning_rate  = abs(self.delta)**(0.1)
            if self.learning_rate > 0.5:
                self.learning_rate = 0.5

            self.bias -= self.delta
            
            return self.delta

        def Thresh(self, target):
            """back propogation based on threshold function (outlayer ONLY)"""
            
            new_delta        = self.axon - target
            self.del_delta   = new_delta - self.delta
            self.delta       = new_delta
            self.instability = self.delta**2 + self.del_delta**2
            
            self.learning_rate  = 1

            self.bias -= self.delta
            
            return self.delta


        if   self.t_function == "Sigmoid":
            return Sig(self, target)
        elif self.t_function == "Threshold":
            return Thresh(self, target)

    
    def learn(self, target = False):
        """
        Trains the neuron

        If a target value is provided, the neuron computes its error based
        on that target. this is only applicable to neurons on the output layer.

        target inputs should not be passed to hidden neurons. hidden
        neurons can calculate their own errors based on the error values
        of their parent neurons through back propogation.
        """

        self.r_age = self.r_age + 1

        # back propogate 
        self.delta = self.back_transfer(target)
        
        # adjust the weights
        for i,weight in enumerate(self.weights):
            self.weights[i] += self.dendrite[i] * (-self.learning_rate * self.delta)

        return self.delta

    def calc_influence(self):
        """Calculates influence of neuron, not really used right now"""
        
        if self.hasparents:
            self.influence = 0
            for parent in self.parents:
                parent_weights    = [abs(i) for i in parent.weights]
                parent_weight     = parent.weights[parent.children.index(self)]
                parent_impression = abs(parent_weight) / sum(parent_weights)
                self.influence    = float(self.influence) + parent_impression
            return
        else:
            return True


    def add_child(self,child):
        """connects a child to this neuron"""
        
        self.children.append(child)
        self.weights.append(random())
        self.haschildren = True
        return


    def remove_child(self,child):
        """disconnects a child neuron"""
        
        if child in self.children:
            del self.weights[self.children.index(child)]
            del self.dendrite[self.children.index(child)]
            self.children.remove(child)
        if len(self.children) == 0:
            self.haschildren = False
        return


    def add_parent(self,parent):
        """connects a parent to this neuron"""
        
        self.parents.append(parent)
        self.hasparents = True
        return
    

    def remove_parent(self,parent):
        """disconnects a parent neuron"""
        
        if parent in self.parents:
            self.parents.remove(parent)
        if len(self.parents) == 0:
            self.hasparents = False
        return

    
    def interogate(self):
        """ used to quickly print all info on the neuron"""
        
        print("========== neuron = {0} =========".format(self.name))
        print("Attribute     Value")
        for var in vars(self):
            if var != "log":
                whitespace = " "*(14-len(var))
                print("{0}{1}{2}".format(var, whitespace, getattr(self,var)))
        return
