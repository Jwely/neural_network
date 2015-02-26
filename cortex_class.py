from random import random, randint, sample, randrange
import numpy
from neuron_class import neuron

__author__ = "Jeffry Ely, jeff.ely.08@gmal.com"

class cortex:

    """
    A cortex is a collection of neurons, a neural network in and of itself

    A cortex behaves a lot like a single neuron. One key difference is that
    a cortex is never "loaded" with a load function. Instead, the cortex
    loads each of its respective neurons and fires them automatically when
    the "fire" function is executed. It achieves this with the "is_ready"
    function.

    A cortex object also has an "import_state" and "export_state" function to
    write entire cortex state to a text file, or load from one such text file.

    Creates a cortex level output at "self.axons" which is a list of axon values
    on all the output neurons.
    """
    
    def __init__(self,name):
        """Creates a cortex with just a name"""

        # basic attributes
        self.name               = name
        self.influence_table    = []        # table of hidden neuron "influence"
        self.destroyed_neurons  = []        # to save neurons that are "killed"
        self.bias               = 0         # cortex level bias (unused)
        self.accuracy           = 0         # accuracy value of the cortex
        self.col_accuracy       = []        # accuracy by column (output neuron)
        self.row_accuracy       = []        # accuracy by row (input set)
        self.max_err            = 0.001     # max error an ouput can have to
                                            #   be considered ok for training
        self.min_acc            = 1.00      # min portion of all training
                                            #   scenarios required to pass
                                            
        # instability attributes
        self.instability        = 0         # see get_instability()
        self.crit_instability   = 1         # unrecoverable instability value

        # neuron lists
        self.neurons            = []
        self.hidden_neurons     = []
        self.output_neurons     = []
        return
    
        
    def queery(self, cortex_input_sets, silent = True):
        """
        Method for queery of a trained cortex.

        a cortex_input_sets should be a list of lists, where each
        entry in the list is a set of inputs.
        """

        cortex_output = []

        # handles multiple input sets. Lists of lists.
        if isinstance(cortex_input_sets[0],list):
            for cortex_input in cortex_input_sets:
                cortex_output.append(self.fire(cortex_input))

        # handles single input arrays. Lists
        else:
            cortex_output.append(self.fire(cortex_input_sets))

        if not silent:
            for line in cortex_output:
                print(line)
            
        return cortex_output

            
    def t_function(self, transfer_function, neurons):
        """
        Defines the transfer function of any number of neurons in cortex

        typically this is used to set the output neurons to have a threshold
        transfer function while the input neurons retain a sigmoid type

        Supported transfer_function inputs are determined at the neuron level,
        but are at least "Sigmoid" and "Threshold"
        """
        
        for neur in neurons:
            neur.t_function = transfer_function
        return

    
    def populate(self, hidden_neurons):
        
        """
        Fills the cortex with neurons and establishes connections
        
        Cortexes have a layer of hidden neurons, and a layer of output
        neurons to start with, but everything is set up such that
        more complex interconnections may evolve with future work.
        """
                            
        HiddenLayer = []
        OutLayer    = []

        for i in range(hidden_neurons):
            HiddenLayer.append("n_{0}".format(i))

        for i in range(self.output_width):
            OutLayer.append("n_{0}".format(i + hidden_neurons))

        self.hidden_neurons = [ neuron(name) for name in HiddenLayer]
        self.output_neurons = [ neuron(name) for name in OutLayer]

        for neur in self.output_neurons:
            for child in self.hidden_neurons:
                neur.add_child(child)
                child.add_parent(neur)

        self.neurons     = self.hidden_neurons + self.output_neurons
        self.size        = len(self.neurons)
        self.size_hidden = len(self.hidden_neurons)
        self.size_output = len(self.output_neurons)
        return


    def repopulate(self, layer = "All"):
        """used to reset all neurons in the cortex for a fresh start"""

        if layer == "All":
            reset_list = self.neurons
        elif layer == "Output":
            reset_list = self.output_neurons
        elif layer == "Hidden":
            reset_list = self.hidden_neurons

        for neur in reset_list:
            neur.reset()

        return

        
    def ready_to_fire(self,in_neuron,cortex_inputs):
        """
        Checks firing rediness of neuron within the cortex

        If the forward age of all children is one greater than the age
        of the input neuron, or the input neuron has zero children
        then this function loads the neuron and returns True.
        otherwise it returns False.
        """

        # for bottom level neurons
        if not in_neuron.haschildren:
            in_neuron.load(cortex_inputs)
            return True

        # for all neurons with children
        else:
            for child in in_neuron.children:
                if not child.f_age == (in_neuron.f_age +1):
                    return False
            else:
                in_neuron.load()
                return True


    def ready_to_learn(self,in_neuron):

        """
        checks learning rediness of neuron within cortex

        If the reverse age of all parents is one greater than
        the reverse age of the input neuron, or the input neuron
        has zero parents then the function returns True.
        """

        if in_neuron.hasparents == False:
            return True

        else:
            for parent in in_neuron.parents:
                if parent.r_age <(in_neuron.r_age +1):
                    return False
            else:
                return True

    
    def fire(self,cortex_inputs):
        """
        Fires every neuron in the cortex once for a given set of inputs

        returns a list of output values from output neuron axons
        returns an object list of output-layer neurons
        """
        
        unfired_neurons  = self.neurons[:]
        
        while len(unfired_neurons) > 0:

            for neur in unfired_neurons:
                if self.ready_to_fire(neur,cortex_inputs):
                    neur.fire()
                    unfired_neurons.remove(neur)

        self.axons = []
        for neur in self.neurons:
            if not neur.hasparents:
                self.axons.append(neur.axon)
                if not neur in self.output_neurons:
                    self.output_neurons.append(neur)
                    
        return self.axons


    def learn(self, targets):
        """ Allow the cortex to learn. This is done similarly to firing"""
        
        if not len(targets) == len(self.output_neurons):
            raise Exception("num of output neurons must equal num of output targets")

        unlearned_neurons = self.neurons[:]

        while len(unlearned_neurons) >0:

            for neur in self.output_neurons:
                if neur in unlearned_neurons:
                    target = targets[self.output_neurons.index(neur)]
                    neur.learn(target)
                    unlearned_neurons.remove(neur)

            for neur in unlearned_neurons:
                if self.ready_to_learn(neur):
                    neur.learn()
                    unlearned_neurons.remove(neur)
        return
    

    def train(self, max_sessions = 5000):
        """
        Repeatedly calls self.learn to train the cortex. Tracks instability

        Before training can occur a cortex must have run methods
            self.import_training
            self.populate
        If the output data is categorical in nature, the transfer function
        on output neurons should be set to "Threshold" with:
            self.t_function

        use "max_sessions" to limit the time the cortex can spend training

        view the results of training by calling
            self.print_accuracy_report
        """

        if self.size_hidden < len(self.training_input_set[0]):
            raise Exception("Too few hidden neurons! must be >= num of inputs")
        
        current_session = 1
        initial_session = 1

        while current_session < max_sessions and self.accuracy < self.min_acc:
             
            for i, training_input in enumerate(self.training_input_set):
                self.fire(training_input)
                self.learn(self.target_set[i])

            # give cortex time to stabilize, then monitor it regularly
            if current_session % 100 == 0 and initial_session >= 500:
                print("ses: {0} ...".format(current_session))
                self.get_instability()

                # if instability is very low, cortex has converged for better or worse
                if self.instability < 1e-3:
                    if self.get_accuracy() < self.min_acc:
                        print("ses: {0}, Repopulating hidden neurons".format(current_session))
                        self.repopulate("Hidden")
                        initial_session = 1
                        
                elif self.instability >= self.crit_instability:
                    print("ses: {0}, Repopulating all neurons".format(current_session))
                    self.repopulate()
                    initial_session = 1

            current_session += 1
            initial_session += 1
        return

    def get_instability(self):
        """
        Check cortex for convergence by looking at neuron instability. FAST

        Instability is measured by examining the delta values and change in
        delta values for all cortex neurons. Neurons which are learning slowly
        will have very low deltas and even lower del_deltas, but neurons which
        have become unstable will have a high delta, and likely a high del_delta.

        cortex instability is equal to that of the least stable neuron.
        """

        instability = []

        for neur in self.neurons:
            instability.append(neur.instability)

        self.instability = max(instability)
        
        return self.instability    


    def get_accuracy(self):
        """
        Exhaustively tests cortex state with all training data. SLOW

        Outputs a percent of all output values that correctly match
        up to target values within the 'acceptable_error' (acc_err) bounds.

        set a report path to save an accuracy report for the cortex
        """
        
        total_outs          = self.queery(self.training_input_set)
        accuracy_matrix     = total_outs
        self.row_accuracy   = []
        self.col_accuracy   = []

        # build accuracy matrix
        for i,out_set in enumerate(total_outs):
            for j,out in enumerate(out_set):
                if out == self.target_set[i][j]:
                    accuracy_matrix[i][j] = 1
                else:
                    accuracy_matrix[i][j] = 0

        # get row-wise accuracy         
        for i,row in enumerate(accuracy_matrix):
            self.row_accuracy.append(float(sum(row))/len(row))

        # transpose the matrix to get columnwise accuracy
        accuracy_matrix = zip(*accuracy_matrix)
        for i,col in enumerate(accuracy_matrix):
            self.col_accuracy.append(float(sum(col))/len(col))

        # get total accuracy and cortex learning age
        self.accuracy = sum(self.col_accuracy)/len(self.col_accuracy)
        self.learn_age= self.neurons[0].r_age
        
        return self.accuracy
    
    
    def print_accuracy_report(self):
        """ Prints a summary of the present cortex accuracy"""
        
        # print a heads up report to screen if the dataset is small
        print("Cortex age is {0}".format(self.learn_age))
        print("Total accuracy is {0}%".format(100 * self.accuracy))
        print("Output  |  Target")
        for i,training_input in enumerate(self.training_input_set):
            print("{0}  |  {1}".format(
                self.fire(training_input),
                self.target_set[i]))

        print("Accuracy by output value (column)")
        print self.col_accuracy
            
        return

    def save_accuracy_report(self, filepath):
        """saves an accuracy report to filepath"""

        self.accuracy_report_path = filepath
        
        return
        
    def disconnect(self, neuron_1, neuron_2):
        """destroys connections between two neurons, inputs name or instance"""

        neuron_1 = self.find_object(neuron_1)
        neuron_2 = self.find_object(neuron_2)
        
        neuron_1.remove_parent(neuron_2)
        neuron_1.remove_child(neuron_2)
            
        neuron_2.remove_parent(neuron_1)
        neuron_2.remove_child(neuron_1)
        return


    def destroy(self, neuron_1):
        """destroys a neuron, inputs name or instance"""
        
        neuron_1 = self.find_object(neuron_1)
        
        if neuron_1 in self.output_neurons:   
            return # WILL NOT destroy outut neurons                  
        
        for neur in self.neurons:
            self.disconnect(neuron_1, neur)   

        # update cortex information
        self.neurons.remove(neuron_1)
        self.hidden_neurons.remove(neuron_1)
        self.destroyed_neurons.append(neuron_1)
        
        self.size = len(self.neurons)
        self.size_hidden = self.size - self.size_output
        
        print("DESTROYED NEURON WITH NAME '{0}'".format(neuron_1.name))
        return
    
    
    def find_object(self, name):
        """simple sub-function to return a neuron instance for input neuron name"""

        if str(type(name)) == "<type 'str'>":
            for neur in self.neurons:
                if neur.name == name:
                    return neur
            else:
                raise AttributeError("no neuron with name '{0}'".format(name))
            
        else:
            return name


    def map_influence(self):
        """polls all neurons for influence values"""
        
        self.influence_table = []

        # poll neurons for influence
        for neur in self.hidden_neurons:
            neur.calc_influence()
            self.influence_table.append(neur.influence)

        # normalize influence values
        influence_sum = sum(self.influence_table)
        for i,inf_val in enumerate(self.influence_table):
            self.influence_table[i] = self.influence_table[i] / influence_sum

        return

            
    def con_matrix(self):
        """
        Creates a visual connection matrix that is self.size^2 in dims

        +0  indicates no connection
        +1  indicates the row neuron is a parent of the column neuron
        -1  indicates the row neuron is a child of the column neuron
        """

        print('Connection matrix for "{0}" with {1} neurons'.format(self.name,self.size))
        matrix = numpy.zeros((self.size,self.size))

        for x,row in enumerate(self.neurons):
            for y,col in enumerate(self.neurons):
                if col.hasparents:
                    if row in col.parents:
                        matrix[x,y] = 1
                if row.hasparents:
                    if col in row.parents:
                        matrix[x,y] = -1
                    
        print matrix
        return matrix


    def interogate(self):
        """quick function to interogate all neurons for heads up display"""
        
        print("="*80)
        print(" "*20 + "Interogation of cortex '{0}'".format(self.name))
        print("="*80)

        for var in vars(self):
            whitespace = " "*(14-len(var))
            print("{0}{1}{2}".format(var, whitespace, getattr(self,var)))
        
        for neur in self.neurons:
            neur.interogate()
        return
    
    
    def export_state(self,filepath):
        """Exports the entire state of this cortex to a text file"""
        
        header = ("name ; haschildren ; children ; hasparents ; parents"+
                  "; weights ; bias ; f_age ; r_age ; size ; dec ; t_function"+ 
                  "; delta ; del_delta" )

        f = open(filepath,'w+')
        f.write(header + '\n')

        for neuron in self.neurons:
            entry = "{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11};{12};{13}".format(
                neuron.name,
                neuron.haschildren,
                [getattr(child,'name') for child in neuron.children],
                neuron.hasparents,
                [getattr(parent,'name') for parent in neuron.parents],
                neuron.weights,
                neuron.bias,
                neuron.f_age,
                neuron.r_age,
                neuron.size,
                neuron.dec,
                neuron.t_function,
                neuron.delta,
                neuron.del_delta)

            entry = entry.replace("'","").replace("[","").replace("]","")
            f.write(entry + '\n')

        cortex_entry_headers = "name ; accuracy \n"
        cortex_entry = "{0};{1}\n".format(
            self.name,
            self.accuracy)
        
        f.write(cortex_entry_headers)
        f.write(cortex_entry)
        
        f.close()
        return


    def import_state(self,filepath):

        """
        Imports an already trained cortex thats been saved before

        this function demonstrates my lack of understanding of
        python regular expressions.....i think

        Check out the "export_state" funtction for insight on how this
        should be formatted.
        """

        self.imported_from = filepath
        self.neurons = []
        
        with open(filepath,'r+') as f:
            header = next(f)
            for line in f:
                line = line.replace('\n','').replace(' ','').replace("[","").replace("]","")
                info = line.split(';')

                # handles neuron info one row at a time
                if len(info) >10:
                    for i,field in enumerate(info):
                        if ',' in field:
                            info[i] = info[i].split(',')
                        if 'True' in field:
                            info[i] = True
                        elif 'False' in field:
                            info[i] = False

                    neur = neuron(info[0])

                    attributes = ['name','haschildren','children','hasparents',
                                  'parents','weights','bias','f_age','r_age',
                                  'size','dec','t_function','delta','del_delta']

                    for i,attribute in enumerate(attributes):
                        setattr(neur,attribute,info[i])

                    neur.weights    = map(float,neur.weights)
                    neur.bias       = float(neur.bias)
                    neur.f_age      = int(neur.f_age)
                    neur.r_age      = int(neur.r_age)
                    neur.size       = int(neur.size)
                    neur.dec        = int(neur.dec)
                    neur.delta      = float(neur.delta)
                    neur.del_delta  = float(neur.del_delta)

                    self.neurons.append(neur)
                    
                # handles cortex data at the bottom.
                else:
                    cortex_header = info
                    
                    cortex_line = next(f).replace('\n','').replace(' ','').replace("[","").replace("]","")
                    cortex_info = cortex_line.split(';')

                    for i,attribute in enumerate(cortex_header):
                        setattr(self, attribute, cortex_info[i])

                    self.accuracy = float(self.accuracy)

                self.size = len(self.neurons)
                
            for neur in self.neurons: 
                # presently, each neurons children and parents lists
                # are just strings. turn these into object lists.

                if neur.hasparents:
                    if not isinstance(neur.parents,list):#handles single parent lists
                        neur.parents = [neur.parents]
                        
                    for i,parent in enumerate(neur.parents):
                        neur.parents[i] = self.find_object(parent)
                        
                    self.hidden_neurons.append(neur)
                else:
                    neur.parents = []

                if neur.haschildren:
                    if not isinstance(neur.children,list):# handles single child lists
                        neur.children = [neur.children]
                    for i,child in enumerate(neur.children):
                        neur.children[i] = self.find_object(child)
                        
                    if not neur.hasparents:
                        self.output_neurons.append(neur)
                else:
                    neur.children = []

        f.close()
        return

    def import_training(self, training_data_filepath, normalize = True):

        """
        Read training data as input/ouput format from text file

        input file should have the format
        in0,in1,...,intN;out0,out1,...,outN
        
        Two numpy arrays are returned:
            inarray     (2d matrix of training input datasets)
            outarray    (1d array of training output)
        These combine to represent the entire training dataset

        normalization of inputs is performed by default
        """
        
        self.training_input_set = []
        self.target_set         = []
        
        with open(training_data_filepath) as f:
            header = next(f)
            names          = header.split(";")
            self.in_names  = names[0].split(',')
            self.out_names = names[1].split(',')
            
            for line in f:
                indata,outdata = line.split(';')
                outdata = map(float, outdata.replace('\n','').split(','))
                indata  = map(float, indata.split(','))
                
                self.training_input_set.append(indata)
                self.target_set.append(outdata)

        f.close()

        self.output_width = len(self.target_set[0])

        if normalize:
            self.training_input_set = self.normalize_training()
            
        return


    def normalize_training(self):
        """normalizes input to output data to speed convergence"""

        inarray_norm = []
        for i,training_set in enumerate(self.training_input_set):
            if max(training_set) != 0:
                training_set = [(float(j)/max(training_set)) for j in training_set]
            inarray_norm.append(training_set)

        return(inarray_norm)