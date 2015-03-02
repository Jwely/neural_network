from random import random, randint, sample, randrange
import numpy
from neuron_class import neuron
import sys

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

        Supported transfer_function inputs are determined at the neuron level,
        but are at least "Sigmoid", "Threshold" and "TanH"
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

        for i in range(self.size_output):
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
        """ Used to reset entire layers in the cortex for a fresh start"""

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
        Checks learning rediness of neuron within cortex

        If the reverse age of all parents is one greater than the reverse age
        of the input neuron, or the input neuron has zero parents then the
        function returns True.
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

        Returns a list of values from output layer neuron axons.
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
    

    def train(self, epoch_incriment, epoch_stabilize, epoch_max):
        """
        Repeatedly calls self.learn to train the cortex. Tracks instability

        Before training can occur a cortex must have run methods
            self.import_training
            self.populate
        If the output data is categorical in nature, the transfer function
        on output neurons should be set to "Threshold" with:
            self.t_function

        Inputs:
            epoch_incriment   number of epochs between accuracy checks
            epoch_stabilize   number of epochs before first accuracy check
            sex_max           maximum number of epochs to run

        view the results of training by calling
            self.print_accuracy_report
        """

        print("\nInitializing training of cortex '{0}'\n".format(self.name))
        current_epoch = 1
        initial_epoch = 1

        session_length = len(self.training_input_set)
        
        while current_epoch < epoch_max and self.accuracy < self.min_acc:
            
            if session_length >= 500:
                print("training")
                
            for i, training_input in enumerate(self.training_input_set):
                self.fire(training_input)
                self.learn(self.target_set[i])
                if i% 500 == 0 and i != 0:
                    print("epoch: \t {0} \t row: {1}".format(current_epoch, i))
                    

            # give cortex time to stabilize, then monitor it regularly
            if current_epoch % epoch_incriment == 0 and initial_epoch >= epoch_stabilize:
                print("epoch: {0} ...".format(current_epoch))
                self.get_instability()

                # if instability is very low, cortex has converged for better or worse
                if self.instability < 1e-3:
                    if self.get_training_accuracy() < self.min_acc:
                        print("epoch: {0}, Repopulating hidden neurons".format(current_epoch))
                        self.repopulate("Hidden")
                        initial_epoch = 1
                        
                elif self.instability >= self.crit_instability:
                    print("epoch: {0}, Repopulating all neurons".format(current_epoch))
                    self.repopulate()
                    initial_epoch = 1

            current_epoch += 1
            initial_epoch += 1
        return


    def train_multithread(self, epoch_incriment, epoch_stabilize, epoch_max , num_threads):
        """
        There is good reason to attept multithreading of cortex training. One
        approach would be to segment the training set into 2+ equal parts and train 2+
        duplicate cortexes for some small number of sessions, then average the weights
        together, then resume separate training, and so on.

        could use?
        import threading
        import multiprocessing
        """
        pass
        

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


    def get_training_accuracy(self):
        """
        Exhaustively tests cortex state with all training data. SLOW

        Outputs a fraction of all output values that correctly match
        up to target values within the 'acceptable_error' (acc_err) bounds.
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
    
    
    def print_training_accuracy_report(self):
        """ Prints a summary of the present cortex accuracy"""

        print("="*50)
        print("Accuracy report for cortex with name '{0}'".format(self.name))
        print("="*50)

        print("\nOutput  |  Target")
        for i,training_input in enumerate(self.training_input_set):
            print("{0}  |  {1}".format(
                self.fire(training_input),
                self.target_set[i]))

        print("\nCortex age is {0}".format(self.learn_age))
                
        print("\nAccuracy by output value (column)")
        for i,out_name in enumerate(self.out_names):
            print("{0} \t  {1}%".format(out_name,100*self.col_accuracy[i]))

        print("-"*50)
        print("Total \t  {0}%".format(100 * self.accuracy))
        print("="*50 + "\n")
        return


    def save_training_accuracy_report(self, filepath):
        """saves an accuracy report to filepath"""

        self.training_accuracy_report_path = filepath
        
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
        
        print("="*50)
        print("Interogation of cortex with name'{0}'".format(self.name))
        print("="*50)

        for var in vars(self):
            whitespace = " "*(25-len(var))
            print("{0}{1}{2}".format(var, whitespace, getattr(self,var)))
        
        for neur in self.neurons:
            neur.interogate()
        return
    
    
    def export_state(self,filepath):
        """Exports the state of this cortex to a text file so it may be saved"""

        with open(filepath, 'w+') as f:
            
            # writes cortex level attributes
            cortex_entry_headers = ("name ; out_names ; size ; size_hidden ;" + 
                                    "size_output; accuracy ; min_acc ; col_accuracy ;" +
                                    "max_err ; crit_instability")

            cortex_entry = "{0};{1};{2};{3};{4};{5};{6};{7};{8};{9}".format(
                self.name,
                self.out_names,
                self.size,
                self.size_hidden,
                self.size_output,
                self.accuracy,
                self.min_acc,
                self.col_accuracy,
                self.max_err,
                self.crit_instability)
            
            f.write(cortex_entry_headers + '\n')
            f.write(cortex_entry + '\n')

            # write rows with neuron level attributes
            header = ("name ; haschildren ; children ; hasparents ; parents"+
                      "; weights ; bias ; f_age ; r_age ; size ; dec ; t_function"+ 
                      "; delta ; del_delta" )

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

        print("="*50)
        print("Importing cortex state from '{0}'".format(filepath))
        print("="*50 + '\n')
        
        self.imported_from = filepath
        self.neurons = []
        self.output_neurons = []
        self.hidden_neurons = []

        with open(filepath,'r+') as f:

            # grabs cortex level information from the top first
            cortex_header = next(f).replace('\n','').replace(' ','').split(';')
            
            cortex_line = next(f).replace('\n','').replace(' ','')
            cortex_line = cortex_line.replace("[","").replace("]","")
            cortex_info = cortex_line.split(';')

            for i,field in enumerate(cortex_info):
                if ',' in field:
                    cortex_info[i] = cortex_info[i].replace("'",'').split(',')   

            for i,attribute in enumerate(cortex_header):
                setattr(self, attribute, cortex_info[i])
                print("imported attribute '{0}' as {1}".format(attribute,cortex_info[i]))

            # correct formats that shouldn't be strings
            self.size               = int(self.size)
            self.size_hidden        = int(self.size_hidden)
            self.size_output        = int(self.size_output)
            self.accuracy           = float(self.accuracy)
            self.min_acc            = float(self.min_acc)
            self.col_accuracy       = map(float,self.col_accuracy)
            self.max_err            = float(self.max_err)
            self.crit_instability   = float(self.crit_instability)
            

            # moves on to grab neuron information
            neuron_header = next(f).replace('\n','').replace(' ','').split(';')
            
            for line in f:
                line = line.replace('\n','').replace(' ','').replace("[","").replace("]","")
                info = line.split(';')

                # handles neuron info one row at a time
                for i,field in enumerate(info):
                    if ',' in field:
                        info[i] = info[i].split(',')
                    if 'True' in field:
                        info[i] = True
                    elif 'False' in field:
                        info[i] = False

                neur = neuron(info[0])

                for i,attribute in enumerate(neuron_header):
                    setattr(neur,attribute,info[i])

                neur.weights    = map(float,neur.weights)
                neur.bias       = float(neur.bias)
                neur.f_age      = int(neur.f_age)
                neur.r_age      = int(neur.r_age)
                neur.size       = int(neur.size)
                neur.dec        = int(neur.dec)
                neur.delta      = float(neur.delta)
                neur.del_delta  = float(neur.del_delta)
                neur.hasparents = bool(neur.hasparents)
                neur.haschildren= bool(neur.haschildren)

                self.neurons.append(neur)
                print("imported member neuron with name '{0}'".format(neur.name))

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
        print("="*50 + '\n')
        
        return

    def import_training(self, training_data_filepath):

        """
        Read training data as input/ouput format from text file

        input file should have the format
        in0,in1,...,intN;out0,out1,...,outN

        normalization of inputs is performed by default
        """
        
        self.training_input_set = []
        self.target_set         = []
        
        with open(training_data_filepath) as f:
            header = next(f)
            names          = header.split(";")
            self.in_names  = names[0].split(',')
            self.out_names = names[1].replace('\n','').split(',')
            
            for line in f:
                indata,outdata = line.split(';')
                outdata = map(float, outdata.replace('\n','').split(','))
                indata  = map(float, indata.split(','))
                
                self.training_input_set.append(indata)
                self.target_set.append(outdata)

        f.close()

        self.size_output = len(self.target_set[0])
        return
    

    def import_training2(self, input_filepath, target_filepath):
        """
        Read training data as separate input and target csv files as
        these are far more readily available than combined csv formats
        used in
        
        @method import_training
        """

        self.training_input_set = []
        self.target_set         = []

        with open(input_filepath) as f:
            header = next(f)
            self.in_names  = header.split(',')
            
            for line in f:
                indata = map(float, line.replace('\n','').split(','))
                self.training_input_set.append(indata)

        f.close()
        
        with open(target_filepath) as f:
            header = next(f)
            self.out_names = header.split(',')
            
            for line in f:
                outdata = map(float, line.replace('\n','').split(','))
                self.target_set.append(outdata)

        f.close()
        
        self.size_output = len(self.target_set[0])
        return
    

    def normalize_training(self, low_bound, high_bound):
        """
        normalizes input data to speed convergence

        When using normalized training sets, it is important to ensure
        that the training set represents the most extreme likely scenarios.
        A trained cortex may not be able to handle inputs outside the range
        of training inputs.
        """

        def normalize(self, dataset, low_bound, high_bound):
            # transpose the training set to get individual column min/maxs
            col_maxs = []
            col_mins = []
            
            column_dataset = zip(*dataset)
            for column in column_dataset:
                col_maxs.append(max(column))
                col_mins.append(min(column))
                
            del column_dataset

            # set up empty new training array normalize the input data.
            dataset_norm = []
            
            for i, data_row in enumerate(dataset):

                row_norm = []
                for j, data in enumerate(data_row):
                    # z = (high_bound - low_bound)*(x - min)/(max - min) + low_bound

                    # catch errors that arise when all inputs in column are identical
                    if (col_maxs[j] - col_mins[j]) ==0:
                        z = 0
                    else:
                        b_range = (high_bound - low_bound)
                        z       = (data - col_mins[j]) / (col_maxs[j] - col_mins[j])
                        z       = (b_range) * z + low_bound
                        
                    row_norm.append(z)
                    
                dataset_norm.append(row_norm)

            return(dataset_norm)

        self.training_input_set = normalize(self, self.training_input_set,
                                            low_bound, high_bound)
        self.target_set         = normalize(self, self.target_set,
                                            low_bound, high_bound)
        return


    def reduce_training(self, num_rows):
        """ quick dirty function to reduce a training set to num_rows entries"""

        self.training_input_set[num_rows:]   = []
        self.target_set[num_rows:]          = []
        return
        






























