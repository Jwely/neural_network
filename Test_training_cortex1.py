from cortex_class import cortex

# set up a cortex
input_path  = r"training_sets\XNOR_XOR_AND_OR_NOR_NAND.txt"
cort        = cortex('cortex_1')
num_hidden  = 4
min_acc     = 1.0

cort.import_training(input_path)
cort.normalize_training(0,1)
cort.populate(num_hidden)
cort.min_acc = min_acc
cort.t_function("Threshold", cort.output_neurons)

# make sure even after export and re-import, the cortex can still train
cort.export_state('initial_state.txt')
cort.import_state('initial_state.txt')

#train the cortex
incriment = 100
stabilize = 500
maximum   = 10000

cort.train(incriment, stabilize, maximum)
cort.print_training_accuracy_report()
cort.export_state('final_state.txt')








