from cortex_class import cortex

# import test data
path = r"training_sets\XNOR_XOR_AND_OR_NOR_NAND.txt"

# set up a cortex
cort = cortex('cortex_1')

cort.import_training(path)
cort.populate(4)
cort.min_acc = 1.0
cort.t_function("Threshold", cort.output_neurons)

# make sure even after export and re-import, the cortex can still train
cort.export_state('initial_state.txt')
cort.import_state('initial_state.txt')

#train the cortex
cort.train(10000)
cort.print_training_accuracy_report()
cort.export_state('final_state.txt')






