from cortex_class import cortex

# import test data
path = r"training_sets\XNOR_XOR_AND_OR_NOR_NAND.txt"

# set up a cortex
cort = cortex('cortex_1')

cort.import_training(path)
cort.populate(4)
cort.min_acc = 1.0
cort.t_function("Threshold", cort.output_neurons)
cort.train(10000)
cort.print_accuracy_report()




