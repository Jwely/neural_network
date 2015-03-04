from cortex_class import cortex

# import test data
path = r"training_sets\AND.txt"

# set up a cortex
cort = cortex('cortex_1')

cort.import_training(path)
cort.populate(0)    # 0 hidden layer neurons
cort.min_acc = 1.0
cort.t_function("Threshold", cort.output_neurons)

#train the cortex
cort.train(100, 50, 10000)
cort.print_training_accuracy_report()
cort.interogate()






