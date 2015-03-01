from cortex_class import cortex

# import test data
input_path = r"training_sets\TJ_data_input.csv"
target_path = r"training_sets\TJ_data_target.csv"

# set up a cortex
cort = cortex('cortex_1')

cort.import_training2(input_path, target_path)


##cort.populate()
##cort.min_acc = 1.0
##cort.t_function("Threshold", cort.output_neurons)
##
### make sure even after export and re-import, the cortex can still train
##cort.export_state('initial_state.txt')
##cort.import_state('initial_state.txt')
##
###train the cortex
##cort.train(10000)
##cort.print_training_accuracy_report()
##cort.export_state('final_state.txt')






