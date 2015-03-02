from cortex_class import cortex

# set up a cortex
input_path = r"training_sets\TJ_data_input.csv"
target_path = r"training_sets\TJ_data_target.csv"
cort = cortex('cortex_1')
num_hidden  = 16
min_acc     = 0.50
max_err     = 0.01

cort.import_training2(input_path, target_path)
cort.normalize_training(-1,1)
cort.reduce_training(5)
cort.populate(num_hidden)
cort.min_acc = min_acc
cort.max_err = max_err
cort.crit_instability = 2
cort.t_function("TanH", cort.neurons)

#train the cortex
incriment = 100
stabilize = 600
maximum   = 10000

cort.train(incriment, stabilize, maximum)
cort.print_training_accuracy_report()
cort.export_state('final_state.txt')






