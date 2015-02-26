from cortex_class import cortex

cortex_path = "cortexes\All_logic_gates_cortex9.txt"
cort = cortex("cort1")

cort.import_state(cortex_path)
cort.interogate()


