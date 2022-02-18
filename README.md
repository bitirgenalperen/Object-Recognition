## For Part-1, values I tried and the best values set as global variables on the top.
--------------------------------------------------------------------------------
## Best values for k-means and k-nn are set as global variables on the top.
--------------------------------------------------------------------------------
# There is a function called "custom_process_selector" that allows the user to choose the operations from command-line, below are the samples for this purpose:
## train selector_type cluster_count
trains and creates cluster and data files locally
## validation selector_type cluster_count
runs on the validation data and creates the validation data instances locally
## evaluate eval_type sift_type cluster_count neigboor_count
evaluates the accuracy with given preferences
## run eval_dir, sift_type, cluster_count, neigboor_count
runs train, validation, and evaluation in a compact form
## run-best
starts running the best parameters, and prints the accuracy (this will also create the confussion matrix)
## run-test
starts running the best parameters, classifies the test data and write the results to file
## part1-sift
starts running for different sift parameters
## part1-dense
starts running for different dense-sift parameters
## part2
starts running for different k values of k-means
## part3
starts running for different k values of k-nn
## clear
cleares all the .pkl files that are locally stored