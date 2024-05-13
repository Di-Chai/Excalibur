| Keywords    | Explanation                                                                                                                                                      |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| dataset     | Name of the experimental dataset                                                                                                                                |
| m           | Number of rows of the matrix                                                                                                                                     |
| n           | Number of columns of the matrix                                                                                                                                  |
| task        | Type of task; could be SVD or one of its applications                                                                                                            |
| top_k       | the number of top singular vectors, while '-1' means computing all the vectors                                                                                   |
| num_clients | Number of clients                                                                                                                                                |
| seed        | The random seed                                                                                                                                                  |
| is_memmap   | Whether offload the large matrices to SSD or keep everything in RAM                                                                                  |
| evaluete    | Whether evaluate the accuracy of the results (some large-scale experiments may take a long time to evaluate).                                                    |
| bandwidth   | Networking bandwidth                                                                                                                                             |
| latency     | Networking latency                                                                                                                                               |
| total_time  | The total time cost                                                                                                                                              |
| u_orth_test | The orthogonal test of left singular vectors U, i.e., ||UU^T-I||                                                                                                 |
| v_orth_test | The orthogonal test of right singular vectors V, i.e., ||VV^T-I||                                                                                                |
| svd_error   | The reconstruction error of SVD (only when `task=svd`)                                                                                                           |
| u_pd        | The projection distance between left singular vectors U and the ground truth (we have pre-computed the ground truth and put the results in the dataset folder.). |
| v_pd        | The projection distance between right singular vectors V and the ground truth                                                                                    |
| lr_error    | The linear regression error (only when `task=lr`)                                                                                                                |
| CPU         | The average CPU usage of each peer                                                                                                                          |
| Memory      | The average memory usage of each peer                                                                                                                       |
| netIn       | The average amount of received messages of each peer                                                                                                        |
| netOut      | The average amount of sent-out messages of each peer                                                                                                        |
| blockIn     | The average amount of disk read of each peer                                                                                                                |
| blockOut    | The average amount of disk write of each peer                                                                                                               |

[Go back to README](../README.md)