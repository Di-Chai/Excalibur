#include "base.hpp"
#include "utils.hpp"
#include "bbtsvd.hpp"
#include <map>
#include <pthread.h>
#include <semaphore.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <queue>
#include <random>
#include <assert.h>

struct ClientMeta{
    size_t m;
    size_t local_n;
    int cid;
    unsigned int shared_seed;
};

struct ReceiverMeta{
    int port;
    int num_client;
};

void* handle_one_connection(void* param);
void* create_server(void* params);

class DecentralizedClient{
    public:

    // SVD concig
    int svd_mode, top_k;
    double *w_lr, *y;

    // Meta info
    int client_id;
    size_t m, local_n, global_n=0, reduced_global_n=0;
    map<int, size_t> n_pos, reduced_n_pos;
    map<int, size_t> local_ns, reduced_local_ns;

    // Local data
    double *raw_x, *masked_x;
    double *shrinked_x; // util variable
    double *sigma, *local_v, *x_diagonal, *x_upper_diagonal;

    double *qr_u, *FinalU, *FinalVT;
    double *mask_shared_v;
    
    // Timer
    struct timeval start, finish, t1, t2, comm_start, comm_finish;
    struct timeval run_begin, run_finish;

    // Communicator
    string listen;
    int my_next_client, my_front_client;
    pthread_t server_tid;
    int num_client;
    map<int, int> client_socket;
    map<int, struct sockaddr_in*> server_addr;
    int base_server_port, my_server_port;
    
    // Mask
    unsigned int public_seed, private_seed;
    int global_mask_size, local_mask_size;

    // DecentralizedClient(int cid);
    DecentralizedClient(int cid, int num_client, int mode, size_t topk, size_t drow, size_t dcol, int port, string listen, string dpath, bool _is_memmap, bool _evaluate, string _log_dir, unsigned int seed, int opt);
    ~DecentralizedClient();
    
    // Load data
    bool is_memmap;
    string dpath;
    void load_data();
    void remove_input_data();
    
    // Evaluate
    int opt_control;
    bool evaluate;
    string log_dir;
    double time_init=0, time_local_qr=0, time_global_qr=0, time_apply_mask=0, time_exchange_low_dim=0, time_bi_diag=0, time_bsvd=0, time_combine_svd_remove_mask=0, time_remove_mask=0, time_combine_svd;
    double time_communication=0;
    
    // Communicators
    void init_server_and_client();
    void send_to(int cid, char* buffer, size_t size);
    template <typename T> void receive(T** buffer, int cid);
    void broadcast(char* buffer, size_t size);
    void ring_all_reduce_sum(double* buffer, int size);
    void all_gather(char** buffer, size_t size, int *c_group, int nc, int src);
    
    // Computations
    void local_qr_reduction(size_t m, size_t n, double **raw_x, double *q, bool transpose_x, bool compute_q);
    
    string filename_with_cid(string filename);
    
    // Run
    void run();
};
