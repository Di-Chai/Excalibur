#include "client.hpp"

INITIALIZE_EASYLOGGINGPP

struct ClientConnection{
    int connection;
    char clientIP[INET_ADDRSTRLEN] = "";
    struct sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    int max_receive;
};

size_t max_comm_size = 1073741824; //134217728;
size_t receive_memmap_threshold = max_comm_size * 10;
sem_t sem_x;
queue<void*> *message_queue;
int num_connection = 0;

void* handle_one_connection(void* param){
    struct ClientConnection* CC = (struct ClientConnection*) param;
    int cid;
    recv(CC->connection, &cid, sizeof(cid), MSG_WAITALL);
    cout << "Connection From Client-" << cid << " with " << CC->clientIP << ":" << ntohs(CC->clientAddr.sin_port) << endl;

    sem_wait(&sem_x);
    num_connection += 1;
    sem_post(&sem_x);

    int file_counter = 0;

    size_t len, message_size = 0;
    while(true){
        len = recv(CC->connection, &message_size, sizeof(message_size), MSG_WAITALL);
        if(len <= 0) {
            cout << "Client-" << cid << " Disconnected " << CC->clientIP << ":" << ntohs(CC->clientAddr.sin_port) << endl;
            sem_wait(&sem_x);
            num_connection -= 1;
            sem_post(&sem_x);
            break;
        }
        // ToDo Change to smart allocate
        void* msg;
        string filename = get_log_path_from_env() + "/Client" + to_string(get_client_id_from_env()) + "_receive_from_" + to_string(cid) + "_" + to_string(file_counter++) + ".mat";
        get_space(&msg, message_size, message_size > receive_memmap_threshold, filename);
        if(message_size > max_comm_size){
            len = 0;
            for(size_t i=0; i<message_size; i+=max_comm_size){
                len += recv(CC->connection, (void*)((char*)msg+i), min(max_comm_size, message_size-i), MSG_WAITALL);
                cout << "Receive batch " << i << " " << len << endl;
            }
        }
        else len = recv(CC->connection, msg, message_size, MSG_WAITALL);
        if(len <= 0){
            cout << "Client-" << cid << " Disconnected " << CC->clientIP << ":" << ntohs(CC->clientAddr.sin_port);
            sem_wait(&sem_x);
            num_connection -= 1;
            sem_post(&sem_x);
            break;
        }
        if(len != message_size){
            cout << "Error with len=" << len << ", message_size=" << message_size << endl; 
            assert(len == message_size);
        }
        
        // Response code
        // std::cout << "Sending to " << CC->clientIP << ":" << ntohs(CC->clientAddr.sin_port) << std::endl;
        // sendto(CC->connection, buf, len, 0, (struct sockaddr*)&(CC->clientAddr), CC->clientAddrLen);
        
        // Push msg into Queue
        sem_wait(&sem_x);
        message_queue[cid].push(msg);
        sem_post(&sem_x);
    }
    cout << "Client Handler Exit" << endl;
    pthread_exit(NULL);
}

void* create_server(void* params){

    std::cout << "Create Receiver" << std::endl;
    
    struct ReceiverMeta* rm = (struct ReceiverMeta*) params;

    // socket
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenfd == -1) {
        std::cout << "Error: socket" << std::endl;
        return 0;
    }
    // bind
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(rm->port);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(listenfd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        std::cout << "Error: bind" << std::endl;
        return 0;
    }
    // listen
    if(listen(listenfd, 100) == -1) {
        std::cout << "Error: listen" << std::endl;
        return 0;
    }

    int num_listener = rm->num_client - 1;
    // Client threats
    pthread_t client_threats[num_listener];
    int thread_counter = 0;
    for(int i=0; i<num_listener; i++){
        struct ClientConnection* CC = new struct ClientConnection;
        CC->connection = accept(listenfd, (struct sockaddr*)&(CC->clientAddr), &CC->clientAddrLen);
        inet_ntop(AF_INET, &(CC->clientAddr.sin_addr), CC->clientIP, INET_ADDRSTRLEN);
        std::cout << "Connected " << CC->clientIP << ":" << ntohs(CC->clientAddr.sin_port) << std::endl;
        if (CC->connection < 0) {
            std::cout << "Error: accept" << std::endl;
            continue;
        }
        // Create thread to handle the connection
        if (pthread_create(&client_threats[thread_counter++], NULL, handle_one_connection, CC) != 0)
            cout << "Failed to create thread" << endl;
    }
    for(int i=0; i<num_listener; i++){
        int join_return = pthread_join(client_threats[i], NULL) != 0;
    }
    cout << "All Connections Exited" << endl;
    close(listenfd);
    pthread_exit(NULL);
}

DecentralizedClient::DecentralizedClient(int _cid, int _num_client, int _mode, size_t _topk, size_t _drow, size_t _dcol, int _port, string _listen, string _dpath, bool _is_memmap, bool _evaluate, string _log_dir, unsigned int seed, int opt){
    // Set Params
    client_id = _cid;
    num_client = _num_client;
    m = _drow;
    local_n = _dcol;
    base_server_port = _port;
    is_memmap = _is_memmap;
    evaluate = _evaluate;
    log_dir = _log_dir;
    opt_control = opt;

    // Put Env
    char client_id_env[50];
    snprintf(client_id_env, sizeof(client_id_env), "CLIENT_ID=%d", client_id);
    putenv(client_id_env);

    char log_path_env[100];
    sprintf(log_path_env, "LOG_PATH=%s", log_dir.c_str());
    putenv(log_path_env);
    
    // Config Easylogging++
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.parseFromText("*GLOBAL:\n FILENAME = " + log_dir + "/Client" + to_string(client_id) + ".log");  
    el::Loggers::reconfigureLogger("default", defaultConf);  
    
    // unhandled
    listen = _listen;
    dpath = _dpath;
    svd_mode = _mode;
    top_k = _topk;
    
    my_next_client = (client_id + 1) % num_client;
    my_front_client = (client_id - 1 + num_client) % num_client;
    my_server_port = base_server_port + client_id;
    
    // Log Params
    LOG(INFO) << "Parameters Start";
    LOG(INFO) << "My ClientID = " << client_id;
    LOG(INFO) << " Num Client = " << num_client;
    LOG(INFO) << "   SVD Mode = " << svd_mode;
    LOG(INFO) << "       TopK = " << top_k;
    LOG(INFO) << "          M = " << m;
    LOG(INFO) << "    Local N = " << local_n;
    LOG(INFO) << "  Base Port = " << base_server_port;
    LOG(INFO) << "    My Port = " << my_server_port;
    LOG(INFO) << "     Listen = " << listen;
    LOG(INFO) << "      dpath = " << dpath;
    LOG(INFO) << "  is_memmap = " << is_memmap;
    LOG(INFO) << "   evaluate = " << evaluate;
    LOG(INFO) << "    log_dir = " << log_dir;
    LOG(INFO) << "       seed = " << seed;
    LOG(INFO) << "opt_control = " << opt_control;
    LOG(INFO) << "Parameters End";
    
    /*
    svd_mode = 0, Full SVD
    svd_mode = 1, Two Side LSA
    svd_mode = 2, Left Side PCA
    svd_mode = 3, Right Side PCA
    svd_mode = 4, Linear Regression
    */
    
    // Set random seed
    // random_device seed; seed();
    srand(seed);

    // Set data
    load_data();

    // Init sender and recvier
    sem_init(&sem_x, 0, 1);
    message_queue = new queue<void*>[num_client];
    init_server_and_client();
    
    // Starting to run
    run();
}

void DecentralizedClient::load_data(){
    load_memmap_file(&raw_x, dpath + "/Client" + to_string(client_id) + ".mat", !is_memmap);
    cout << dpath + "/Client" + to_string(client_id) + ".mat" << endl;
}

void DecentralizedClient::remove_input_data(){
    remove((dpath + "/Client" + to_string(client_id) + ".mat").c_str());
    if(client_id == 0) remove((dpath + "/command.txt").c_str());
}

DecentralizedClient::~DecentralizedClient(){
    // Close client sockets
    for(int i=0; i<num_client; i++){
        if(i != client_id) close(client_socket[i]);
    }
    delete[] x_diagonal;
    delete[] x_upper_diagonal;
    delete[] message_queue;
    // Waiting server to finish
    pthread_join(server_tid, NULL);
    free_all_space();
}

void DecentralizedClient::init_server_and_client(){
    // Init server
    struct ReceiverMeta receiver_meta = {.port=my_server_port, .num_client=num_client};
    pthread_create(&server_tid, NULL, create_server, (void*)&receiver_meta); sleep(2);
    // Listen to other Clients
    int util_pos = listen.rfind(".");
    for(int i=0; i<num_client; i++){
        if(i == client_id) continue;
        server_addr[i] = new struct sockaddr_in();
        server_addr[i]->sin_family = AF_INET;
        server_addr[i]->sin_port = htons(base_server_port + i);
        string target_server_addr = listen.substr(0, util_pos+1) + to_string(stoi(listen.substr(util_pos+1, listen.size())) + i);
        server_addr[i]->sin_addr.s_addr = inet_addr(target_server_addr.c_str());
        cout << "Client-" << client_id << " connecting to " << target_server_addr << endl;
        client_socket[i] = socket(AF_INET, SOCK_STREAM, 0);
        if(client_socket[i] == -1){
            cout << "Socket Error" << endl;
            exit(-1);
        }
        while(connect(client_socket[i], (struct sockaddr*)server_addr[i], sizeof(*(server_addr[i]))) < 0){usleep(1);}
        LOG(INFO) << "About to Send Client ID";
        send(client_socket[i], (char*)&client_id, 4, 0);
        LOG(INFO) << "Client ID Sent";
    }
}

struct SendToData{
    int cid;
    char* buffer;
    size_t size;
};

void DecentralizedClient::send_to(int cid, char* buffer, size_t size){
    // 1) send data size
    send(client_socket[cid], (char*)&size, sizeof(size), 0);
    // 2) send actural data
    if(size > max_comm_size){
        // Send by batch
        for(size_t i=0; i<size; i+=max_comm_size){
            cout << "Send batch " << i << " " << min(max_comm_size, size-i) << endl;
            send(client_socket[cid], buffer+i, min(max_comm_size, size-i), 0);
        }
    }
    else send(client_socket[cid], buffer, size, 0);
}

template <typename T> void DecentralizedClient::receive(T** buffer, int cid){
    while(true){
        sem_wait(&sem_x);
        if(!message_queue[cid].empty()){
            sem_post(&sem_x);
            break;
        }
        sem_post(&sem_x);
        usleep(1);
    }
    sem_wait(&sem_x);
    *buffer = (T*) message_queue[cid].front();
    message_queue[cid].pop();
    sem_post(&sem_x);
}

void DecentralizedClient::broadcast(char* buffer, size_t size){
    for(int i=0; i<num_client; i++){
        if(i == client_id) continue;
        send_to(i, buffer, size);
    }
}

inline void util_sum(double *a, double *b, size_t size){
    // b = a + b
    for(size_t i=0; i<size; i++) b[i] = b[i] + a[i];
}

template <class T> void util_replace(T *a, T *b, size_t size){
    // b = b
    #pragma omp parallel for
    for(size_t i=0; i<size; i++) b[i] = a[i];
}

void DecentralizedClient::ring_all_reduce_sum(double* buffer, int size){
    gettimeofday(&comm_start, NULL);
    if(size < (num_client * 10000)){
        broadcast((char*)buffer, size*8);
        double* recv_msg;
        for(int i=0; i<num_client; i++){
            if(i == client_id) continue;
            receive(&recv_msg, i);
            util_sum(recv_msg, buffer, size);
            free_space((void*)recv_msg);
        }
    }
    else{
        size_t *data_split = new size_t[num_client+1]();
        // size_t data_split[num_client+1] = {0};
        for(int i=0; i<num_client; i++) data_split[i+1] = (int)(size / num_client);
        if((size % num_client) != 0){
            for(int i=1; i<=(size % num_client); i++) data_split[i] += 1;
        }
        for(int i=2; i<=num_client; i++) data_split[i] += data_split[i-1];
        int di; // target data index
        double *recv_msg;
        // Scatter
        for(int i=0; i<(num_client-1); i++){
            di = (client_id - i + num_client) % num_client;
            send_to(my_next_client, (char*)(buffer+data_split[di]), (data_split[di+1]-data_split[di])*8);
            receive(&recv_msg, my_front_client);
            di = (di - 1 + num_client) % num_client;
            util_sum(recv_msg, buffer+data_split[di], data_split[di+1]-data_split[di]);
            free_space((void*)recv_msg);
        }
        // All gather
        for(int i=0; i<(num_client-1); i++){
            di = (client_id + 1 - i + num_client) % num_client;
            send_to(my_next_client, (char*)(buffer+data_split[di]), (data_split[di+1]-data_split[di])*8);
            receive(&recv_msg, my_front_client);
            di = (di - 1 + num_client) % num_client;
            util_replace(recv_msg, buffer+data_split[di], data_split[di+1]-data_split[di]);
            free_space((void*)recv_msg);
        }
    }
    gettimeofday(&comm_finish, NULL);
    time_communication += time_diff(comm_start, comm_finish);
}

void DecentralizedClient::all_gather(char** buffer, size_t size, int *c_group, int nc, int src){
    gettimeofday(&comm_start, NULL);
    if(size <= nc || nc == 2){
        if(client_id == src){
            for(int i=0; i<nc; i++) if(c_group[i] != client_id) send_to(c_group[i], *buffer, size);
        }
        else receive(buffer, src);  // Will allocate new space
    }
    else{
        size_t *data_split = new size_t[nc]();
        for(int i=0; i<nc-1; i++) data_split[i+1] = (int)(size / (nc-1));
        if((size % (nc-1)) != 0){
            for(int i=1; i<=(size % (nc-1)); i++) data_split[i] += 1;
        }
        for(int i=2; i<=nc-1; i++) data_split[i] += data_split[i-1];
        // Allocate space at target clients
        if(client_id != src) get_space((void**)buffer, size, is_memmap, filename_with_cid("tmp_gather.mat"));
        // Src sends slices to targets
        int counter = 0;
        char *recv_msg;
        int fake_cid, group_pre_cid, group_next_cid;
        for(int i=0; i<nc; i++){
            if(c_group[i] == src) continue;
            if(client_id == src) {
                send_to(c_group[i], (*buffer)+data_split[counter], data_split[counter+1]-data_split[counter]);
                cout << client_id << " send to " << c_group[i] << " " << data_split[counter+1]-data_split[counter] << endl;
            }
            else if(client_id == c_group[i]){
                fake_cid = counter;
                if(c_group[(i-1+nc)%nc] != src) group_pre_cid = c_group[(i-1+nc)%nc];
                else group_pre_cid = c_group[(i-2+nc)%nc];
                if(c_group[(i+1)%nc] != src) group_next_cid = c_group[(i+1)%nc];
                else group_next_cid = c_group[(i+2)%nc];
                receive(&recv_msg, src); 
                util_replace(recv_msg, (*buffer)+data_split[counter], data_split[counter+1]-data_split[counter]);
                free_space(recv_msg);
            }
            counter += 1;
        }
        if(client_id != src){
            int di; // target data index
            for(int i=0; i<(nc-2); i++){
                di = (fake_cid - i + nc-1) % (nc-1);
                send_to(group_next_cid, (char*)(*buffer+data_split[di]), data_split[di+1]-data_split[di]);
                receive(&recv_msg, group_pre_cid);
                di = (di - 1 + nc-1) % (nc-1);
                util_replace(recv_msg, (*buffer)+data_split[di], data_split[di+1]-data_split[di]);
                free_space((void*)recv_msg);
            }
        }
    }
    gettimeofday(&comm_finish, NULL);
    time_communication += time_diff(comm_start, comm_finish);
}

string DecentralizedClient::filename_with_cid(string filename){
    return log_dir + "/Client" + to_string(client_id) + "_" + filename;
}

void DecentralizedClient::local_qr_reduction(size_t m, size_t n, double **raw_x, double *q, bool transpose_x, bool compute_q){
    // Only supprt column-format, and m > n
    assert(m > n);
    double *x = *raw_x;
    double *qr_tau; get_space(&qr_tau, n, false, "qr_tau");

    // Use CblasColMajor for efficiency.
    LAPACKE_dgeqrf(CblasColMajor, m, n, x, m, qr_tau);
    if(compute_q){
        #pragma omp parallel for
        for(int i=0; i<n; i++)
        cblas_dcopy(m-i, x+i*m+i, 1, q+i*m+i, 1);
        // Generate q
        LAPACKE_dorgqr(CblasColMajor, m, n, n, q, m, qr_tau);
    }
    free_space(qr_tau);
    
    // Shrink X
    get_space(&shrinked_x, n*n, is_memmap, filename_with_cid("shrinked_x_" + to_string(m) + "_" + to_string(n) + ".mat"));
    
    if(transpose_x)
    for(size_t i=0; i<n; i++)
    cblas_dcopy(i+1, x+i*m, 1, shrinked_x+i, n);
    else
    for(size_t i=0; i<n; i++)
    cblas_dcopy(i+1, x+i*m, 1, shrinked_x+i*n, 1);
    free_space(x);
    *raw_x = shrinked_x;
}

void DecentralizedClient::run(){

    LOG(INFO) << "Start DecFedSVD";

    if(evaluate){
        get_space(&masked_x, m * local_n, is_memmap, filename_with_cid("masked_x.mat"));
        deep_copy(raw_x, masked_x, m, local_n);
    }
    else{
        // record masked_x/raw_x, which will be auto deleted (no evaluation)
        // since the raw data is modified during computation.
        masked_x = raw_x;
        record_space((void*)masked_x, m*local_n*8, is_memmap, dpath + "/Client" + to_string(client_id) + ".mat");
    } 

    LOG(INFO) << "Masked X Set";

    while(true){
        sem_wait(&sem_x);
        if(num_connection == (num_client - 1)){
            sem_post(&sem_x);
            break;
        }
        sem_post(&sem_x);
        usleep(1);
    }
    
    LOG(INFO) << "All Clients Connected, DecFedSVD Begin";

    gettimeofday(&run_begin, NULL);
    gettimeofday(&start, NULL);
    
    // Opt control
    bool sequential_cache_frequent_computation = false; // false is better
    bool all_gather_broadcast = true; // true is better
    double local_qr_threshold = 5.0; // only perform local qr then max(m/n, n/m) > local_qr_threshold
    bool dense_mask_opt = true;
    bool overlap_pipeline = true;
    if(opt_control == 0){dense_mask_opt = false; overlap_pipeline = false;}
    else if(opt_control == 1){dense_mask_opt = true; overlap_pipeline = false;}
    else if(opt_control == 2){dense_mask_opt = false; overlap_pipeline = true;}
    else if(opt_control == 3){dense_mask_opt = true; overlap_pipeline = true;}
    
    // SVD mode
    bool compute_u = svd_mode != 3 ? true : false;
    bool compute_v = svd_mode != 2 ? true : false;
    
    // 1) Init, sync client important information
    LOG(INFO) << "Sync client important information";
    struct ClientMeta cm = {.m= m, .local_n= local_n, .cid= client_id, .shared_seed=0};
    if(client_id == 0){
        public_seed=(unsigned int) rand();
        // public_seed = 0;
        cm.shared_seed = public_seed;
    }
    broadcast((char*)&cm, sizeof(cm));
    struct ClientMeta* recv_cm;
    global_n = local_n;
    reduced_global_n = (local_n/m)>local_qr_threshold?min(local_n, m):local_n;
    for(int i=0; i<num_client; i++){
        if(i == client_id) continue;
        receive(&recv_cm, i);
        local_ns[recv_cm->cid] = recv_cm->local_n;
        reduced_local_ns[recv_cm->cid] = (recv_cm->local_n/m)>local_qr_threshold?min(m, recv_cm->local_n):recv_cm->local_n;
        global_n += local_ns[recv_cm->cid];
        reduced_global_n += reduced_local_ns[recv_cm->cid];
        if(recv_cm->cid == 0) public_seed = recv_cm->shared_seed;
        free_space((void*)recv_cm);
    }
    local_ns[client_id] = local_n;
    reduced_local_ns[client_id] = (local_n/m)>local_qr_threshold?min(local_n, m):local_n;

    for(int i=0; i<num_client; i++){
        n_pos[i]=0;
        reduced_n_pos[i]=0;
        for(int j=0; j < i; j++){
            n_pos[i] += local_ns[j];
            reduced_n_pos[i] += reduced_local_ns[j];
        }
    }
    n_pos[num_client] = global_n;
    reduced_n_pos[num_client] = reduced_global_n;
    
    // Set private mask seed
    private_seed = (unsigned int) rand();

    // Mask & Preprocess
    size_t k = min(m, global_n);
    size_t local_k = min(k, local_n);
    size_t givens_mask_times = 32;
    double *shrinkedA;

    gettimeofday(&finish, NULL);
    time_init = time_diff(start, finish);
    
    // Data Mask && Reduction (local_n > m) && Data Re-partition
    if(m > global_n){
        // Mask X
        LOG(INFO) << "Starting to add mask (tall-and-skinny)";
        if(sequential_cache_frequent_computation && client_id != 0) {char *msg; receive(&msg, client_id-1);}
        gettimeofday(&start, NULL);
        if(dense_mask_opt){
            block_shift_mask(local_k, m, givens_mask_times, masked_x, true, true, private_seed);
            block_shift_mask(local_k, m, givens_mask_times, masked_x, true, false, public_seed);
        }
        else{
            // Dense mask when block_size = k or m
            block_mask(local_k, m, local_k, masked_x, true, true, private_seed);
            block_mask(local_k, m, m, masked_x, true, false, public_seed);
        }
        
        gettimeofday(&finish, NULL);
        if(sequential_cache_frequent_computation){
            char msg_s = 'Y';
            if(client_id < (num_client-1)){
                send_to(client_id+1, &msg_s, 1);
                char *msg_r; receive(&msg_r, num_client-1);
            }
            else broadcast(&msg_s, 1);
        }
        time_apply_mask = time_diff(start, finish);
        LOG(INFO) << "Masking Costs (tall-and-skinny) " << time_apply_mask;
    }
    else{
        // Data Reduction
        if(m < local_n && (local_n / m) > local_qr_threshold){
            LOG(INFO) << "Local QR pre-processing on the right (short-and-wide)";
            if(sequential_cache_frequent_computation && client_id != 0) {char *msg; receive(&msg, client_id-1);}
            gettimeofday(&start, NULL);
            if(compute_v) get_space(&local_v, k*local_n, is_memmap, filename_with_cid("local_v.mat"));
            local_qr_reduction(local_n, m, &masked_x, local_v, false, compute_v);
            gettimeofday(&finish, NULL);
            if(sequential_cache_frequent_computation){
                char msg_s = 'Y';
                if(client_id < (num_client-1)){
                    send_to(client_id+1, &msg_s, 1);
                    char *msg_r; receive(&msg_r, num_client-1);
                }
                else broadcast(&msg_s, 1);
            }
            time_local_qr = time_diff(start, finish);
            LOG(INFO) << "Local QR Costs(1) (short-and-wide) " << time_local_qr;
        }
        // Mask X
        LOG(INFO) << "Starting to add mask (short-and-wide)";
        gettimeofday(&start, NULL);
        if(dense_mask_opt){
            block_shift_mask(m, reduced_local_ns[client_id], givens_mask_times, masked_x, true, true, public_seed);
            block_shift_mask(m, reduced_local_ns[client_id], givens_mask_times, masked_x, true, false, private_seed);
        }
        else{
            // Dense mask when block_size = k or m
            LOG(DEBUG) << "reduced_local_ns[client_id] = " << reduced_local_ns[client_id];
            block_mask(m, reduced_local_ns[client_id], m, masked_x, true, true, public_seed);
            block_mask(m, reduced_local_ns[client_id], reduced_local_ns[client_id], masked_x, true, false, private_seed);
        }
        gettimeofday(&finish, NULL); 
        time_apply_mask = time_diff(start, finish);
        LOG(INFO) << "Masking Costs(1) (short-and-wide) " << time_apply_mask;
    }

    double *masked_x_repart;
    // u_reduction_tall_and_skinny is used to hold part of U when the matrix is tall and skinny
    double *u_reduction_tall_and_skinny;
    
    // Util Variables
    vector<size_t> m_part;
    for(int i=0; i<num_client; i++) m_part.push_back(m / num_client);
    for(int i=0; i<((int)m % num_client); i++) m_part[i] += 1;
    size_t max_m_part = 0; for(int i=0; i<num_client; i++) max_m_part = (max_m_part > m_part[i]) ? max_m_part : m_part[i];
    vector<size_t> m_part_pos;
    for(int i=0; i<num_client; i++){
        size_t tmp_counter = 0;
        for(int j=0; j<i; j++)
        tmp_counter += m_part[j];
        m_part_pos.push_back(tmp_counter);
    }
    m_part_pos.push_back(m);

    size_t counter;
    if(m > global_n){
        LOG(INFO) << "Starting to Re-distribute MaskedX (tall-and-skinny)";
        gettimeofday(&start, NULL);
        counter = 0;
        // Send Partitions
        get_space(&masked_x_repart, global_n*m_part[client_id], is_memmap, filename_with_cid("masked_x_repart.mat"));
        double* tmp_send; get_space(&tmp_send, max_m_part*local_n, is_memmap, filename_with_cid("tmp_send.mat"));
        for(int i=0; i<num_client; i++){
            if(i != client_id){
                // mkl_domatcopy('R', 'N', local_n, m_part[i], 1.0, masked_x+counter, m, tmp_send, m_part[i]);
                deep_copy('N', local_n, m_part[i], masked_x+counter, m, tmp_send, m_part[i]);
                send_to(i, (char*)tmp_send, local_n * m_part[i] * sizeof(double));
            }
            counter += m_part[i];
        }
        free_space(tmp_send);
        LOG(INFO) << "MaskedX Part Send (tall-and-skinny)";
        // Receive Partitions and reformat masked_x
        double* tmp_receive;
        for(int i=0; i<num_client; i++){
            if(i == client_id) 
            deep_copy('N', local_ns[i], m_part[client_id], masked_x+m_part_pos[i], m, masked_x_repart+n_pos[i]*m_part[client_id], m_part[client_id]);
            else{
                receive(&tmp_receive, i);
                deep_copy('N', local_ns[i], m_part[client_id], tmp_receive, m_part[client_id], masked_x_repart+n_pos[i]*m_part[client_id], m_part[client_id]);
                free_space((void*)tmp_receive);
            }
        }
        LOG(INFO) << "MaskedX Part Received (tall-and-skinny)";
        free_space(masked_x);
        masked_x = masked_x_repart;
        gettimeofday(&finish, NULL); time_exchange_low_dim = time_diff(start, finish);
        time_communication += time_exchange_low_dim;

        if(m_part[client_id] > global_n){
            LOG(INFO) << "Starting to Local QR Reduction (tall-and-skinny)";
            if(sequential_cache_frequent_computation && client_id != 0) {char *msg; receive(&msg, client_id-1);}
            gettimeofday(&start, NULL);
            if(compute_u) get_space(&u_reduction_tall_and_skinny, m_part[client_id]*global_n, is_memmap, filename_with_cid("u_reduction_tall_and_skinny.mat"));
            local_qr_reduction(m_part[client_id], global_n, &masked_x, u_reduction_tall_and_skinny, false, compute_u);
            gettimeofday(&finish, NULL);
            if(sequential_cache_frequent_computation){
                char msg_s = 'Y';
                if(client_id < (num_client-1)){
                    send_to(client_id+1, &msg_s, 1);
                    char *msg_r; receive(&msg_r, num_client-1);
                }
                else broadcast(&msg_s, 1);
            }
            time_local_qr = time_diff(start, finish);
            LOG(INFO) << "Local QR Reduction (tall-and-skinny) Costs " << time_local_qr;
        }
    }
    else{
        LOG(INFO) << "Starting to Re-distribute MaskedX (short-and-wide)";
        gettimeofday(&start, NULL);
        counter = 0;
        get_space(&masked_x_repart, reduced_global_n*m_part[client_id], is_memmap, filename_with_cid("masked_x_repart.mat"));
        for(int i=0; i<num_client; i++){
            if(i != client_id) send_to(i, (char*)(masked_x+counter*reduced_local_ns[client_id]), reduced_local_ns[client_id] * m_part[i] * sizeof(double));
            counter += m_part[i];
        }
        // Receive Partitions and reformat masked_x
        double* tmp_receive;
        for(int i=0; i<num_client; i++){
            if(i == client_id)
            deep_copy('N', m_part[client_id], reduced_local_ns[i], masked_x+m_part_pos[i]*reduced_local_ns[i], reduced_local_ns[i], masked_x_repart+reduced_n_pos[i], reduced_global_n);
            else{
                receive(&tmp_receive, i);
                deep_copy('N', m_part[client_id], reduced_local_ns[i], tmp_receive, reduced_local_ns[i], masked_x_repart+reduced_n_pos[i], reduced_global_n);
                free_space((void*)tmp_receive);
            }
        }
        free_space(masked_x);
        masked_x = masked_x_repart;
        gettimeofday(&finish, NULL); time_exchange_low_dim = time_diff(start, finish);
        time_communication += time_exchange_low_dim;
        
        // Global QR Reduction
        if(reduced_global_n > m){
            gettimeofday(&start, NULL);
            LOG(INFO) << "Starting to global QR reduction";
            if(compute_v) get_space(&mask_shared_v, reduced_global_n*m, is_memmap, filename_with_cid("masked_shared_v.mat"));
            double* qr_tau = new double[m]();
            for(int i=0; i<num_client; i++){
                size_t size_total_send = (2 * reduced_global_n - 1 - m_part_pos[i] - m_part_pos[i+1]) * (m_part_pos[i+1] - m_part_pos[i]) / 2 + (m_part_pos[i+1] - m_part_pos[i]);
                if(i == client_id){
                    // Here n_pos[i] -> m_part_pos?
                    LAPACKE_dgeqrf(CblasColMajor, reduced_global_n-m_part_pos[i], m_part[i], masked_x+m_part_pos[i], reduced_global_n, qr_tau+m_part_pos[i]);
                    double *msg_total;
                    get_space(&msg_total, size_total_send, is_memmap, filename_with_cid("msg_total.mat"));
                    size_t counter = 0;
                    for(int j=m_part_pos[i]; j<m_part_pos[i+1]; j++){
                        msg_total[counter++] = qr_tau[j];
                        cblas_dswap(reduced_global_n-1-j, masked_x+j+1+(j-m_part_pos[i])*reduced_global_n, 1, msg_total+counter, 1);
                        if(compute_v) cblas_dcopy(reduced_global_n-1-j, msg_total+counter, 1, mask_shared_v+j*reduced_global_n+j+1, 1);
                        counter += (reduced_global_n-1-j);
                    }
                    LOG(INFO) << "Message about to send total " << size_total_send*8;
                    if(!all_gather_broadcast){
                        if(compute_v) broadcast((char*)msg_total, size_total_send*8);
                        else for(int cid=client_id+1; cid<num_client; cid++) send_to(cid, (char*)msg_total, size_total_send*8);
                    }
                    else{
                        int *c_group;
                        if(compute_v){
                            c_group = new int[num_client]; for(int j=0; j<num_client; j++) c_group[j] = j;
                            all_gather((char**)&msg_total, size_total_send*8, c_group, num_client, client_id);
                        }
                        else{
                            if((num_client-client_id)>1){
                                c_group = new int[num_client-client_id]; for(int j=client_id; j<num_client; j++) c_group[j-client_id] = j;
                                LOG(INFO) << "All gather on " << num_client-client_id << " clients";
                                all_gather((char**)&msg_total, size_total_send*8, c_group, num_client-client_id, client_id);
                            }
                        }
                    }
                    LOG(INFO) << "CID=" << client_id << " Finish Local QR";
                    free_space(msg_total);
                }
                else{
                    if(compute_v || client_id > i){
                        if(!compute_v) get_space(&mask_shared_v, (reduced_global_n-m_part_pos[i])*(m_part_pos[i+1]-m_part_pos[i]), is_memmap, filename_with_cid("mask_shared_v.mat"));
                        LOG(INFO) << "CID=" << client_id << " Begin Receive QR From " << i;
                        LOG(INFO) << "Receive in total";
                        double *msg_recv_total;
                        if(!all_gather_broadcast) receive(&msg_recv_total, i);
                        else {
                            int *c_group;
                            if(compute_v){
                                c_group = new int[num_client](); for(int j=0; j<num_client; j++) c_group[j] = j;
                                all_gather((char**)&msg_recv_total, size_total_send*8, c_group, num_client, i);
                            }
                            else{
                                c_group = new int[num_client-i]; for(int j=i; j<num_client; j++) c_group[j-i] = j;
                                all_gather((char**)&msg_recv_total, size_total_send*8, c_group, num_client-i, i);
                            }
                        }
                        size_t counter = 0;
                        for(int j=m_part_pos[i]; j<m_part_pos[i+1]; j++){
                            qr_tau[j] = msg_recv_total[counter++];
                            if(compute_v) cblas_dcopy(reduced_global_n-1-j, msg_recv_total+counter, 1, mask_shared_v+j*reduced_global_n+j+1, 1);
                            else cblas_dcopy(reduced_global_n-1-j, msg_recv_total+counter, 1, mask_shared_v+(j-m_part_pos[i])*(reduced_global_n-m_part_pos[i])+j-m_part_pos[i]+1, 1);
                            counter += (reduced_global_n-1-j);
                        }
                        free_space((void*)msg_recv_total);
                        LOG(INFO) << "CID=" << client_id << " Finished Receive QR From " << i;
                    }
                    if(client_id > i){
                        if(compute_v)
                        LAPACKE_dormqr(CblasColMajor, 'L', 'T', reduced_global_n-m_part_pos[i], m_part[client_id], m_part[i], mask_shared_v+m_part_pos[i]*reduced_global_n+m_part_pos[i], reduced_global_n, qr_tau+m_part_pos[i], masked_x+m_part_pos[i], reduced_global_n);
                        else
                        LAPACKE_dormqr(CblasColMajor, 'L', 'T', reduced_global_n-m_part_pos[i], m_part[client_id], m_part[i], mask_shared_v, reduced_global_n-m_part_pos[i], qr_tau+m_part_pos[i], masked_x+m_part_pos[i], reduced_global_n);
                        LOG(INFO) << "CID=" << client_id << " Finish Apply QR From " << i;
                        if(!compute_v) free_space(mask_shared_v);
                    }
                }
            }
            if(compute_v) LAPACKE_dorgqr(CblasColMajor, reduced_global_n, m, m, mask_shared_v, reduced_global_n, qr_tau);
            delete[] qr_tau;
            // Shrink masked_x
            get_space(&shrinked_x, m * m_part[client_id], false, filename_with_cid("shrinked_x.mat"));
            // mkl_domatcopy('R', 'T', m_part[client_id], m, 1.0, masked_x, reduced_global_n, shrinked_x, m_part[client_id]);
            deep_copy('T', m_part[client_id], m, masked_x, reduced_global_n, shrinked_x, m_part[client_id]);
            free_space(masked_x);
            masked_x = shrinked_x;
            gettimeofday(&finish, NULL); time_global_qr = time_diff(start, finish);
        }
        else{
            // Inplace Transpose X
            mkl_dimatcopy('R', 'T', m_part[client_id], reduced_global_n, 1.0, masked_x, reduced_global_n, m_part[client_id]);
        }
    }
    
    LOG(INFO) << "Starting to Ralha Bidiagonalization";
    gettimeofday(&start, NULL);
    
    size_t bidiagonal_m, bidiagonal_local_n, bidiagonal_n;
    if(m > global_n){
        bidiagonal_m = global_n;
        bidiagonal_local_n = min(m_part[client_id], global_n);
        bidiagonal_n = 0;
        for(int i=0; i<num_client; i++) bidiagonal_n += min(m_part[i], global_n);
    }
    else{
        bidiagonal_m = m;
        bidiagonal_local_n = m_part[client_id];
        bidiagonal_n = m;
    }
    // util variables
    double *vt_x = new double[bidiagonal_m]();
    double *ralha_bidiagonal_house, *ralha_bidiagonal_v;
    double *ralha_bidiagonal_tau = new double[bidiagonal_m]();

    // Allocate Space
    get_space(&ralha_bidiagonal_house, bidiagonal_m*bidiagonal_m, is_memmap, filename_with_cid("bidiag_house.mat"));
    ralha_bidiagonal_house[0] = 1.0;
    get_space(&ralha_bidiagonal_v, bidiagonal_m*bidiagonal_local_n, is_memmap, filename_with_cid("ralha_bidiagonal_v.mat"));
    
    x_diagonal = new double[bidiagonal_m]();
    x_upper_diagonal = new double[bidiagonal_m-1]();

    if(overlap_pipeline){
        size_t msg_size = bidiagonal_m+3;
        double* msg_send = new double[msg_size]();
        
        x_diagonal[0] = pow(cblas_dnrm2(bidiagonal_local_n, masked_x, 1), 2.0);
        ring_all_reduce_sum(x_diagonal, 1);
        x_diagonal[0] = pow(x_diagonal[0], 0.5);
        cblas_daxpy(bidiagonal_local_n, 1.0/x_diagonal[0], masked_x, 1, ralha_bidiagonal_v, 1);

        double td[7] = {0};
        
        for(int i=0; i<bidiagonal_m-2; i++){
            gettimeofday(&t1, NULL);
            cblas_dgemv(
                CblasRowMajor, CblasNoTrans, bidiagonal_m-1-i, bidiagonal_local_n, 1.0, masked_x+(i+1)*bidiagonal_local_n, bidiagonal_local_n, masked_x+i*bidiagonal_local_n, 1, 0, 
                msg_send, 1
            );
            gettimeofday(&t2, NULL); td[0] += time_diff(t1, t2);

            gettimeofday(&t1, NULL);
            if(i > 0){
                msg_send[bidiagonal_m-i-1] = cblas_ddot(bidiagonal_local_n, masked_x+i*bidiagonal_local_n, 1, ralha_bidiagonal_v+(i-1)*bidiagonal_local_n, 1);
                msg_send[bidiagonal_m-i] = cblas_ddot(bidiagonal_local_n, ralha_bidiagonal_v+(i-1)*bidiagonal_local_n, 1, ralha_bidiagonal_v+(i-1)*bidiagonal_local_n, 1);
                msg_send[bidiagonal_m-i+1] = cblas_ddot(bidiagonal_local_n, masked_x+i*bidiagonal_local_n, 1, masked_x+i*bidiagonal_local_n, 1);
                msg_send[bidiagonal_m-i+2] = cblas_ddot(bidiagonal_local_n, ralha_bidiagonal_v+(i-1)*bidiagonal_local_n, 1, masked_x+i*bidiagonal_local_n, 1);
            }
            gettimeofday(&t2, NULL); td[1] += time_diff(t1, t2);
            
            // Ring Reduce Sum
            gettimeofday(&t1, NULL);
            ring_all_reduce_sum(msg_send, msg_size-i);
            gettimeofday(&t2, NULL); td[2] += time_diff(t1, t2);
            
            gettimeofday(&t1, NULL);

            cblas_dcopy(bidiagonal_m-1-i, msg_send, 1, ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), bidiagonal_m);
            if(i > 0) x_upper_diagonal[i-1] = msg_send[bidiagonal_m-i-1];
            
            // Generate Householder
            LAPACKE_dlarfg(bidiagonal_m-1-i, ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1)+bidiagonal_m, bidiagonal_m, ralha_bidiagonal_tau+1+i);
            ralha_bidiagonal_house[(i+1)*(bidiagonal_m+1)] = 1.0;
            gettimeofday(&t2, NULL); td[3] += time_diff(t1, t2);
            
            // Apply Householder
            gettimeofday(&t1, NULL);
            cblas_dgemv(CblasColMajor, CblasNoTrans, bidiagonal_local_n, bidiagonal_m-1-i, 1.0, masked_x+(i+1)*bidiagonal_local_n, bidiagonal_local_n, ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), bidiagonal_m, 0, vt_x, 1);
            gettimeofday(&t2, NULL); td[4] += time_diff(t1, t2);

            gettimeofday(&t1, NULL);
            cblas_dger(CblasColMajor, bidiagonal_local_n, bidiagonal_m-1-i, -1 * *(ralha_bidiagonal_tau+1+i), vt_x, 1, ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), bidiagonal_m, masked_x+(i+1)*bidiagonal_local_n, bidiagonal_local_n);
            gettimeofday(&t2, NULL); td[5] += time_diff(t1, t2);

            gettimeofday(&t1, NULL);
            if(i > 0){
                cblas_daxpy(bidiagonal_local_n, -x_upper_diagonal[i-1], ralha_bidiagonal_v+(i-1)*bidiagonal_local_n, 1, masked_x+i*bidiagonal_local_n, 1);
                x_diagonal[i] = pow(msg_send[bidiagonal_m-i] * pow(x_upper_diagonal[i-1], 2) + msg_send[bidiagonal_m-i+1] - 2 * x_upper_diagonal[i-1] * msg_send[bidiagonal_m-i+2], 0.5);
                cblas_daxpy(bidiagonal_local_n, 1.0/x_diagonal[i], masked_x+i*bidiagonal_local_n, 1, ralha_bidiagonal_v+i*bidiagonal_local_n, 1);
            }
            gettimeofday(&t2, NULL); td[6] += time_diff(t1, t2);
            
            // LOG(INFO) << "Time " << td[0] / (i+1) << " " << td[1] / (i+1) << " " << td[2] / (i+1) << " " << td[3] / (i+1) << " " << td[4] / (i+1) << " " << td[5] / (i+1) << " " << td[6] / (i+1);
            // cout << "i=" << i << " alpha=" << x_diagonal[i] << " beta=" << x_upper_diagonal[i] << endl;
            
        }
        
        for(int i=0; i<6; i++) td[i] /= (bidiagonal_m-2);
        LOG(INFO) << "Time " << td[0] << " " << td[1] << " " << td[2] << " " << td[3] << " " << td[4] << " " << td[5] << " " << td[6];
        
        for(int i=bidiagonal_m-2; i<bidiagonal_m; i++){
            x_upper_diagonal[i-1] = cblas_ddot(bidiagonal_local_n, masked_x+i*bidiagonal_local_n, 1, ralha_bidiagonal_v+(i-1)*bidiagonal_local_n, 1);
            ring_all_reduce_sum(x_upper_diagonal+i-1, 1);
            cblas_daxpy(bidiagonal_local_n, -x_upper_diagonal[i-1], ralha_bidiagonal_v+(i-1)*bidiagonal_local_n, 1, masked_x+i*bidiagonal_local_n, 1);
            x_diagonal[i] = pow(cblas_dnrm2(bidiagonal_local_n, masked_x+i*bidiagonal_local_n, 1), 2.0);
            ring_all_reduce_sum(x_diagonal+i, 1);
            x_diagonal[i] = pow(x_diagonal[i], 0.5);
            cblas_daxpy(bidiagonal_local_n, 1.0/x_diagonal[i], masked_x+i*bidiagonal_local_n, 1, ralha_bidiagonal_v+i*bidiagonal_local_n, 1);
        }

        delete[] msg_send;
        delete[] vt_x;
        free_space(masked_x);
    }
    else{
        LOG(INFO) << "Not overlapping the pipeline!";
        ralha_bidiagonal_house[0] = 1.0;
        double* msg_send = new double[bidiagonal_m-1]();
        for(int i=0; i<bidiagonal_m-2; i++){
            cblas_dgemv(
                CblasRowMajor, CblasNoTrans, bidiagonal_m-1-i, bidiagonal_local_n, 1.0, masked_x+(i+1)*bidiagonal_local_n, bidiagonal_local_n, masked_x+i*bidiagonal_local_n, 1, 0, 
                ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), bidiagonal_m
            );
            // Send 
            cblas_dcopy(bidiagonal_m-1-i, ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), bidiagonal_m, msg_send, 1);
            ring_all_reduce_sum(msg_send, bidiagonal_m-1-i);
            cblas_dcopy(bidiagonal_m-1-i, msg_send, 1, ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), bidiagonal_m);
            // Generate Householder
            LAPACKE_dlarfg(bidiagonal_m-1-i, ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1)+bidiagonal_m, bidiagonal_m, ralha_bidiagonal_tau+1+i);
            ralha_bidiagonal_house[(i+1)*(bidiagonal_m+1)] = 1.0;
            // Apply Householder
            cblas_dgemv(CblasRowMajor, CblasTrans, bidiagonal_m-1-i, bidiagonal_local_n, 1.0, masked_x+(i+1)*bidiagonal_local_n, bidiagonal_local_n, ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), bidiagonal_m, 0, vt_x, 1);
            cblas_dger(CblasRowMajor, bidiagonal_m-1-i, bidiagonal_local_n, -1 * *(ralha_bidiagonal_tau+1+i), ralha_bidiagonal_house+(i+1)*(bidiagonal_m+1), bidiagonal_m, vt_x, 1, masked_x+(i+1)*bidiagonal_local_n, bidiagonal_local_n);
        }
        LOG(INFO) << "Compueta Alpha, Beta, and V";
        x_diagonal[0] = pow(cblas_dnrm2(bidiagonal_local_n, masked_x, 1), 2.0);
        ring_all_reduce_sum(x_diagonal, 1);
        x_diagonal[0] = pow(x_diagonal[0], 0.5);
        cblas_daxpy(bidiagonal_local_n, 1.0/x_diagonal[0], masked_x, 1, ralha_bidiagonal_v, 1);
        for(int i=1; i<bidiagonal_m; i++){
            x_upper_diagonal[i-1] = cblas_ddot(bidiagonal_local_n, masked_x+i*bidiagonal_local_n, 1, ralha_bidiagonal_v+(i-1)*bidiagonal_local_n, 1);
            ring_all_reduce_sum(x_upper_diagonal+i-1, 1);
            cblas_daxpy(bidiagonal_local_n, -x_upper_diagonal[i-1], ralha_bidiagonal_v+(i-1)*bidiagonal_local_n, 1, masked_x+i*bidiagonal_local_n, 1);
            x_diagonal[i] = pow(cblas_dnrm2(bidiagonal_local_n, masked_x+i*bidiagonal_local_n, 1), 2.0);
            ring_all_reduce_sum(x_diagonal+i, 1);
            x_diagonal[i] = pow(x_diagonal[i], 0.5);
            cblas_daxpy(bidiagonal_local_n, 1.0/x_diagonal[i], masked_x+i*bidiagonal_local_n, 1, ralha_bidiagonal_v+i*bidiagonal_local_n, 1);
        }
    }

    LOG(INFO) << "Lower Bi-diagonal to Upper Bi-diagonal";
    double beta_utils, *cs = new double[bidiagonal_m*2]();
    for(int i=0; i<bidiagonal_m-1; i++){
        beta_utils = x_upper_diagonal[i];
        cblas_drotg(x_diagonal+i, &beta_utils, cs+i*2, cs+i*2+1);
        x_upper_diagonal[i] = *(cs+i*2+1) * x_diagonal[i+1];
        x_diagonal[i+1] = *(cs+i*2) * x_diagonal[i+1];
    }
    
    gettimeofday(&finish, NULL); time_bi_diag = time_diff(start, finish);
    LOG(INFO) << "One-side Householder reflector Costs " << time_bi_diag;
    LOG(INFO) << "Bidiagonalization Finished";

    // print_matrix("x_diagonal", 1, 1000, x_diagonal);
    // print_matrix("x_upper_diagonal", 1, 999, x_upper_diagonal);

    LOG(INFO) << "Starting to Bidiagonal SVD";
    gettimeofday(&start, NULL);
    double *sigma, *bidiagonal_u, *bidiagonal_v;
    double sigma_upper_bound = pow(cblas_dnrm2(bidiagonal_m, x_diagonal, 1), 2) + pow(cblas_dnrm2(bidiagonal_m-1, x_upper_diagonal, 1), 2);
    int sigma_count = negative_count(bidiagonal_m, x_diagonal, x_upper_diagonal, sigma_upper_bound);
    int sigma_count_zero = negative_count(bidiagonal_m, x_diagonal, x_upper_diagonal, 0);
    sigma_count -= sigma_count_zero;
    LOG(INFO) << "Ori Sigma Count = " << sigma_count;
    if(top_k > 0) sigma_count = min(top_k, sigma_count);
    LOG(INFO) << "Adj Sigma Count = " << sigma_count;
    get_space(&bidiagonal_u, bidiagonal_m * sigma_count, is_memmap, filename_with_cid("bidiagonal_u.mat"));
    get_space(&bidiagonal_v, bidiagonal_m * sigma_count, is_memmap, filename_with_cid("bidiagonal_v.mat"));
    if(svd_mode == 0 || svd_mode == 4){ // Full SVD or Linear Regression
        // Bidiagonal Divide and Conquer SVD
        double* _q; int * _iq;
        LAPACKE_dbdsdc(CblasRowMajor, 'U', 'I', sigma_count, x_diagonal, x_upper_diagonal, bidiagonal_u, sigma_count, bidiagonal_v, bidiagonal_m, _q, _iq);
        sigma = x_diagonal;
    }
    else{
        // Bidiagonal Bisection and Twisted SVD
        sigma = new double[sigma_count]();
        bidiagonal_bisection_twisted_svd(bidiagonal_m, x_diagonal, x_upper_diagonal, bidiagonal_u, bidiagonal_v, sigma, sigma_count);
    }

    // Adjust the sign of U and V such that all peers have exactly the same results
    int util_pointer;
    for(int i=0; i<sigma_count; i++){
        util_pointer = cblas_idamax(bidiagonal_m, bidiagonal_u+i, sigma_count);
        if(bidiagonal_u[i+util_pointer*sigma_count] < 0){
            cblas_dscal(bidiagonal_m, -1, bidiagonal_u+i, sigma_count);
            cblas_dscal(bidiagonal_m, -1, bidiagonal_v+i*bidiagonal_m, 1);
        }
    }
    gettimeofday(&finish, NULL); time_bsvd = time_diff(start, finish);

    LOG(INFO) << "Combining SVD together";
    gettimeofday(&start, NULL);
    // for lr
    double *sigma_ut_y;
    if(compute_u){
        /*
            m_part[client_id] <= global_n: ralha_bidiagonal_v, bidiagonal_v
            m_part[client_id] > global_n: u_reduction_tall_and_skinny
        */
        double *part_final_u; get_space(&part_final_u, sigma_count*m_part[client_id], is_memmap, filename_with_cid("part_final_u.mat"));
        // bidiagonal_v.T @ ralha_bidiagonal_v.T
        if(m_part[client_id] > global_n){
            double *tmp_final_u; get_space(&tmp_final_u, sigma_count*bidiagonal_local_n, is_memmap, filename_with_cid("tmp_final_u.mat"));
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sigma_count, bidiagonal_local_n, bidiagonal_m, 1.0, bidiagonal_v, bidiagonal_m, ralha_bidiagonal_v, bidiagonal_local_n, 0.0, tmp_final_u, bidiagonal_local_n);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sigma_count, m_part[client_id], bidiagonal_local_n, 1.0, tmp_final_u, bidiagonal_local_n, u_reduction_tall_and_skinny, m_part[client_id], 0.0, part_final_u, m_part[client_id]);
            free_space(tmp_final_u);
            free_space(u_reduction_tall_and_skinny);
        }
        else{
            // bidiagonal_local_n = m_part[client_id]
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sigma_count, bidiagonal_local_n, bidiagonal_m, 1.0, bidiagonal_v, bidiagonal_m, ralha_bidiagonal_v, bidiagonal_local_n, 0.0, part_final_u, bidiagonal_local_n);
        }
        free_space(bidiagonal_v);
        free_space(ralha_bidiagonal_v);

        if(svd_mode == 4){
            // Linear Regression
            get_space(&sigma_ut_y, sigma_count, is_memmap, filename_with_cid("sigma_ut_y.mat"));
            if(client_id == (num_client-1)){
                // Label Holder, add mask to y
                load_memmap_file(&y, dpath + "/y.mat", true);                
                if(dense_mask_opt){
                    block_shift_mask(1, m, givens_mask_times, y, true, false, public_seed);
                }
                else{
                    // Dense mask when block_size = k or m
                    block_mask(1, m, m, y, true, false, public_seed);
                }
                // Send masked y (part) to other parties
                for(int i=0; i<num_client-1; i++){
                    send_to(i, (char*)(y+m_part_pos[i]), m_part[i]*sizeof(double));
                }
                // Local Compute sigma_ut_y
                cblas_dgemv(CblasRowMajor, CblasNoTrans, sigma_count, m_part[client_id], 1.0, part_final_u, m_part[client_id], y+m_part_pos[client_id], 1, 0, sigma_ut_y, 1);
                free_space(y);
            }
            else{
                double *masked_y_part;
                receive(&masked_y_part, num_client-1);
                // Local Compute sigma_ut_y
                cblas_dgemv(CblasRowMajor, CblasNoTrans, sigma_count, m_part[client_id], 1.0, part_final_u, m_part[client_id], masked_y_part, 1, 0, sigma_ut_y, 1);
                free_space(masked_y_part);
            }
            for(size_t i=0; i<sigma_count; i++) sigma_ut_y[i] /= sigma[i];
        }
        else{
            // SVD/LSA/PCA
            // Broadcast FinalU
            get_space(&FinalU, m*sigma_count, is_memmap, filename_with_cid("FinalU.mat"));
            if(!all_gather_broadcast)
            {
                broadcast((char*)part_final_u, m_part[client_id]*sigma_count*8);
                for(int i=0; i<num_client; i++){
                    double *tmp_receive;
                    if(i == client_id) tmp_receive = part_final_u;
                    else receive(&tmp_receive, i);
                    if(m > global_n)
                    deep_copy('N', sigma_count, m_part[i], tmp_receive, m_part[i], FinalU+m_part_pos[i], m);
                    else
                    deep_copy('T', sigma_count, m_part[i], tmp_receive, m_part[i], FinalU+m_part_pos[i]*sigma_count, sigma_count);
                    free_space((void*)tmp_receive);
                }
            }
            else{
                gettimeofday(&comm_start, NULL);
                if(m > global_n) deep_copy('N', sigma_count, m_part[client_id], part_final_u, m_part[client_id], FinalU+m_part_pos[client_id], m);
                else deep_copy('T', sigma_count, m_part[client_id], part_final_u, m_part[client_id], FinalU+m_part_pos[client_id]*sigma_count, sigma_count);
                free_space(part_final_u);
                int di;
                for(int i=0; i<(num_client-1); i++){
                    di = (client_id - i + num_client) % num_client;
                    if(m > global_n){
                        double* msg_send; get_space(&msg_send, sigma_count*m_part[di], is_memmap, filename_with_cid("recv_msg_send.mat"));
                        deep_copy('N', sigma_count, m_part[di], FinalU+m_part_pos[di], m, msg_send, m_part[di]);
                        send_to(my_next_client, (char*)msg_send, sigma_count*m_part[di]*8);
                        free_space(msg_send);
                    }
                    else{
                        send_to(my_next_client, (char*)(FinalU+m_part_pos[di]*sigma_count), sigma_count*m_part[di]*8);
                    }
                    double *recv_msg; receive(&recv_msg, my_front_client);
                    di = (di - 1 + num_client) % num_client;
                    if(m > global_n) deep_copy('N', sigma_count, m_part[di], recv_msg, m_part[di], FinalU+m_part_pos[di], m);
                    else deep_copy('N', m_part[di], sigma_count, recv_msg, sigma_count, FinalU+m_part_pos[di]*sigma_count, sigma_count);
                    free_space(recv_msg);
                }
                gettimeofday(&comm_finish, NULL);
                time_communication += time_diff(comm_start, comm_finish);
            }
            // Remove Mask
            if(m > global_n){
                if(dense_mask_opt){
                    block_shift_mask(sigma_count, m, givens_mask_times, FinalU, false, false, public_seed);
                }
                else{
                    // Dense mask when block_size = k or m
                    block_mask(sigma_count, m, m, FinalU, false, false, public_seed);
                }
            } 
            else{
                if(dense_mask_opt){
                    block_shift_mask(m, sigma_count, givens_mask_times, FinalU, false, true, public_seed);
                }
                else{
                    // Dense mask when block_size = k or m
                    block_mask(m, sigma_count, m, FinalU, false, true, public_seed);
                }
            }
        }
    }

    double* tmp_final_v;
    if(compute_v){
        /*
        m >= global_n: bidiagonal_u, ralha_bidiagonal_house
        m < global_n: mask_shared_v
        m < local_n: local_v
        */
        // Givens + BidiagonalU
        for(int i=bidiagonal_m-2; i>=0; i--)
        cblas_drot(sigma_count, bidiagonal_u+i*sigma_count, 1, bidiagonal_u+(i+1)*sigma_count, 1, *(cs+i*2), -*(cs+i*2+1));
        if(m >= global_n){
            // Tall-and-skinny Matrix
            // Generate ralha_bidiagonal_house, the @ with bidiagonal_u
            int info = LAPACKE_dormqr(CblasRowMajor, 'L', 'N', (int)global_n, (int)sigma_count, (int)global_n, ralha_bidiagonal_house, global_n, ralha_bidiagonal_tau, bidiagonal_u, sigma_count);
            if(svd_mode == 4){
                // LR
                get_space(&w_lr, local_n, false, "w_lr");
                double *w_lr_global_with_mask; get_space(&w_lr_global_with_mask, global_n, false, "w_lr_global_with_mask");
                cblas_dgemv(CblasRowMajor, CblasNoTrans, global_n, sigma_count, 1.0, bidiagonal_u, sigma_count, sigma_ut_y, 1, 0, w_lr_global_with_mask, 1);
                for(int i=0; i<num_client; i++)
                if(i != client_id) send_to(i, (char*)(w_lr_global_with_mask+n_pos[i]), local_ns[i]*sizeof(double));
                // Copy local w_lr
                deep_copy(w_lr_global_with_mask+n_pos[client_id], w_lr, 1, local_n);
                // Receive w_lr from other parties
                for(int i=0; i<num_client; i++)
                if(i != client_id){
                    double *tmp_receive_w_lr;
                    receive(&tmp_receive_w_lr, i);
                    util_sum(tmp_receive_w_lr, w_lr, local_n);
                    free_space(tmp_receive_w_lr);
                }
                // Remove mask of w_lr
                if(dense_mask_opt){
                    block_shift_mask(1, local_n, givens_mask_times, w_lr, false, false, private_seed);
                }
                else{
                    // Dense mask when block_size = k or m
                    block_mask(1, local_n, local_n, w_lr, true, false, private_seed);
                }
            }
            else{
                // Other Mode
                get_space(&FinalVT, sigma_count*local_n, is_memmap, filename_with_cid("FinalVT.mat"));
                deep_copy('T', local_n, sigma_count, bidiagonal_u+n_pos[client_id]*sigma_count, sigma_count, FinalVT, local_n);
                // Remove Mask
                if(dense_mask_opt){
                    block_shift_mask(sigma_count, local_n, givens_mask_times, FinalVT, false, false, private_seed);
                }
                else{
                    // Dense mask when block_size = k or m
                    if(m > global_n) block_mask(sigma_count, local_n, local_n, FinalVT, true, false, private_seed);
                    else block_mask(sigma_count, local_n, local_n, FinalVT, false, false, private_seed);
                }
            }
        }
        else{
            assert(svd_mode != 4); // only implemented vertical-lr in this project, however, horizontal-lr is also supported.
            LAPACKE_dormqr(CblasRowMajor, 'L', 'N', m, sigma_count, m, ralha_bidiagonal_house, m, ralha_bidiagonal_tau, bidiagonal_u, sigma_count);
            string tmp_final_v_filename = (m < local_n) ? "tmp_final_v.mat" : "FinalVT.mat";
            get_space(&tmp_final_v, reduced_local_ns[client_id]*sigma_count, is_memmap, filename_with_cid(tmp_final_v_filename));
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, sigma_count, reduced_local_ns[client_id], m, 1.0, bidiagonal_u, sigma_count, mask_shared_v+reduced_n_pos[client_id], reduced_global_n, 0, tmp_final_v, reduced_local_ns[client_id]);
            // Remove Mask
            if(dense_mask_opt){
                block_shift_mask(sigma_count, reduced_local_ns[client_id], givens_mask_times, tmp_final_v, false, false, private_seed);
            }
            else{
                // Dense mask when block_size = k or m
                block_mask(sigma_count, reduced_local_ns[client_id], reduced_local_ns[client_id], tmp_final_v, false, false, private_seed);
            }
            if(m < local_n && (local_n / m) > local_qr_threshold){
                get_space(&FinalVT, sigma_count*local_n, is_memmap, filename_with_cid("FinalVT.mat"));
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sigma_count, local_n, reduced_local_ns[client_id], 1.0, tmp_final_v, reduced_local_ns[client_id], local_v, local_n, 0, FinalVT, local_n);
                free_space(local_v);
            }
            else{
                FinalVT = tmp_final_v;
            }
        }
        free_space(ralha_bidiagonal_house);
        free_space(bidiagonal_u);
    }
    gettimeofday(&finish, NULL); time_combine_svd_remove_mask = time_diff(start, finish);
    
    LOG(INFO) << "Finish DecFedSVD";
    gettimeofday(&run_finish, NULL);
    LOG(INFO) << "DecFedSVD totally Costs " << time_diff(run_begin, run_finish);
    LOG(INFO) << "Overall Communication Cost " << time_communication;
    LOG(INFO) << "time_init=" << time_init;
    LOG(INFO) << "time_local_qr=" << time_local_qr;
    LOG(INFO) << "time_global_qr=" << time_global_qr;
    LOG(INFO) << "time_apply_mask=" << time_apply_mask;
    LOG(INFO) << "time_exchange_low_dim=" << time_exchange_low_dim;
    LOG(INFO) << "time_bi_diag=" << time_bi_diag;
    LOG(INFO) << "time_bsvd=" << time_bsvd;
    LOG(INFO) << "time_combine_svd_remove_mask=" << time_combine_svd_remove_mask;
    
    if(evaluate){
        if(svd_mode != 4){
            // Orthogonal Test
            if(m > global_n){
                if(compute_u && client_id == 0) LOG(INFO) << "Final U OrthTest " << orthogonal_test(FinalU, sigma_count, m, true);
                if(compute_v) LOG(INFO) << "Final V OrthTest " << orthogonal_test(FinalVT, sigma_count, local_n, false);
            }else{
                if(compute_u && client_id == 0) LOG(INFO) << "Final U OrthTest " << orthogonal_test(FinalU, m, sigma_count, false);
            }
        }
        if(svd_mode == 0){
            double *v_ground_truth; load_memmap_file(&v_ground_truth, dpath + "/vt.mat");
            double singular_vector_error_vt = 0;
            for(size_t i=0; i<sigma_count; i++){
                double sigma_wise_error = 0;
                for(size_t j=0; j<local_n; j++)
                sigma_wise_error += pow(abs(FinalVT[i*local_n+j]) - abs(v_ground_truth[i*global_n+n_pos[client_id]+j]), 2);
                singular_vector_error_vt += sigma_wise_error;
                // if(client_id == 0) cout << " i=" << i << " err=" << sigma_wise_error;
            }
            ring_all_reduce_sum(&singular_vector_error_vt, 1);
            singular_vector_error_vt = pow(singular_vector_error_vt / (sigma_count * global_n), 0.5);
            // LOG(INFO) << "Singular Vector Error (RMSE) VT " << singular_vector_error_vt << endl;
            
            // Debug Informations
            // sleep(client_id);
            // print_matrix("FinalV", 1, local_n, FinalVT);
            // print_matrix("v_ground_truth", 1, global_n, v_ground_truth);

            // if(client_id == 0) print_matrix("Sigma", 1, sigma_count, sigma);
            
            if(m > global_n){
                mkl_dimatcopy('R', 'T', sigma_count, local_n, 1, FinalVT, local_n, sigma_count);
                LOG(INFO) << "Final SVD Error " << svd_reconstruction_error(FinalVT, sigma, FinalU, raw_x, local_n, m, sigma_count);

            }else{
                LOG(INFO) << "Final SVD Error " << svd_reconstruction_error(FinalU, sigma, FinalVT, raw_x, m, local_n, sigma_count);
            }
            // if(client_id == 0){
            //     if(m <= global_n) mkl_dimatcopy('R', 'T', sigma_count, m, 1, FinalU, m, sigma_count);
            //     double *u_ground_truth; load_memmap_file(&u_ground_truth, dpath + "/u.mat", false);
            //     double singular_vector_error_u = singular_vector_error_rmse(sigma_count, m, u_ground_truth, FinalU);
            //     LOG(INFO) << "Singular Vector Error (RMSE) U " << singular_vector_error_u;
            //     LOG(INFO) << "Singular Vector Error (RMSE) Average " << (singular_vector_error_u + singular_vector_error_vt) / 2;
            // }
        }
        if(svd_mode == 1 || svd_mode == 2){
            // Projection Distance of U (By client 0)
            if(client_id == 0){
                double *u_ground_truth;
                load_memmap_file(&u_ground_truth, dpath + "/u.mat", false);
                if(m <= global_n) mkl_dimatcopy('R', 'T', m, sigma_count, 1, FinalU, sigma_count, m);
                double error = 0;
                #pragma omp parallel for reduction(+ : error)
                for(int i=0; i<m; i++){
                for(int j=i; j<m; j++){
                    error += abs(
                        cblas_ddot(sigma_count, u_ground_truth+i, m, u_ground_truth+j, m) -
                        cblas_ddot(sigma_count, FinalU+i, m, FinalU+j, m) 
                    );
                }
                }
                LOG(INFO) << "Projection Distance U " << error / (m * (m+1) / 2);
            }
        }
        if(svd_mode == 1 || svd_mode == 3){
            // Projection Distance of VT
            double *v_ground_truth;
            load_memmap_file(&v_ground_truth, dpath + "/vt.mat");
            double error = 0;
            #pragma omp parallel for reduction(+ : error)
            for(int i=0; i<local_n; i++)
            for(int j=i; j<local_n; j++){
                error += abs(
                    cblas_ddot(sigma_count, FinalVT+i, local_n, FinalVT+j, local_n) - 
                    cblas_ddot(sigma_count, v_ground_truth+i+n_pos[client_id], global_n, v_ground_truth+j+n_pos[client_id], global_n)
                );
            }
            LOG(INFO) << "Projection Distance VT " << error / (local_n * (local_n+1) / 2);
        }
        if(svd_mode == 4){
            // reload y (since it is released before)
            load_memmap_file(&y, dpath + "/y.mat", true);
            // x @ w
            double *xw; get_space(&xw, m, false, "xw");
            cblas_dgemv(CblasRowMajor, CblasTrans, local_n, m, 1.0, raw_x, m, w_lr, 1, 0, xw, 1); // only vertical-lr is implemented
            // if(m >= global_n) cblas_dgemv(CblasRowMajor, CblasTrans, local_n, m, 1.0, raw_x, m, w_lr, 1, 0, xw, 1);
            // else cblas_dgemv(CblasRowMajor, CblasNoTrans, m, local_n, 1.0, raw_x, local_n, w_lr, 1, 0, xw, 1);
            ring_all_reduce_sum(xw, m);
            if(client_id == (num_client-1)){
                LOG(INFO) << "LR MSE Error " << mean_square_error(m, 1, xw, y);
            }
        }
    }
    LOG(INFO) << "Evaluation Done";
}
