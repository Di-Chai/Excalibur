#include "base.hpp"
#include "utils.hpp"
#include "bbtsvd.hpp"
#include "client.hpp"
using namespace std;

/*

0) Wine data
1) MNIST data
2) ML100K
3) ML25M
4) Synthetic
5) Synthetic LR

*/


void get_num_data_partitions(size_t **data_partitions, size_t n, int num_clients){
    *data_partitions = new size_t[num_clients]();
    for(int i=0; i<num_clients; i++){
        (*data_partitions)[i] = n / num_clients;
        if(i < (n % num_clients)) (*data_partitions)[i] += 1;
    }
}

void handle_cached_data(double *x, double *y, size_t m, size_t n, string dpath, int num_clients, bool evaluate, int svd_mode){
    size_t *data_partitions; get_num_data_partitions(&data_partitions, n + (svd_mode == 4 ? 1 : 0), num_clients);
    assert(num_clients <= n);
    double *yy; get_space(&yy, m, true, dpath + "/y.mat"); deep_copy(y, yy, m, 1);
    size_t counter = 0;
    for(int i=0; i<num_clients; i++){
        double *xx; get_space(&xx, m * data_partitions[i], true, dpath + "/Client" + to_string(i) + ".mat");
        for(int j=0; j<data_partitions[i]; j++){
            if(m > n) cblas_dcopy(m, x+counter+j, n, xx+j*m, 1);
            else cblas_dcopy(m, x+counter+j, n, xx+j, data_partitions[i]);
        }
        counter += data_partitions[i];
        if(svd_mode == 4 && i == (num_clients-1)){
            if(m > n) for(size_t j=0; j<m; j++) xx[(data_partitions[i]-1)*m + j] = 1.0;
            else for(size_t j=0; j<m; j++) xx[data_partitions[i]-1 + j*data_partitions[i]] = 1.0;
        }
    }
}


int main(int argc, char **argv){

    // omp_set_num_threads(omp_get_max_threads());
    
    // CL parameters
    unsigned int seed = atoi(argv[1]);
    int datasets = atoi(argv[2]);
    int num_clients = atoi(argv[3]);
    size_t m = atol(argv[4]);  // number of sample
    size_t n = atol(argv[5]);  // number of feature
    bool evaluate = atoi(argv[6]) == 0 ? false : true;
    int svd_mode = atoi(argv[7]);

    cout << "Generate Data Parameters:" << " datasets=" << datasets << " num_clients=" << num_clients << " m=" << m << " n=" << n << " evaluate=" << evaluate << " svd_mode=" << svd_mode << endl; 

    size_t k = min(m, n);
    assert(num_clients > 1);
    srand(seed);

    string base_path = "/data/";
    map<int, string> dpaths = {
        {0, base_path + "datasets/wine"},
        {1, base_path + "datasets/mnist"},
        {2, base_path + "datasets/ml100k"},
        {3, base_path + "datasets/ml25m"},
        {4, base_path + "datasets/synthetic"},
        {5, base_path + "datasets/syntheticlr"},
        {6, base_path + "datasets/synthetic_very_large"},
    };

    for(auto iter=dpaths.begin(); iter!=dpaths.end(); iter++){
        if(access(iter->second.c_str(), F_OK) == -1) mkdir(iter->second.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
    }
    
    switch (datasets)
    {
    case 0: {
        m = 6497;
        n = 11;
        double *x; load_memmap_file(&x, dpaths[datasets] + "/wine_x.mat");
        double *y; load_memmap_file(&y, dpaths[datasets] + "/wine_y.mat");
        handle_cached_data(x, y, m, n, dpaths[datasets], num_clients, true, svd_mode);
        break;
    };
    case 1: {
        m = 70000;
        n = 784;
        double *x; load_memmap_file(&x, dpaths[datasets] + "/mnist_x.mat");
        double *y; load_memmap_file(&y, dpaths[datasets] + "/mnist_y.mat");
        handle_cached_data(x, y, m, n, dpaths[datasets], num_clients, true, svd_mode);
        break;
    };
    case 2: {
        m = 943;
        n = 500;
        double *x; load_memmap_file(&x, dpaths[datasets] + "/ml100k_x.mat");
        double *y; load_memmap_file(&y, dpaths[datasets] + "/ml100k_y.mat");
        handle_cached_data(x, y, m, n, dpaths[datasets], num_clients, true, svd_mode);
        break;
    };
    case 3: {
        m = 62423;
        n = 162541;
        double *x; load_memmap_file(&x, dpaths[datasets] + "/ml25m_x.mat");
        double *y; load_memmap_file(&y, dpaths[datasets] + "/ml25m_y.mat");
        handle_cached_data(x, y, m, n, dpaths[datasets], num_clients, false, svd_mode);
        break;
    };
    case 5: {
        // Vertical Linear Regression
        size_t *data_partitions; get_num_data_partitions(&data_partitions, n+1, num_clients);
        size_t n_informative = n * 0.9;
        double *ground_truth; get_space(&ground_truth, n+1, true, dpaths[datasets] + "/ground_truth.mat");

        random_normal_matrix(ground_truth, 1, n_informative);
        random_uniform_matrix(ground_truth+n, 1, 1, -100, 100);

        cblas_dscal(n_informative, 100, ground_truth, 1);

        double *y; get_space(&y, m, true, dpaths[datasets] + "/y.mat");
        size_t counter = 0;
        for(int i=0; i<num_clients; i++){
            double *x; get_space(&x, m * data_partitions[i], true, dpaths[datasets] + "/Client" + to_string(i) + ".mat");
            random_normal_matrix(x, m, data_partitions[i]);
            if(m > n){
                if(i == (num_clients-1)) for(size_t j=0; j<m; j++) x[(data_partitions[i]-1)*m + j] = 1.0; // Bias term
                cblas_dgemv(CblasRowMajor, CblasTrans, data_partitions[i], m, 1.0, x, m, ground_truth+counter, 1, 1.0, y, 1); 
            }
            else{
                if(i == (num_clients-1)) for(size_t j=0; j<m; j++) x[data_partitions[i]-1 + j*data_partitions[i]] = 1.0; // Bias term
                cblas_dgemv(CblasRowMajor, CblasNoTrans, m, data_partitions[i], 1.0, x, data_partitions[i], ground_truth+counter, 1, 1.0, y, 1);
            }
            counter += data_partitions[i];
        }
        // Add noise to y
        double *noise; get_space(&noise, m, false, "None");
        random_normal_matrix(noise, 1, m);
        for(size_t i=0; i<m; i++) y[i] += noise[i];
        break;
    };
    case 4:{
        double alpha = 0.01;
        k = min(m, n);
        size_t *data_partitions; get_num_data_partitions(&data_partitions, n, num_clients);
        double *u, *s, *vt;
        get_space(&u, m*k, true, dpaths[datasets] + "/u.mat");
        get_space(&vt, k*n, true, dpaths[datasets] + "/vt.mat");

        // Generate random matrix then QR (slow, O(n^3))
        // random_normal_matrix(u, m, k);
        // random_normal_matrix(vt, k, n);
        // double *tau = new double[k]();
        // LAPACKE_dgeqrf(CblasColMajor, m, k, u, m, tau);
        // LAPACKE_dorgqr(CblasColMajor, m, k, k, u, m, tau);
        // LAPACKE_dgeqrf(CblasColMajor, n, k, vt, n, tau);
        // LAPACKE_dorgqr(CblasColMajor, n, k, k, vt, n, tau);

        // generate_random_orthogonal_matrix O(n^2) is faster than the above method
        generate_random_orthogonal_matrix(k, m, u, rand());
        generate_random_orthogonal_matrix(k, n, vt, rand());
        
        double *sigma = new double[k]();
        for(size_t i=0; i<k; i++) sigma[i] = pow(i+1, -alpha);
        if(m <= n){
            for(size_t i=0; i<k; i++) cblas_dscal(m, sigma[i], u+i*m, 1);
        }
        else{
            for(size_t i=0; i<k; i++) cblas_dscal(n, sigma[i], vt+i*n, 1);
        }

        // print_matrix("Ground Truth Sigma", 1, 10, sigma);
        
        size_t counter = 0;
        for(int i=0; i<num_clients; i++){
            double *client_x;
            get_space(&client_x, m*data_partitions[i], true, dpaths[datasets] + "/Client" + to_string(i) + ".mat");
            if(m > n) cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, data_partitions[i], k, 1.0, u, m, vt+counter, n, 0, client_x, m);
            else cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, data_partitions[i], m, k, 1.0, vt+counter, n, u, m, 0, client_x, data_partitions[i]);
            counter += data_partitions[i];
        }
        if(!evaluate){
            remove((dpaths[datasets] + "/u.mat").c_str());
            remove((dpaths[datasets] + "/vt.mat").c_str());
        }
        else{
            if(m <= n){
                for(size_t i=0; i<k; i++) cblas_dscal(m, 1 / sigma[i], u+i*m, 1);
            }
            else{
                for(size_t i=0; i<k; i++) cblas_dscal(n, 1 / sigma[i], vt+i*n, 1);
            }
        }
        break;
    };
    case 6:{
        // Directly generate random normal synthetic data
        // Only used in very large scale experimenting when 4 costs too much time generating the data
        k = min(m, n);
        size_t *data_partitions; get_num_data_partitions(&data_partitions, n, num_clients);
        for(int i=0; i<num_clients; i++){
            double *client_x;
            get_space(&client_x, m*data_partitions[i], true, dpaths[datasets] + "/Client" + to_string(i) + ".mat");
            random_normal_matrix(client_x, m, data_partitions[i]);
        }
        break;
    }
    }
    
    cout << "Data Generated!" << endl;
    
}