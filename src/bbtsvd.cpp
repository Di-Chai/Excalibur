#include "bbtsvd.hpp"

#define OPT_THRESHOLD 10000
#define RATIO_THRESHOLD 2
#define BLOACK_MASK_METHOD 1  // 0: Direct Orthogonal 1: Givens Rotation (more efficient)

int negative_count(int n, double* x_diagonal, double* x_upper_diagonal, double mu){
   int count = 0;
   double d = 1.0, t = 0.0;
   for(int i=0; i<n; i++){
      if(i > 0) t = t * (pow(x_upper_diagonal[i-1], 2) / d) - pow(mu, 2);
      else t = - pow(mu, 2);
      d = pow(x_diagonal[i], 2) + t;
      if(d < 0) count++;
   }
   return count;
}

void bisection_singular_values(int n, double* x_diagonal, double* x_upper_diagonal, double a, double b, double tol, int top_k, double* sigma, int* sigma_count){
   
   int nc_a = negative_count(n, x_diagonal, x_upper_diagonal, a);
   int nc_b = negative_count(n, x_diagonal, x_upper_diagonal, b);

   int nc_mid;
   *sigma_count = 0;
   if(nc_a <= nc_b && top_k > 0){
      struct BISECTION_SV job = {.low= a, .nc_low= nc_a, .high= b, .nc_high= nc_b};
      stack<struct BISECTION_SV> work_list;
      work_list.push(job);
      while(!work_list.empty()){
         job = work_list.top();
         work_list.pop();
         double mid = (job.low + job.high) / 2;
         if((job.high - job.low) <= tol || mid == job.low || mid == job.high){
            sigma[*sigma_count] = mid;
            *sigma_count = *sigma_count + 1;
            if(*sigma_count >= top_k) break;
         }
         else{
            nc_mid = negative_count(n, x_diagonal, x_upper_diagonal, mid);
            if(nc_mid > job.nc_low){
               struct BISECTION_SV new_job = {.low=job.low, .nc_low=job.nc_low, .high=mid, .nc_high=nc_mid};
               work_list.push(new_job);
            }
            if(job.nc_high > nc_mid){
               struct BISECTION_SV new_job = {.low=mid, .nc_low=nc_mid, .high=job.high, .nc_high=job.nc_high};
               work_list.push(new_job);
            }
         }
      }
   }
}

void ldl_decomposition(int n, double* x_diagonal, double* x_upper_diagonal, double* l, double* d){
   /* LDL Decomposition of XT@X */
   d[0] = pow(x_diagonal[0], 2);
   double ab;
   for(int i=1; i<n; i++){
      ab = x_diagonal[i-1] * x_upper_diagonal[i-1];
      l[i-1] = ab / d[i-1];
      d[i] = pow(x_diagonal[i], 2) + pow(x_upper_diagonal[i-1], 2) - pow(ab, 2) / d[i-1];
   }
}

void inverse_iteration(int n, int k, double* l, double* u, double* v, int ldv){
   v[k*ldv] = 1.0;
   for(int i=k-1; i>=0; i--) v[i*ldv] = -l[i] * v[(i+1)*ldv];
   for(int i=k+1; i<n; i++) v[i*ldv] = -u[i-1] * v[(i-1)*ldv];
   double v_norm_inv = 1.0 / cblas_dnrm2(n, v, ldv);
   cblas_dscal(n, v_norm_inv, v, ldv);
}

void dstqds(int n, double* l, double* d, double tau, double* l_plus, double* d_plus, double* s){
   s[0] = -tau;
   for(int i=0; i<n-1; i++){
      d_plus[i] = d[i] + s[i];
      l_plus[i] = d[i] * l[i] / d_plus[i];
      s[i+1] = l_plus[i] * l[i] * s[i] - tau;
   }
   d_plus[n-1] = d[n-1] + s[n-1];
}

void dpqds(int n, double *l, double* d, double tau, double* u_plus, double* r_plus, double* p){
   p[n-1] = d[n-1] - tau;
   for(int i=n-2; i>=0; i--){
      r_plus[i+1] = d[i] * pow(l[i], 2) + p[i+1];
      u_plus[i] = l[i] * d[i] / r_plus[i+1];
      p[i] = p[i+1] * d[i] / r_plus[i+1] - tau;
   }
   r_plus[0] = p[0];
}

void one_pair_singular_vector_twisted_factorization(int n, int omege_count, double omega, double* l, double* d, double* x_diagonal, double* x_upper_diagonal, double* v, double* u){
   double omege_2 = pow(omega, 2);

   double* l_plus = new double[n-1]();
   double* d_plus = new double[n]();
   double* s = new double[n + 1]();       // +1 for coupling
   double* u_plus = new double[n-1]();
   double* r_plus = new double[n + 1]();  // +1 for coupling
   double* p = new double[n + 1]();       // +1 for coupling
   double* _gamma = new double[n]();

   dstqds(n, l, d, omege_2, l_plus, d_plus, s);
   dpqds(n, l, d, omege_2, u_plus, r_plus, p);

   for(int i=0; i<n; i++) _gamma[i] = s[i] + p[i] + omege_2;
   int k = cblas_idamin(n, _gamma, 1);

   // Inverse iteration to get V
   inverse_iteration(n, k, l_plus, u_plus, v, 1);

   // The Coupling strategy
   s[n] = -omege_2; p[n] = 1.0; r_plus[n] = 1.0;
   for(int i=0; i<n; i++){
      d_plus[i] = s[i+1] / s[i] * d_plus[i];
      _gamma[i] = -omege_2 * _gamma[i] * r_plus[i+1] / (s[i] * p[i+1]);
   }
   k = cblas_idamin(n, _gamma, 1);
   for(int i=0; i<n; i++) r_plus[i] = p[i] / p[i+1] * r_plus[i+1];
   for(int i=0; i<n-1; i++){
      l_plus[i] = x_diagonal[i+1] / d_plus[i] * x_upper_diagonal[i];
      u_plus[i] = x_diagonal[i+1] / r_plus[i+1] * x_upper_diagonal[i];
   }
   // Inverse iteration to get U
   inverse_iteration(n, k, l_plus, u_plus, u, omege_count);
   // free memory
   delete[] l_plus;
   delete[] d_plus;
   delete[] s;
   delete[] u_plus;
   delete[] r_plus;
   delete[] p;
   delete[] _gamma;
}

// Only for bi-diagonal
void two_side_singular_vectors_twisted_factorization(int n, int omege_count, double* omege, double* x_diagonal, double* x_upper_diagonal, double* u, double* v){
   double* l = new double[n-1]();
   double* d = new double[n]();
   ldl_decomposition(n, x_diagonal, x_upper_diagonal, l, d);
   for(int i=0; i<omege_count; i++){
      one_pair_singular_vector_twisted_factorization(n, omege_count, omege[i], l, d, x_diagonal, x_upper_diagonal, v+i*n, u+i);
   }
   delete[] l;
   delete[] d;
}

void generate_house(int n, double* x, int incX, double* beta){
   double sigma = 0.0;
   for(int i=1; i<n; i++) sigma += pow(x[i*incX], 2);
   double x_0 = x[0]; x[0] = 1;
   if(sigma == 0 && x_0 >= 0)
      *beta = 0.0;
   else if (sigma == 0 && x_0 < 0)
      *beta = 2.0;
   else{
      double mu = pow(pow(x_0, 2) + sigma, 0.5);
      if(x_0 <= 0) x[0] = x_0 - mu;
      else x[0] = -sigma / (x_0 + mu);
      *beta = 2 * pow(x[0], 2) / (sigma + pow(x[0], 2));
      cblas_dscal(n, 1 / x[0], x, incX);
   }
}

void find_best_convert(struct SquareMaskConvert* smc){
    size_t threshold = OPT_THRESHOLD;
    assert(smc->k <= (threshold * threshold));
    size_t k = smc->k - smc->remain;
    size_t tmp_m = k, tmp_n = 1;
    for(size_t i=threshold; i>1; i--)
    if(k % i == 0){
        size_t k_div_i = k / i;
        if(i > k_div_i && k_div_i < threshold && k_div_i > 1 && (i - k_div_i) < (tmp_m - tmp_n) && ((double)i / (double)k_div_i) < RATIO_THRESHOLD){
            tmp_m = i;
            tmp_n = k / i;
        }
    }
    if(tmp_m < threshold){
        // Found satisfied results
        smc->m = tmp_m;
        smc->n = tmp_n;
    }
    else{
        smc->remain += 1;
        find_best_convert(smc);
    }
}

void naive_givens_mask_parallel(size_t m, size_t n, size_t mask_size, double* x, bool mask, bool is_left, unsigned int seed){
    // Set seed
    srand(seed);
    auto dre = default_random_engine(seed);
    // Generate random givens
    int k = is_left ? m : n;

    struct timeval start, finish;

    vector<vector<int>> pair_mask_index;
    vector<int> tmp_pmi;
    for(int i=0; i<k; i++) tmp_pmi.push_back(i);
    for(int i=0; i<mask_size; i++) {shuffle(tmp_pmi.begin(), tmp_pmi.end(), dre); pair_mask_index.push_back(tmp_pmi);}

    double *c, *s;
    double cs_root;
    int one_pass_size = (k - (k%2))/2;
    int cs_size = mask_size * one_pass_size;
    get_space(&c, cs_size, false, "None"); random_uniform_vector(cs_size, c, 1, -1, 1);
    get_space(&s, cs_size, false, "None"); random_uniform_vector(cs_size, s, 1, -1, 1);
    for(int i=0; i<cs_size; i++){
        cs_root = pow(pow(c[i], 2)+pow(s[i], 2), 0.5);
        c[i] /= cs_root;
        s[i] /= cs_root;
    }

    for(int i=0; i<mask_size; i++){
        #pragma omp parallel for
        for(int j=0; j<one_pass_size; j++){
            int cs_index, pmi_index;
            cs_index = i*one_pass_size+j;
            if(mask) pmi_index = i; else pmi_index = mask_size-1-i;
            if(is_left) cblas_drot(n, x + pair_mask_index[pmi_index][j*2] * n, 1, x + pair_mask_index[pmi_index][j*2+1] * n, 1, c[cs_index], (mask?1:-1) * s[cs_index]);
            else  cblas_drot(m, x + pair_mask_index[pmi_index][j*2], n, x + pair_mask_index[pmi_index][j*2+1], n, c[cs_index], (mask?1:-1) * s[cs_index]);
        }
    }
}

void givens_mask_cache_optimized_parallel(size_t m, size_t n, size_t mask_times, double* x, bool mask, bool is_left, unsigned int seed){
    // Set seed
    srand(seed); auto dre = default_random_engine(seed);

    // Generate random givens
    int k = is_left ? m : n;
    vector<vector<int>> pair_mask_index;
    for(int i=0; i<mask_times; i++){
        vector<int> tmp_pmi;
        for(int i=0; i<k; i++) tmp_pmi.push_back(i);
        shuffle(tmp_pmi.begin(), tmp_pmi.end(), dre); 
        pair_mask_index.push_back(tmp_pmi);
    }

    if(!mask) reverse(pair_mask_index.begin(), pair_mask_index.end());

    int one_pass_size = (k - (k%2))/2;
    int cs_size = mask_times * one_pass_size;
    double *c = new double[cs_size]{}, *s = new double[cs_size]{};
    random_uniform_vector(cs_size, c, 1, -1, 1);
    random_uniform_vector(cs_size, s, 1, -1, 1);
    
    double cs_root;
    for(int i=0; i<cs_size; i++){
        cs_root = pow(pow(c[i], 2)+pow(s[i], 2), 0.5);
        c[i] /= cs_root;
        s[i] /= cs_root;
    }
    
    if(is_left){
        #pragma omp parallel for
        for(int k=0; k<n; k+=1){
            double *x_block = new double[m]{};
            for(int i=0; i<m; i++) x_block[i] = x[i*n+k];
            for(int i=0; i<mask_times; i++){
                for(int j=0; j<one_pass_size; j++){
                    int cs_index; 
                    if(mask) cs_index = i*one_pass_size+j;
                    else cs_index = (mask_times-1-i)*one_pass_size+j;
                    int x1 = min(pair_mask_index[i][j*2], pair_mask_index[i][j*2+1]);
                    int x2 = max(pair_mask_index[i][j*2], pair_mask_index[i][j*2+1]);
                    double new_x = c[cs_index] * x_block[x1] + (mask?1:-1) * s[cs_index] * x_block[x2];
                    x_block[x2] = (mask?-1:1) * s[cs_index] * x_block[x1] + c[cs_index] * x_block[x2];
                    x_block[x1] = new_x;
                }
            }
            for(int i=0; i<m; i++) x[i*n+k] = x_block[i];
            delete[] x_block;
        }
    }
    else{
        #pragma omp parallel for
        for(int k=0; k<m; k+=1){
            double *x_block = new double[n]{};
            for(int i=0; i<n; i++) x_block[i] = x[k*n+i];
            for(int i=0; i<mask_times; i++){
                for(int j=0; j<one_pass_size; j++){
                    int cs_index; 
                    if(mask) cs_index = i*one_pass_size+j;
                    else cs_index = (mask_times-1-i)*one_pass_size+j;
                    int x1 = min(pair_mask_index[i][j*2], pair_mask_index[i][j*2+1]);
                    int x2 = max(pair_mask_index[i][j*2], pair_mask_index[i][j*2+1]);
                    double new_x = c[cs_index] * x_block[x1] + (mask?1:-1) * s[cs_index] * x_block[x2];
                    x_block[x2] = (mask?-1:1) * s[cs_index] * x_block[x1] + c[cs_index] * x_block[x2];
                    x_block[x1] = new_x;
                }
            }
            for(int i=0; i<n; i++) x[k*n+i] = x_block[i];
            delete[] x_block;
        }
    }
}

void generate_random_orthogonal_matrix(int matrix_layout, size_t m, double* house, double* tau, unsigned int seed){
    srand(seed);
    int iseed[4];
    double *tmp_house = new double[m];
    if(matrix_layout == CblasRowMajor){
        for(size_t i=0; i<m; i++){
            for(size_t r=0; r<4; r+=1) iseed[r] = rand() % 4096;
            if(iseed[3] % 2 == 0) iseed[3] = (iseed[3]+1) % 4096;
            LAPACKE_dlarnv(3, iseed, i+1, house+i*m);
        }
        for(size_t i=0; i<m-1; i++){
            for(size_t j=0; j<(m-i); j++) tmp_house[j] = house[i*m+i+j*m];
            LAPACKE_dlarfg(m-i, tmp_house, tmp_house+1, 1, tau+i);
            for(size_t j=0; j<(m-i); j++) house[i*m+i+j*m] = tmp_house[j];
        }
    }
    else{
        for(size_t i=0; i<m; i++){
            for(size_t r=0; r<4; r+=1) iseed[r] = rand() % 4096;
            if(iseed[3] % 2 == 0) iseed[3] = (iseed[3]+1) % 4096;
            LAPACKE_dlarnv(3, iseed, m-i, house+i*m+i);
        }
        for(size_t i=0; i<m-1; i++){
            for(size_t j=0; j<(m-i); j++) tmp_house[j] = house[i*m+i+j];
            LAPACKE_dlarfg(m-i, tmp_house, tmp_house+1, 1, tau+i);
            for(size_t j=0; j<(m-i); j++) house[i*m+i+j] = tmp_house[j];
        } 
    }
    house[m*m-1] = 1.0;
    delete[] tmp_house;
}

void generate_random_orthogonal_matrix(int matrix_layout, size_t m, size_t k, double* house, double* tau, unsigned int seed){
    srand(seed);
    int iseed[4];
    double *tmp_house = new double[m];
    assert(k <= m);
    if(matrix_layout == CblasRowMajor){
        // for(size_t i=0; i<m; i++){
        //     for(size_t r=0; r<4; r+=1) iseed[r] = rand() % 4096;
        //     if(iseed[3] % 2 == 0) iseed[3] = (iseed[3]+1) % 4096;
        //     LAPACKE_dlarnv(3, iseed, min(i+1, k), house+i*k);
        // }
        for(size_t r=0; r<4; r+=1) iseed[r] = rand() % 4096;
        if(iseed[3] % 2 == 0) iseed[3] = (iseed[3]+1) % 4096;
        LAPACKE_dlarnv(3, iseed, m*k, house);
        for(size_t i=0; i<min(k, m-1); i++){
            for(size_t j=0; j<(m-i); j++) tmp_house[j] = house[i*k+i+j*k];
            LAPACKE_dlarfg(m-i, tmp_house, tmp_house+1, 1, tau+i);
            for(size_t j=0; j<(m-i); j++) house[i*k+i+j*k] = tmp_house[j];
        }
    }
    else{
        // for(size_t i=0; i<m; i++){
        //     for(size_t r=0; r<4; r+=1) iseed[r] = rand() % 4096;
        //     if(iseed[3] % 2 == 0) iseed[3] = (iseed[3]+1) % 4096;
        //     LAPACKE_dlarnv(3, iseed, m-i, house+i*m+i);
        // }
        // for(size_t i=0; i<m-1; i++){
        //     for(size_t j=0; j<(m-i); j++) tmp_house[j] = house[i*m+i+j];
        //     LAPACKE_dlarfg(m-i, tmp_house, tmp_house+1, 1, tau+i);
        //     for(size_t j=0; j<(m-i); j++) house[i*m+i+j] = tmp_house[j];
        // } 
        assert(matrix_layout == CblasRowMajor); // ToDo: future work
    }
    if(k == m) {house[m*k-1] = 1.0;}
    delete[] tmp_house;
}

void generate_random_orthogonal_matrix(size_t m, size_t n, double* x, unsigned int seed){
    srand(seed);
    int iseed[4];
    double *tmp_house = new double[m];
    double *tau = new double[min(m, n)];
    if(m >= n){
        for(size_t i=0; i<m; i++){
            for(size_t r=0; r<4; r+=1) iseed[r] = rand() % 4096;
            if(iseed[3] % 2 == 0) iseed[3] = (iseed[3]+1) % 4096;
            LAPACKE_dlarnv(3, iseed, min(i+1, n), x+i*n);
        }
        for(size_t i=0; i<n-1; i++) LAPACKE_dlarfg(m-i, x+i*n+i, x+i*n+i+n, n, tau+i);
        LAPACKE_dorgqr(CblasRowMajor, m, n, n, x, n, tau);
    }
    else{
        for(size_t i=0; i<m; i++){
            for(size_t r=0; r<4; r+=1) iseed[r] = rand() % 4096;
            if(iseed[3] % 2 == 0) iseed[3] = (iseed[3]+1) % 4096;
            LAPACKE_dlarnv(3, iseed, n-i, x+i*n+i);
        }
        for(size_t i=0; i<m-1; i++) LAPACKE_dlarfg(n-i, x+i*n+i, x+i*n+i+1, 1, tau+i);
        LAPACKE_dorgqr(CblasColMajor, n, m, m, x, n, tau);
    }
    delete[] tmp_house;
    delete[] tau;
}

void shuffle_matrix(size_t m, size_t n, double *x, unsigned int seed, bool is_left, bool unshuffle){
    srand(seed);
    auto dre = default_random_engine(seed);
    size_t k = is_left ? m : n;

    vector<int> shuffle_index;
    for(size_t i=0; i<k; i++) shuffle_index.push_back(i);
    shuffle(shuffle_index.begin(), shuffle_index.end(), dre);
    
    size_t ri = 0;
    if(k % 2 != 0){
        ri = rand() % (k-1);
        if(unshuffle){
            if(is_left) cblas_dswap(n, x+n*(k-1), 1, x+n*ri, 1);
            else cblas_dswap(m, x+k-1, n, x+ri, n);
        }
    }
    
    size_t one_pass_size = (k - (k%2))/2;
    if(is_left)
    for(size_t i=0; i<one_pass_size; i++) 
    cblas_dswap(n, x+n*shuffle_index[2*i], 1, x+n*shuffle_index[2*i+1], 1);
    else
    for(size_t i=0; i<one_pass_size; i++) 
    cblas_dswap(m, x+shuffle_index[2*i], n, x+shuffle_index[2*i+1], n);
    
    if(k % 2 != 0 and !unshuffle){
        if(is_left) cblas_dswap(n, x+n*(k-1), 1, x+n*ri, 1);
        else cblas_dswap(m, x+k-1, n, x+ri, n);
    }
}

void block_mask(size_t m, size_t n, size_t block_size, double* x, bool mask, bool is_left, unsigned int seed){
    // allocate space for masks
    size_t k_batch = 10000; // reduce memory usage
    double *mask_house; get_space(&mask_house, block_size*min(block_size, k_batch), false, "None");
    double *mask_tau; get_space(&mask_tau, min(block_size-1, k_batch), false, "None");

    if((is_left && m <= 1) || (!is_left && n <= 1)) return;
    if(is_left)
    for(size_t i=0; i<m; i+=block_size){
        size_t real_block_size = min(block_size, m-i);
        if(real_block_size < k_batch){
            generate_random_orthogonal_matrix(CblasRowMajor, real_block_size, mask_house, mask_tau, seed);
            LAPACKE_dormqr(CblasRowMajor, 'L', mask?'N':'T', real_block_size, n, real_block_size, mask_house, real_block_size, mask_tau, x+i*n, n);
        }
        else{
            int counter = 0;
            vector<size_t> batch_index, real_k_batch;
            for(size_t j=0; j<real_block_size; j+=k_batch){
                batch_index.push_back(j);
                real_k_batch.push_back(min(k_batch, real_block_size-j));
                counter += 1;
            }
            real_k_batch[counter-1] -= 1;
            if(mask){reverse(real_k_batch.begin(), real_k_batch.end()); reverse(batch_index.begin(), batch_index.end());}
            for(int j=0; j<counter; j+=1){
                if(real_k_batch[j] == 0) continue;
                generate_random_orthogonal_matrix(CblasRowMajor, real_block_size-batch_index[j], real_k_batch[j], mask_house, mask_tau, seed);
                LAPACKE_dormqr(CblasRowMajor, 'L', mask?'N':'T', real_block_size-batch_index[j], n, real_k_batch[j], mask_house, real_k_batch[j], mask_tau, x+(i+batch_index[j])*n, n);
                for(size_t k=0; k<(real_block_size-batch_index[j])*real_k_batch[j]; k++) mask_house[k] = 0;
            }
        }
    }
    else
    for(size_t i=0; i<n; i+=block_size){
        size_t real_block_size = min(block_size, n-i);
        if(real_block_size < k_batch){
            generate_random_orthogonal_matrix(CblasRowMajor, real_block_size, mask_house, mask_tau, seed);
            LAPACKE_dormqr(CblasRowMajor, 'R', mask?'N':'T', m, real_block_size, real_block_size, mask_house, real_block_size, mask_tau, x+i, n);
        }
        else{
            int counter = 0;
            vector<size_t> batch_index, real_k_batch;
            for(size_t j=0; j<real_block_size; j+=k_batch){
                batch_index.push_back(j);
                real_k_batch.push_back(min(k_batch, real_block_size-j));
                counter += 1;
            }
            real_k_batch[counter-1] -= 1;
            if(mask){reverse(real_k_batch.begin(), real_k_batch.end()); reverse(batch_index.begin(), batch_index.end());}
            for(size_t j=0; j<counter; j+=1){
                if(real_k_batch[j] == 0) continue;
                generate_random_orthogonal_matrix(CblasRowMajor, real_block_size-batch_index[j], real_k_batch[j], mask_house, mask_tau, seed);
                LAPACKE_dormqr(CblasRowMajor, 'R', mask?'N':'T', m, real_block_size-batch_index[j], real_k_batch[j], mask_house, real_k_batch[j], mask_tau, x+i+batch_index[j], n);
            }
        }
    }
    free_space(mask_house);
    free_space(mask_tau);
}

void shuffle_block_mask(size_t m, size_t n, size_t block_size, int shuffle_times, double* x, bool mask, bool is_left, unsigned int seed){
    // set random seed
    srand(seed);

    struct timeval start, finish;

    vector<unsigned int> shuffle_seed, mask_seed;
    for(int st=0; st<shuffle_times; st++) {shuffle_seed.push_back(rand()); mask_seed.push_back(rand());}
    if(!mask){
        std::reverse(shuffle_seed.begin(), shuffle_seed.end());
        std::reverse(mask_seed.begin(), mask_seed.end());
    }
    // allocate space for masks
    double *mask_house; get_space(&mask_house, block_size*block_size, false, "None");
    double *mask_tau; get_space(&mask_tau, block_size-1, false, "None");
    
    for(int st=0; st<shuffle_times; st++){
        // Shuffle X
        // gettimeofday(&start, NULL);
        if(mask) shuffle_matrix(m, n, x, shuffle_seed[st], is_left, true);
        // gettimeofday(&finish, NULL); cout << "Shuffle Matrix Cost " << time_diff(start, finish) << endl;
        // gettimeofday(&start, NULL);
        srand(mask_seed[st]);
        if(is_left){
            for(size_t i=0; i<m; i+=block_size){
                size_t real_block_size = min(block_size, m-i);
                generate_random_orthogonal_matrix(CblasRowMajor, real_block_size, mask_house, mask_tau, mask_seed[st]);
                LAPACKE_dormqr(CblasRowMajor, 'L', mask?'N':'T', real_block_size, n, real_block_size, mask_house, real_block_size, mask_tau, x+i*n, n);
            }
        }
        else{
            for(size_t i=0; i<n; i+=block_size){
                size_t real_block_size = min(block_size, n-i);
                generate_random_orthogonal_matrix(CblasRowMajor, real_block_size, mask_house, mask_tau, mask_seed[st]);
                LAPACKE_dormqr(CblasRowMajor, 'R', mask?'N':'T', m, real_block_size, real_block_size, mask_house, real_block_size, mask_tau, x+i, n);
            }
        }
        // gettimeofday(&finish, NULL); cout << "Mask Matrix Cost " << time_diff(start, finish) << endl;
        if(!mask) shuffle_matrix(m, n, x, shuffle_seed[st], is_left, false);
    }
    free_space(mask_house);
    free_space(mask_tau);
}

void block_shift_mask(size_t m, size_t n, size_t mask_times, double* x, bool mask, bool is_left, unsigned int seed){
    // Set random seed
    srand(seed);
    
    int optimized_threshold = OPT_THRESHOLD;

    assert(BLOACK_MASK_METHOD == 0 || BLOACK_MASK_METHOD == 1);

    size_t k = is_left ? m : n;
    
    // If k <= 10000, directly run givens masks
    if(k <= optimized_threshold){
        if(BLOACK_MASK_METHOD == 1) givens_mask_cache_optimized_parallel_with_simd(m, n, mask_times, x, mask, is_left, seed);
        if(BLOACK_MASK_METHOD == 0) block_mask(m, n, k, x, mask, is_left, seed);
    }
    // If k > 10000, run more efficient solutions
    else{
        struct SquareMaskConvert smc = {.k = k, .remain=0};        
        find_best_convert(&smc);
        cout << "BlockGivensShift Converting " << smc.k << " = " << smc.m << " * " << smc.n << " + " << smc.remain << endl;
        size_t virtual_matrix_m = smc.m, virtual_matrix_n = smc.n, remain_matrix_size = smc.remain;
        bool remain = remain_matrix_size > 0;
        
        double *x_block; get_space(&x_block, k-remain_matrix_size, false, "None");
        double *x_block_shift; get_space(&x_block_shift, k-remain_matrix_size, false, "None");
        double *x_remain_block;  if(remain) get_space(&x_remain_block, remain_matrix_size + virtual_matrix_m, false, "None");
        
        if(mask){
            for(size_t i=0; i<(is_left?n:m); i++){
                // Copy data from x
                cblas_dcopy(k-remain_matrix_size, x+i*(is_left?1:n), is_left?n:1, x_block, 1);
                // Handle Remain
                if(remain){
                    cblas_dcopy(virtual_matrix_m, x_block, virtual_matrix_n, x_remain_block, 1);
                    cblas_dcopy(
                        remain_matrix_size,
                        x+i*(is_left?1:n)+(k-remain_matrix_size)*(is_left?n:1), is_left?n:1,
                        x_remain_block+virtual_matrix_m, 1
                    );
                    if(BLOACK_MASK_METHOD == 1) 
                    givens_mask_cache_optimized_parallel_with_simd(1, remain_matrix_size + virtual_matrix_m, mask_times, x_remain_block, true, false, seed+1000);
                    if(BLOACK_MASK_METHOD == 0) 
                    block_mask(1, remain_matrix_size + virtual_matrix_m, remain_matrix_size + virtual_matrix_m, x_remain_block, true, false, seed+1000);
                    cblas_dcopy(virtual_matrix_m, x_remain_block, 1, x_block, virtual_matrix_n);
                }
                // Block Mask (1)
                if(BLOACK_MASK_METHOD == 1) 
                givens_mask_cache_optimized_parallel_with_simd(virtual_matrix_m, virtual_matrix_n, mask_times, x_block, true, true, seed+2000);
                if(BLOACK_MASK_METHOD == 0) 
                block_mask(virtual_matrix_m, virtual_matrix_n, virtual_matrix_m, x_block, true, true, seed+2000);
                // Shift Matrix
                // #pragma omp parallel for
                for(int j=0; j<virtual_matrix_m; j++)
                for(int k=0; k<virtual_matrix_n; k++){
                    int shift = (k-j) % ((int)virtual_matrix_n);
                    if(shift < 0) shift += (int)virtual_matrix_n;
                    x_block_shift[j*virtual_matrix_n+k] = x_block[j*virtual_matrix_n + shift];
                }
                // Block Mask (2)
                if(BLOACK_MASK_METHOD == 1){
                    givens_mask_cache_optimized_parallel_with_simd(virtual_matrix_m, virtual_matrix_n, mask_times, x_block_shift, true, true, seed+3000);
                    givens_mask_cache_optimized_parallel_with_simd(virtual_matrix_m, virtual_matrix_n, mask_times, x_block_shift, true, false, seed+4000);
                }
                if(BLOACK_MASK_METHOD == 0){
                    block_mask(virtual_matrix_m, virtual_matrix_n, virtual_matrix_m, x_block_shift, true, true, seed+3000);
                    block_mask(virtual_matrix_m, virtual_matrix_n, virtual_matrix_n, x_block_shift, true, false, seed+4000);
                }
                // Handle Remain Again to have enough randomness
                if(remain){
                    cblas_dcopy(virtual_matrix_m, x_block_shift, virtual_matrix_n, x_remain_block, 1);
                    // print_matrix("remain_matrix 2", 1, remain_matrix_size + virtual_matrix_m, x_remain_block);
                    if(BLOACK_MASK_METHOD == 1) 
                    givens_mask_cache_optimized_parallel_with_simd(1, remain_matrix_size + virtual_matrix_m, mask_times, x_remain_block, true, false, seed+5000);
                    if(BLOACK_MASK_METHOD == 0)
                    block_mask(1, remain_matrix_size + virtual_matrix_m, remain_matrix_size + virtual_matrix_m, x_remain_block, true, false, seed+5000);
                    cblas_dcopy(virtual_matrix_m, x_remain_block, 1, x_block_shift, virtual_matrix_n);
                    cblas_dcopy(remain_matrix_size, x_remain_block+virtual_matrix_m, 1, x+i*(is_left?1:n)+(k-remain_matrix_size)*(is_left?n:1), is_left?n:1);
                }
                // Put data back to x
                cblas_dcopy(k-remain_matrix_size, x_block_shift, 1, x+i*(is_left?1:n), is_left?n:1);
            }
        }
        else{
            for(size_t i=0; i<(is_left?n:m); i++){
                // Copy data from masked x
                cblas_dcopy(k-remain_matrix_size, x+i*(is_left?1:n), is_left?n:1, x_block_shift, 1);
                // Handle Remain
                if(remain){
                    cblas_dcopy(
                        remain_matrix_size,
                        x + i*(is_left?1:n) + (k-remain_matrix_size) * (is_left?n:1), is_left?n:1,
                        x_remain_block+virtual_matrix_m, 1
                    );
                    cblas_dcopy(virtual_matrix_m, x_block_shift, virtual_matrix_n, x_remain_block, 1);
                    if(BLOACK_MASK_METHOD == 1) 
                    givens_mask_cache_optimized_parallel_with_simd(1, remain_matrix_size + virtual_matrix_m, mask_times, x_remain_block, false, false, seed+5000);
                    if(BLOACK_MASK_METHOD == 0)
                    block_mask(1, remain_matrix_size + virtual_matrix_m, remain_matrix_size + virtual_matrix_m, x_remain_block, false, false, seed+5000);
                    cblas_dcopy(virtual_matrix_m, x_remain_block, 1, x_block_shift, virtual_matrix_n);
                }
                // Remove Block Mask (2)
                if(BLOACK_MASK_METHOD == 1){
                    givens_mask_cache_optimized_parallel_with_simd(virtual_matrix_m, virtual_matrix_n, mask_times, x_block_shift, false, true, seed+3000);
                    givens_mask_cache_optimized_parallel_with_simd(virtual_matrix_m, virtual_matrix_n, mask_times, x_block_shift, false, false, seed+4000);
                }
                if(BLOACK_MASK_METHOD == 0){
                    block_mask(virtual_matrix_m, virtual_matrix_n, virtual_matrix_m, x_block_shift, false, true, seed+3000);
                    block_mask(virtual_matrix_m, virtual_matrix_n, virtual_matrix_n, x_block_shift, false, false, seed+4000);
                }
                // Shift back
                #pragma omp parallel for
                for(int j=0; j<virtual_matrix_m; j++)
                for(int k=0; k<virtual_matrix_n; k++)
                    x_block[j*virtual_matrix_n+k] = x_block_shift[j*virtual_matrix_n + (k+j) % ((int)virtual_matrix_n)];
                // Remove Block Mask (1)
                if(BLOACK_MASK_METHOD == 1)
                givens_mask_cache_optimized_parallel_with_simd(virtual_matrix_m, virtual_matrix_n, mask_times, x_block, false, true, seed+2000);
                if(BLOACK_MASK_METHOD == 0)
                block_mask(virtual_matrix_m, virtual_matrix_n, virtual_matrix_m, x_block, false, true, seed+2000);
                // Handle Remain
                if(remain){
                    cblas_dcopy(virtual_matrix_m, x_block, virtual_matrix_n, x_remain_block, 1);
                    if(BLOACK_MASK_METHOD == 1) 
                    givens_mask_cache_optimized_parallel_with_simd(1, remain_matrix_size + virtual_matrix_m, mask_times, x_remain_block, false, false, seed+1000);
                    if(BLOACK_MASK_METHOD == 0)
                     block_mask(1, remain_matrix_size + virtual_matrix_m, remain_matrix_size + virtual_matrix_m, x_remain_block, false, false, seed+1000);
                    cblas_dcopy(virtual_matrix_m, x_remain_block, 1, x_block, virtual_matrix_n);
                    cblas_dcopy(remain_matrix_size, x_remain_block+virtual_matrix_m, 1, x+i*(is_left?1:n)+(k-remain_matrix_size)*(is_left?n:1), is_left?n:1);
                }
                cblas_dcopy(k-remain_matrix_size, x_block, 1, x+i*(is_left?1:n), is_left?n:1);
            }
        }
        free_space(x_block);
        free_space(x_block_shift);
        if(remain) free_space(x_remain_block);
    }
}

void house_mask(size_t m, size_t n, size_t mask_size, double* x, bool mask, bool is_left, unsigned int seed){
   // Set Seed
   srand(seed);
   // Init mask house and tau
   size_t k = is_left ? m : n;
   double* mask_house;
   double* mask_tau = new double[mask_size]();
   char *client_id = getenv("CLIENT_ID");
   string filename = "Client" + to_string(atoi(client_id == NULL ? "9999" : client_id)) + "_mask_house.mat";
   if((k * mask_size) > 5368709120) cout << "Mask using memmap! Might be very slow!" << endl;
   get_space(&mask_house, k * mask_size, (k * mask_size) > 5368709120, filename);  // 40GB

   random_normal_matrix(mask_house, mask_size, k);
   LAPACKE_dgeqrf(CblasColMajor, k, mask_size, mask_house, k, mask_tau);

   // Apply Mask
   char debug = !(is_left^mask)?'N':'T';
   // cout << "Debug Error Mask m=" << m << " n=" << n << " mask=" << mask_size << " is_left=" << is_left << " is_mask=" << mask << " k=" << k << " T=" << debug << endl;
   LAPACKE_dormqr(CblasColMajor, is_left?'L':'R', !(is_left^mask)?'N':'T', m, n, mask_size, mask_house, k, mask_tau, x, m);
   // LAPACKE_dormqr(CblasColMajor, is_left?'L':'R', mask?'N':'T', m, n, mask_size, mask_house, k, mask_tau, x, m);
   // LAPACKE_dormqr(CblasRowMajor, is_left?'L':'R', !(is_left^mask)?'N':'T', m, n, mask_size, mask_house, mask_size, mask_tau, x, n);
   // Release Memory
   delete[] mask_tau;
   free_space(mask_house);
}

void householder_reflector_mask(size_t m, size_t n, size_t mask_size, double* x, bool mask, bool is_left, unsigned int seed){
   // Set Seed
   srand(seed);
   // Init mask house and tau
   size_t k = is_left ? m : n;
   double* mask_house = new double[k * mask_size]();
   double* mask_tau = new double[mask_size]();
   // Create Masks
   for(size_t i=0; i<mask_size; i++){
      mask_house[i*k+i] = 1.0;
      random_uniform_vector(k-i-1, mask_house+i*k+i+1, 1, -1, 1);
      mask_tau[i] = 2 / pow(cblas_dnrm2(k-i, mask_house+i*k+i, 1), 2);
   }
   // Apply Mask
   char L, T;
   if(m >= n){
      L = is_left ? 'L' : 'R';
      T = !(is_left ^ mask) ? 'N' : 'T';
      LAPACKE_dormqr(LAPACK_COL_MAJOR, L, T, m, n, mask_size, mask_house, k, mask_tau, x, m);
   }
   else{
      L = is_left ? 'R' : 'L';
      T = (is_left ^ mask) ? 'N' : 'T';
      LAPACKE_dormqr(LAPACK_COL_MAJOR, L, T, n, m, mask_size, mask_house, k, mask_tau, x, n);
   }
   // Release Memory
   delete[] mask_house;
   delete[] mask_tau;
}

void givens_mask_cache_optimized_parallel_with_simd(size_t m, size_t n, size_t mask_times, double* x, bool mask, bool is_left, unsigned int seed){
    // Set seed
    srand(seed); auto dre = default_random_engine(seed);
   
   //  cout << "m=" << m << " n=" << n << " MT=" << mask_times << " mask=" << mask <<  " isleft=" << is_left << " seed=" << seed << endl;

    // Generate random givens
    size_t mask_size = is_left ? m : n;
    if(mask_size <= 1) {cout << "Not Masking, Since matrix only has one dimension!" << endl; return;}
    vector<vector<size_t>> pair_mask_index;
    for(size_t i=0; i<mask_times; i++){
        // cout << "Debug i=" << i << endl;
        vector<size_t> tmp_pmi;
        for(size_t j=0; j<mask_size; j++) tmp_pmi.push_back(j);
        // Shuffle Index
        shuffle(tmp_pmi.begin(), tmp_pmi.end(), dre);
        // Extra process when m/n is odd
        if(mask_size % 2 == 1) {
            tmp_pmi.push_back(rand() % mask_size);
            while(tmp_pmi[mask_size] == tmp_pmi[mask_size-1]) tmp_pmi[mask_size] = rand() % mask_size;
        }
        // Reverse index if not mask
        if(!mask) reverse(tmp_pmi.begin(), tmp_pmi.end());
        // Reorder index (group by 2)
        for(size_t j=0; j<((mask_size%2)==0?mask_size:(mask_size+1)); j+=2) 
        if(tmp_pmi[j] > tmp_pmi[j+1]) swap(tmp_pmi[j], tmp_pmi[j+1]);

        // for(int j=0; j<((mask_size%2)==0?mask_size:(mask_size+1)); j++) cout << tmp_pmi[j] << " "; cout << endl;

        // Cache index to pair_mask_index
        pair_mask_index.push_back(tmp_pmi);
    }
    // Reverse group index if not mask
    if(!mask) reverse(pair_mask_index.begin(), pair_mask_index.end());
    
    size_t one_pass_size = ((mask_size % 2) == 0 ? mask_size : (mask_size + 1)) / 2;
    vector<vector<double>> c, s;
    double cs_root;
    for(size_t i=0; i<mask_times; i++){
        // cout << "Debug i=" << i << endl;
        vector<double> ci(one_pass_size, 0), si(one_pass_size, 0);
        for(size_t j=0; j<one_pass_size; j++){
            ci[j] = random_value(-1, 1); si[j] = random_value(-1, 1);
            cs_root = pow(pow(ci[j], 2)+pow(si[j], 2), 0.5);
            ci[j] /= cs_root; si[j] /= cs_root;
            // cout << ci[j] << " " << si[j];
        }
        // cout << endl;
        if(!mask) {reverse(ci.begin(), ci.end()); reverse(si.begin(), si.end());}
        c.push_back(ci);
        s.push_back(si);
    }
    if(!mask) {reverse(c.begin(), c.end()); reverse(s.begin(), s.end());}
    // cout << "Set C/S Done size=" << mask_times * one_pass_size << endl;

    if(is_left){
        #pragma omp parallel for
        for(int k=0; k<(n-(n%4)); k+=4){
            double *x_block = new double[4*m]{};
            for(int i=0; i<m; i++) copy(x+i*n+k, x+i*n+k+4, x_block+i*4);
            for(int i=0; i<mask_times; i++)
            for(int j=0; j<one_pass_size; j++){
                int x1 = pair_mask_index[i][j*2] * 4;
                int x2 = pair_mask_index[i][j*2+1] * 4;

               //  double tmp_c, tmp_s;
               //  tmp_c = random_value(-1, 1);
               //  tmp_s = pow(1 - pow(tmp_c, 2), 0.5);

                __m256d cc = _mm256_set1_pd(c[i][j]);
                __m256d ss = _mm256_set1_pd((mask?1:-1) * s[i][j]);
                __m256d _ss = _mm256_set1_pd((mask?-1:1) * s[i][j]);

               //  __m256d cc = _mm256_set1_pd(tmp_c);
               //  __m256d ss = _mm256_set1_pd((mask?1:-1) * tmp_s);
               //  __m256d _ss = _mm256_set1_pd((mask?-1:1) * tmp_s);

                __m256d xx = _mm256_loadu_pd(x_block + x1);
                __m256d yy = _mm256_loadu_pd(x_block + x2);
                __m256d new_x = _mm256_add_pd(_mm256_mul_pd(xx, cc), _mm256_mul_pd(yy, ss));
                __m256d new_y = _mm256_add_pd(_mm256_mul_pd(xx, _ss), _mm256_mul_pd(yy, cc));
                _mm256_storeu_pd(x_block + x1, new_x);
                _mm256_storeu_pd(x_block + x2, new_y);
            }
            for(int i=0; i<m; i++) copy(x_block+i*4, x_block+i*4+4, x + i*n + k);
            delete[] x_block;
        }
        // Unrolling to handle the rest
        if((n%4) != 0){
            int left_size = n % 4, left_pointer = n - left_size;
            double *x_block = new double[left_size*m]{};
            // cout << "Debug correct left_pointer=" << left_pointer << endl;
            for(int i=0; i<m; i++) copy(x+i*n+left_pointer, x+i*n+left_pointer+left_size, x_block+i*left_size);
            for(int i=0; i<mask_times; i++)
            for(int j=0; j<one_pass_size; j++){
                int x1 = pair_mask_index[i][j*2] * left_size;
                int x2 = pair_mask_index[i][j*2+1] * left_size;
                #pragma omp simd
                for(int kk=0; kk<left_size; kk++){
                    double new_x = c[i][j] * x_block[x1+kk] + (mask?1:-1) * s[i][j] * x_block[x2+kk];
                    x_block[x2+kk] = (mask?-1:1) * s[i][j] * x_block[x1+kk] + c[i][j] * x_block[x2+kk];
                    x_block[x1+kk] = new_x;
                }
            }
            for(int i=0; i<m; i++) copy(x_block+i*left_size, x_block+i*left_size+left_size, x + i*n + left_pointer);
        }
    }
    else{
        #pragma omp parallel for
        for(int k=0; k<(m-(m%4)); k+=4){
            double *x_block = new double[4*n]{};
            for(int i=0; i<4; i++) cblas_dcopy(n, x+(k+i)*n, 1, x_block+i, 4);
            for(int i=0; i<mask_times; i++)
            for(int j=0; j<one_pass_size; j++){
                int x1 = pair_mask_index[i][j*2] * 4;
                int x2 = pair_mask_index[i][j*2+1] * 4;
                __m256d cc = _mm256_set1_pd(c[i][j]);
                __m256d ss = _mm256_set1_pd((mask?1:-1) * s[i][j]);
                __m256d _ss = _mm256_set1_pd((mask?-1:1) * s[i][j]);
                __m256d xx = _mm256_loadu_pd(x_block + x1);
                __m256d yy = _mm256_loadu_pd(x_block + x2);
                __m256d new_x = _mm256_add_pd(_mm256_mul_pd(xx, cc), _mm256_mul_pd(yy, ss));
                __m256d new_y = _mm256_add_pd(_mm256_mul_pd(xx, _ss), _mm256_mul_pd(yy, cc));
                _mm256_storeu_pd(x_block + x1, new_x);
                _mm256_storeu_pd(x_block + x2, new_y);
            }
            for(int i=0; i<4; i++) cblas_dcopy(n, x_block+i, 4, x+(k+i)*n, 1);
            delete[] x_block;
        }
        // Unrolling to handle the rest
        if((m%4) != 0){
            int left_size = m % 4, left_pointer = m - left_size;
            double *x_block = new double[left_size*n]{};
            // cout << "Debug correct left_pointer=" << left_pointer << endl;
            for(int i=0; i<left_size; i++) cblas_dcopy(n, x+(left_pointer+i)*n, 1, x_block+i, left_size);
            for(int i=0; i<mask_times; i++)
            for(int j=0; j<one_pass_size; j++){
                int x1 = pair_mask_index[i][j*2] * left_size;
                int x2 = pair_mask_index[i][j*2+1] * left_size;
                #pragma omp simd
                for(int kk=0; kk<left_size; kk++){
                    double new_x = c[i][j] * x_block[x1+kk] + (mask?1:-1) * s[i][j] * x_block[x2+kk];
                    x_block[x2+kk] = (mask?-1:1) * s[i][j] * x_block[x1+kk] + c[i][j] * x_block[x2+kk];
                    x_block[x1+kk] = new_x;
                }
            }
            for(int i=0; i<left_size; i++) cblas_dcopy(n, x_block+i, left_size, x+(left_pointer+i)*n, 1);
        }
    }
}

void bidiagonal_bisection_twisted_svd(int k, double *alpha, double *beta, double **bidiagonal_u, double **bidiagonal_v, double **sigma, int* sigma_count){

   // (1) Bisection singular values
   *sigma = new double[k]();
   double sigma_upper_bound = pow(cblas_dnrm2(k, alpha, 1), 2) + pow(cblas_dnrm2(k-1, beta, 1), 2);
   bisection_singular_values(k, alpha, beta, 0, sigma_upper_bound, 1e-20, k, *sigma, sigma_count);
   
   // (2) Bisection singular values
   *bidiagonal_u = new double[k * (*sigma_count)]();
   *bidiagonal_v = new double[k * (*sigma_count)]();
   two_side_singular_vectors_twisted_factorization(k, (*sigma_count), *sigma, alpha, beta, *bidiagonal_u, *bidiagonal_v);

   // (3) Adjust the sign of v
   int util_pointer;
   double signer;
   for(int i=0; i<(*sigma_count); i++){
      util_pointer = cblas_idamax(k, (*bidiagonal_v)+i*k, 1);
      if(util_pointer == 0) 
         signer = (*bidiagonal_u)[util_pointer*(*sigma_count)+i] * alpha[util_pointer] / (*sigma)[i];
      else 
         signer = ((*bidiagonal_u)[(util_pointer-1)*(*sigma_count)+i] * beta[util_pointer-1] + (*bidiagonal_u)[util_pointer*(*sigma_count)+i] * alpha[util_pointer]) / (*sigma)[i];
      if((signer * (*bidiagonal_v)[i*k+util_pointer]) < 0){
         cblas_dscal(k, -1, (*bidiagonal_v)+i*k, 1);
      }
   }
}

void bidiagonal_bisection_twisted_svd(int k, double *alpha, double *beta, double **bidiagonal_u, double **bidiagonal_v, double **sigma, int* sigma_count, bool is_memmap){
   
   // (1) Bisection singular values
   *sigma = new double[k]();
   double sigma_upper_bound = pow(cblas_dnrm2(k, alpha, 1), 2) + pow(cblas_dnrm2(k-1, beta, 1), 2);
   bisection_singular_values(k, alpha, beta, 0, sigma_upper_bound, 1e-20, k, *sigma, sigma_count);
   
   // (2) Bisection singular values
   get_space(bidiagonal_u, k * *(sigma_count), is_memmap, "bidiagonal_u.mat");
   get_space(bidiagonal_v, k * *(sigma_count), is_memmap, "bidiagonal_v.mat");
   two_side_singular_vectors_twisted_factorization(k, (*sigma_count), *sigma, alpha, beta, *bidiagonal_u, *bidiagonal_v);

   // (3) Adjust the sign of v
   int util_pointer;
   double signer;
   for(int i=0; i<(*sigma_count); i++){
      util_pointer = cblas_idamax(k, (*bidiagonal_v)+i*k, 1);
      if(util_pointer == 0) 
         signer = (*bidiagonal_u)[util_pointer*(*sigma_count)+i] * alpha[util_pointer] / (*sigma)[i];
      else 
         signer = ((*bidiagonal_u)[(util_pointer-1)*(*sigma_count)+i] * beta[util_pointer-1] + (*bidiagonal_u)[util_pointer*(*sigma_count)+i] * alpha[util_pointer]) / (*sigma)[i];
      if((signer * (*bidiagonal_v)[i*k+util_pointer]) < 0){
         cblas_dscal(k, -1, (*bidiagonal_v)+i*k, 1);
      }
   }
}

void bidiagonal_bisection_twisted_svd(int k, double *alpha, double *beta, double *bidiagonal_u, double *bidiagonal_v, double *sigma, int top_k){
   
   // (1) Bisection singular values
   int sigma_count = 0;
   double sigma_upper_bound = pow(cblas_dnrm2(k, alpha, 1), 2) + pow(cblas_dnrm2(k-1, beta, 1), 2);
   bisection_singular_values(k, alpha, beta, 0, sigma_upper_bound, 1e-20, top_k, sigma, &sigma_count);
   assert(sigma_count >= top_k);
   
   // (2) Bisection singular values
   two_side_singular_vectors_twisted_factorization(k, top_k, sigma, alpha, beta, bidiagonal_u, bidiagonal_v);

   // (3) Adjust the sign of v
   int util_pointer;
   double signer;
   for(int i=0; i<top_k; i++){
      util_pointer = cblas_idamax(k, bidiagonal_v+i*k, 1);
      if(util_pointer == 0) 
         signer = bidiagonal_u[util_pointer*top_k+i] * alpha[util_pointer] / sigma[i];
      else 
         signer = (bidiagonal_u[(util_pointer-1)*top_k+i] * beta[util_pointer-1] + bidiagonal_u[util_pointer*top_k+i] * alpha[util_pointer]) / sigma[i];
      if((signer * bidiagonal_v[i*k+util_pointer]) < 0){
         cblas_dscal(k, -1, bidiagonal_v+i*k, 1);
      }
   }

   // (4) Check None
   for(int i=0; i<top_k; i++){
      if(abs(sigma[i]-alpha[i]) < 1e-15){
        bidiagonal_u[i+i*top_k] = 1.0;
        for(int j=i+1; j<top_k; j++) bidiagonal_u[i*top_k+j] = 0.0;
        for(int j=i+1; j<k; j++) bidiagonal_u[j*top_k+i] = 0.0;
        for(int j=0; j<i; j++) bidiagonal_v[i*k+j] = 0.0;
        bidiagonal_v[i*k+i] = 1.0;
        for(int j=i+1; j<k; j++) bidiagonal_v[i*k+j] = 0.0;
      }
   }
}

void mmap_drot(size_t n, double *x, size_t incX, double *y, size_t incY, double c, double s){
   #pragma omp parallel for
   for(size_t i=0; i<n; i++){
      double _tmp_x = x[i*incX] * c + y[i*incY] * s; 
      double _tmp_y = x[i*incX] * -s + y[i*incY] * c;
      x[i*incX] = _tmp_x;
      y[i*incY] = _tmp_y;
   }
}

