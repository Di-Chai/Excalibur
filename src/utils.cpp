#include "utils.hpp"

struct SpaceInfo{
   void* x;
   size_t size;
   bool is_memmap;
   string filename;
};

vector<SpaceInfo> space_array;
sem_t sem_sa;
bool sem_sa_init = false;


double abs_error(size_t m, size_t n, double* a, double* b){
   double error = 0;
   for(size_t i=0; i<m; i++)
   for(size_t j=0; j<n; j++)
   error += abs(a[i*n+j] - b[i*n+j]);
   return error / ((double)m * (double)n);
}

double mean_square_error(size_t m, size_t n, double* a, double* b){
   double error = 0;
   for(size_t i=0; i<m; i++)
   for(size_t j=0; j<n; j++)
   error += pow(a[i*n+j] - b[i*n+j], 2);
   return error / ((double)m * (double)n);
}

double singular_vector_error_rmse(size_t m, size_t n, double* a, double* b){
   double error = 0;
   for(size_t i=0; i<m; i++)
   for(size_t j=0; j<n; j++)
   error += pow(abs(a[i*n+j]) - abs(b[i*n+j]), 2);
   return pow(error / ((double)m * (double)n), 0.5);
}

double orthogonal_test(double *x, size_t m, size_t n, bool XXT){
   size_t xx_size = XXT ? m : n;
   double* xx = (double*) malloc(xx_size * xx_size * 8); memset(xx, 0, xx_size * xx_size * 8);
   if(XXT){
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
         m, m, n, 1.0, x, n, x, n, 0.0, xx, m);
   }
   else{
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
         n, n, m, 1.0, x, n, x, n, 0.0, xx, n);
   }
   double sum = 0.0;
   double sum_max = 0.0;
   for(size_t i=0; i<xx_size; i++)
   for(size_t j=0; j<xx_size; j++){
      if(i != j){
         sum += abs(xx[i*xx_size+j]);
         sum_max = sum_max > abs(xx[i*xx_size+j]) ? sum_max : abs(xx[i*xx_size+j]);
      }
   }
   sum /= (xx_size * xx_size - xx_size);
   std::cout << "Orthogonal Test " << sum << " Max " << sum_max << std::endl;
   free(xx);
   return sum;
}

void explicit_diagonal(double* s, size_t n, double* result){
   for(size_t i=0; i<n; i++) result[i*n+i] = s[i];
}

void explicit_bidiagonal(double* alpha, double * beta, size_t n, double* result){
   explicit_diagonal(alpha, n, result);
   for(size_t i=0; i<(n-1); i++){
      result[i*n+i+1] = beta[i];
   }
}

int get_client_id_from_env(){
   char *client_id = getenv("CLIENT_ID");
   return atoi(client_id == NULL ? "9999" : client_id);
}

string get_log_path_from_env(){
   string log_path = getenv("LOG_PATH");
   return log_path;
}

double reconstruction_error(double* u, double* explicit_s, double* v, double* x, size_t m, size_t n){

   // int k = min(m, n);
   // double* us = new double[m * k]();
   // double* xvt = new double[m * k]();
   // double error = 0.0;
   // // Compute us
   // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, k, 1.0, u, k, explicit_s, k, 0, us, k);
   // // Compute xvt
   // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, n, 1.0, x, n, v, n, 0, xvt, k);
   // std::cout << abs_error(m, k, us, xvt) << std::endl;
   // delete[] us, xvt;
   size_t k = min(m, n);
   int client_id = get_client_id_from_env();
   string log_dir = get_log_path_from_env();
   double* us; get_space(&us, m*k, true, log_dir + "/Client" + to_string(client_id) + "_tmp_re_us.mat");
   double* usv; get_space(&usv, m*n, true, log_dir + "/Client" + to_string(client_id) + "_tmp_re_usv.mat");
   // Compute us
   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, k, 1.0, u, k, explicit_s, k, 0, us, k);
   // Compute usv
   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, us, k, v, n, 0, usv, n);

   // print_matrix("Debug usv", m, n, usv);
   // print_matrix("Debug x", m, n, x);
   
   double error = abs_error(m, n, usv, x);
   free_space(us);
   free_space(usv);

   return error;
}

double reconstruction_error(double* u, double* explicit_s, double* v, double* x, size_t m, size_t n, size_t k){

   // double* us = new double[m * k]();
   // double* xvt = new double[m * k]();
   // double error = 0.0;
   // // Compute us
   // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, k, 1.0, u, k, explicit_s, k, 0, us, k);
   // // Compute xvt
   // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, n, 1.0, x, n, v, n, 0, xvt, k);
   // std::cout << abs_error(m, k, us, xvt) << std::endl;
   // delete[] us, xvt;
   
   int client_id = get_client_id_from_env();
   string log_dir = get_log_path_from_env();
   double* us; get_space(&us, m*k, true, log_dir + "/Client" + to_string(client_id) + "_tmp_re_us.mat");
   double* usv; get_space(&usv, m*n, true, log_dir + "/Client" + to_string(client_id) + "_tmp_re_usv.mat");
   
   // Compute us
   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, k, 1.0, u, k, explicit_s, k, 0, us, k);
   // Compute usv
   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, us, k, v, n, 0, usv, n);

   // print_matrix("USV", m, n, usv);
   // print_matrix("X", m, n, x);
   
   double error = abs_error(m, n, usv, x);
   free_space(us);
   free_space(usv);
   
   return error;
}

double svd_reconstruction_error(double* u, double* s, double* v, double* x, size_t m, size_t n){
   double* explicit_s = new double[min(m, n) * min(m, n)]();
   explicit_diagonal(s, min(m, n), explicit_s);
   double error = reconstruction_error(u, explicit_s, v, x, m, n);
   delete[] explicit_s;
   return error;
}

double svd_reconstruction_error(double* u, double* s, double* v, double* x, size_t m, size_t n, size_t k){
   double* explicit_s = new double[k*k]();
   explicit_diagonal(s, k, explicit_s);
   double error = reconstruction_error(u, explicit_s, v, x, m, n, k);
   delete[] explicit_s;
   return error;
}

void cblas_dcopy_long(size_t n, double *x, size_t incX, double *y, size_t incY){
   size_t overflow_size = 10000000;
   if(n > overflow_size)
   for(size_t i=0; i<n; i+=overflow_size)
   cblas_dcopy(min(overflow_size, n-i), x+i*incX, incX, y+i*incY, incY);
   else cblas_dcopy(n, x, incX, y, incY);
}

void deep_copy(double* X, double *Y, size_t m, size_t n){
   /* Y = X */
   // #pragma omp parallel for
   for(size_t i=0; i<m; i++)
      cblas_dcopy_long(n, X+i*n, 1, Y+i*n, 1);
   // for(size_t i=0; i<m; i++)
   // for(size_t j=0; j<n; j++)
   //    Y[i*n+j] = X[i*n+j];
}

void deep_copy(char transpose, size_t m, size_t n, double* X, size_t incX, double *Y, size_t incY){
   /* Y = X */
   assert(incX >= n);
   // Copy in parallel
   if(transpose == 'T'){
      assert(incY >= m);
      #pragma omp parallel for
      for(size_t i=0; i<m; i++)
         cblas_dcopy(n, X+i*incX, 1, Y+i, incY);
   }
   else{
      assert(incY >= n);
      #pragma omp parallel for
      for(size_t i=0; i<m; i++)
         cblas_dcopy(n, X+i*incX, 1, Y+i*incY, 1);
   }
}

void random_uniform_vector(size_t n, double* x, int incX, double min_value, double max_value){
   for(size_t i=0; i<n; i++) x[i * incX] = random_value(min_value, max_value);
}

void random_uniform_matrix(double *x, size_t m, size_t n, double min_value, double max_value)
{  
   int iseed[4];
	for(size_t i=0; i<m; i++){
      random_uniform_vector(n, x+i*n, 1, min_value, max_value);
   }  
}

void random_normal_matrix(double *x, size_t m, size_t n)
{  
   int iseed[4];
	for(size_t i=0; i<m; i++){
      for(int r=0; r<4; r+=1) iseed[r] = rand() % 4096;
      if(iseed[3] % 2 == 0) iseed[3] = (iseed[3]+1) % 4096;
      // print_matrix("iseed", 1, 4, iseed);
      LAPACKE_dlarnv(3, iseed, n, x+i*n);
   }
}

double time_diff(struct timeval start, struct timeval finish){
   return (finish.tv_sec - start.tv_sec) + (double)(finish.tv_usec - start.tv_usec) / 1000000.0;
}

void load_memmap_file(double** output, string filename){
   int fd = open(filename.c_str(), O_RDWR, S_IRWXU);
   off_t file_size = lseek(fd, 0, SEEK_END);
   cout << "Load array size " << file_size / sizeof(double) << endl;
   *output = (double*) mmap(0, file_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NONBLOCK, fd, 0);
   close(fd);
}

void load_memmap_file(double** output, string filename, bool load_to_memory){
   int fd = open(filename.c_str(), O_RDWR, S_IRWXU);
   off_t file_size = lseek(fd, 0, SEEK_END);
   cout << "Load array size " << file_size / sizeof(double) << endl;
   if(load_to_memory){
      double *file_mmap = (double*) mmap(0, file_size, PROT_READ, MAP_SHARED | MAP_NONBLOCK, fd, 0);
      *output = new double[file_size / sizeof(double)];
      deep_copy(file_mmap, *output, 1, file_size / sizeof(double));
      munmap((void*)file_mmap, file_size);
      cout << "Load to memory" << endl;
   }
   else{
      *output = (double*) mmap(0, file_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NONBLOCK, fd, 0);
   }
   close(fd);
}

void create_memmap_file(void** output, string filename, size_t size){
   int fd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRWXU);
   lseek(fd, size-1, SEEK_SET);
   write(fd, "\0", 1);
   *output = (void*) mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NONBLOCK, fd, 0);
   close(fd);
}

// void create_memmap_file(double** output, string filename, size_t size){
//    int fd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRWXU);
//    lseek(fd, size-1, SEEK_SET);
//    write(fd, "\0", 1);
//    *output = (double*) mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NONBLOCK, fd, 0);
//    close(fd);
// }

void get_space(void** x, size_t size, bool is_memmap, string filename){
   if(!sem_sa_init){
      sem_init(&sem_sa, 0, 1);
      sem_sa_init = true;
   }
   if(is_memmap) create_memmap_file((void**)x, filename, size);
   else *x = calloc(size, 1);
   // else {*x = malloc(size); memset(*x, 0, size);}
   struct SpaceInfo si = {.x=(*x), .size=size, .is_memmap=is_memmap, .filename=filename};
   sem_wait(&sem_sa);
   space_array.push_back(si);
   sem_post(&sem_sa);
   // Debug 
   // cout << "Allocated " << *x << " name " << filename << endl;
}

void get_space(double** x, size_t size, bool is_memmap, string filename){
   get_space((void**)x, size*sizeof(double), is_memmap, filename);
}

void free_space(void* x, size_t size, bool is_memmap, string filename){
   if(is_memmap){
      munmap(x, size);
      remove(filename.c_str());
   }
   else free(x);
}

void free_space(double *x, size_t size, bool is_memmap, string filename){
   free_space((void*)x, size, is_memmap, filename);
}

void free_space(void* x){
   if(!sem_sa_init){
      sem_init(&sem_sa, 0, 1);
      sem_sa_init = true;
   }
   bool released = false;
   sem_wait(&sem_sa);
   for(int i=0; i<space_array.size(); i++){
      if(x == space_array[i].x){
         // Debug
         // cout << "Free " << space_array[i].x << " " << space_array[i].filename << endl;
         free_space(space_array[i].x, space_array[i].size, space_array[i].is_memmap, space_array[i].filename);
         released = true;
         space_array.erase(i + space_array.begin());
         break;
      }
   }
   sem_post(&sem_sa);
   if(!released) cout << "Space " << x << " not released!" << endl;
}

void record_space(void*x, size_t size, bool is_memmap, string filename){
   if(!sem_sa_init){
      sem_init(&sem_sa, 0, 1);
      sem_sa_init = true;
   }
   struct SpaceInfo si = {.x=x, .size=size, .is_memmap=is_memmap, .filename=filename};
   sem_wait(&sem_sa);
   space_array.push_back(si);
   sem_post(&sem_sa);
}

void free_space(double* x){
   free_space((void*)x);
}

void free_all_space(){
   if(!sem_sa_init){
      sem_init(&sem_sa, 0, 1);
      sem_sa_init = true;
   }
   sem_wait(&sem_sa);
   for(int i=space_array.size()-1; i>=0; i--){
      // cout << "Freeing " << space_array[i].x << " name " << space_array[i].filename << endl;
      free_space(space_array[i].x, space_array[i].size, space_array[i].is_memmap, space_array[i].filename);
   }
   sem_post(&sem_sa);
}

void save_to_disk(size_t m, size_t n, double *x, string filename){
   double *x_save;
   create_memmap_file((void**)(&x_save), filename, m * n * sizeof(double));
   deep_copy(x, x_save, m, n);
   cout << "Saved " << filename << endl;
}

// SpaceManager::SpaceManager(size_t size){
//    this->size = size;
// };

// SpaceManager::~SpaceManager(){
//    if(this->is_memmap){
//       munmap(this->x, this->size);
//       remove(this->filename.c_str());
//    }
//    else free(this->x);
// };

// void SpaceManager::create_space(bool is_memmap, string filename){
//    this->is_memmap = is_memmap;
//    this->filename = filename;
//    if(is_memmap) create_memmap_file(&(this->x), filename, this->size);
//    else this->x = malloc(this->size);
// }