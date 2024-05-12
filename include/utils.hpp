#include "base.hpp"
#include <string>
#include <mkl.h>
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <memory.h>
#include <unistd.h>
#include <sys/stat.h>
#include <queue>
#include <cmath>
#include <semaphore.h>

#ifndef UTILS
#define UTILS
double abs_error(size_t m, size_t n, double* a, double* b);
double mean_square_error(size_t m, size_t n, double* a, double* b);
double orthogonal_test(double *x, size_t m, size_t n, bool XXT);
double singular_vector_error_rmse(size_t m, size_t n, double* a, double* b);
void explicit_diagonal(double* s, size_t n, double* result);
void explicit_bidiagonal(double* alpha, double * beta, size_t n, double* result);

double reconstruction_error(double* u, double* explicit_s, double* v, double* x, size_t m, size_t n);
double reconstruction_error(double* u, double* explicit_s, double* v, double* x, size_t m, size_t n, size_t k);

double svd_reconstruction_error(double* u, double* s, double* v, double* x, size_t m, size_t n);
double svd_reconstruction_error(double* u, double* s, double* v, double* x, size_t m, size_t n, size_t k);

inline double random_value(double min_value, double max_value){
   return (rand() / (double) RAND_MAX) * (max_value - min_value) + min_value;
}

void deep_copy(double* X, double *Y, size_t m, size_t n);
void deep_copy(char transpose, size_t m, size_t n, double* X, size_t incX, double *Y, size_t incY);
void random_uniform_vector(size_t n, double* x, int incX, double min_value, double max_value);
void random_uniform_matrix(double *x, size_t m, size_t n, double min_value, double max_value);
void random_normal_matrix(double *x, size_t m, size_t n);

void load_memmap_file(double** output, string filename);
void load_memmap_file(double** output, string filename, bool load_to_memory);

void create_memmap_file(void** output, string filename, size_t size);
void get_space(void** x, size_t size, bool is_memmap, string filename);
void get_space(double** x, size_t size, bool is_memmap, string filename);
void free_space(void* x, size_t size, bool is_memmap, string filename);
void free_space(double *x, size_t size, bool is_memmap, string filename);
void free_space(void* x);
void free_space(double* x);
void free_all_space();
void save_to_disk(size_t m, size_t n, double *x, string filename);
void record_space(void*x, size_t size, bool is_memmap, string filename);

double time_diff(struct timeval start, struct timeval finish);
int get_client_id_from_env();
string get_log_path_from_env();

template <class T> void print_matrix(string desc, size_t m, size_t n, T* a) {
   size_t i, j;
   std::cout << std::endl << desc << std::endl;
   for(i = 0; i < m; i++) {
      for(j = 0; j < n; j++)
      std::cout << " " << a[i*n+j];
      std::cout << std::endl;
   }
   std::cout << std::endl;
}

// class SpaceManager{
//    public:
//    size_t size;
//    void *x;
   
//    SpaceManager(size_t size);
//    ~SpaceManager();
   
//    void create_space(bool is_memmap, string filename);
//    void load_space(bool load_to_memory, string filename);
   
//    private:
//    bool is_memmap;
//    string filename;
// };

#endif