#include <mkl.h>
#include <stack>
#include <cmath>
#include <random>
#include <immintrin.h>
#include "utils.hpp"

#ifndef BBTSVD
#define BBTSVD

struct BISECTION_SV{
   double low;
   int nc_low;
   double high;
   int nc_high;
};

struct SquareMaskConvert{
    size_t k, m, n, remain;
};

int negative_count(int n, double* x_diagonal, double* x_upper_diagonal, double mu);
void bisection_singular_values(int n, double* x_diagonal, double* x_upper_diagonal, double a, double b, double tol, int top_k, double* sigma, int* sigma_count);
void ldl_decomposition(int n, double* x_diagonal, double* x_upper_diagonal, double* l, double* d);
void inverse_iteration(int n, int k, double* l, double* u, double* v, int ldv);
void dstqds(int n, double* l, double* d, double tau, double* l_plus, double* d_plus, double* s);
void dpqds(int n, double *l, double* d, double tau, double* u_plus, double* r_plus, double* p);
void one_pair_singular_vector_twisted_factorization(int n, int omege_count, double omega, double* l, double* d, double* x_diagonal, double* x_upper_diagonal, double* v, double* u);
void two_side_singular_vectors_twisted_factorization(int n, int omege_count, double* omege, double* x_diagonal, double* x_upper_diagonal, double* u, double* v);
void generate_house(int n, double* x, int incX, double* beta);
void house_mask(size_t m, size_t n, size_t mask_size, double* x, bool mask, bool is_left, unsigned int seed);
void givens_mask_cache_optimized_parallel_with_simd(size_t m, size_t n, size_t mask_times, double* x, bool mask, bool is_left, unsigned int seed);
void bidiagonal_bisection_twisted_svd(int k, double *alpha, double *beta, double **bidiagonal_u, double **bidiagonal_v, double **sigma, int* sigma_count);
void bidiagonal_bisection_twisted_svd(int k, double *alpha, double *beta, double **bidiagonal_u, double **bidiagonal_v, double **sigma, int* sigma_count, bool is_memmap);
void bidiagonal_bisection_twisted_svd(int k, double *alpha, double *beta, double *bidiagonal_u, double *bidiagonal_v, double *sigma, int top_k);
void mmap_drot(size_t n, double *x, size_t incX, double *y, size_t incY, double c, double s);
void householder_reflector_mask(size_t m, size_t n, size_t mask_size, double* x, bool mask, bool is_left, unsigned int seed);

void find_best_convert(struct SquareMaskConvert* smc);
void naive_givens_mask_parallel(size_t m, size_t n, size_t mask_size, double* x, bool mask, bool is_left, unsigned int seed);
void givens_mask_cache_optimized_parallel(size_t m, size_t n, size_t mask_times, double* x, bool mask, bool is_left, unsigned int seed);
void generate_random_orthogonal_matrix(int matrix_layout, size_t m, double* house, double* tau, unsigned int seed);
void generate_random_orthogonal_matrix(size_t m, size_t n, double* x, unsigned int seed);
void shuffle_matrix(size_t m, size_t n, double *x, unsigned int seed, bool is_left, bool unshuffle);
void block_mask(size_t m, size_t n, size_t block_size, double* x, bool mask, bool is_left, unsigned int seed);
void shuffle_block_mask(size_t m, size_t n, size_t block_size, int shuffle_times, double* x, bool mask, bool is_left, unsigned int seed);
void block_shift_mask(size_t m, size_t n, size_t mask_times, double* x, bool mask, bool is_left, unsigned int seed);

#endif