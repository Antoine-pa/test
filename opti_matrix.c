#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define ALIGN 64

static inline int imin(int a, int b) { return a < b ? a : b; }

__attribute__((hot))
void matmul(double * restrict A, double * restrict B, double * restrict C,
            int size, int bs)
{
	memset(C, 0, (size_t)size * size * sizeof(double));

	/*
	 * Tiling ii×jj + loop order i-k-j inside each tile.
	 * Inner j-loop = DAXPY: C_row[j] += a_ik * B_row[j]
	 *   → both C and B contiguous in memory → AVX2/FMA auto-vectorises.
	 * collapse(2) on (ii,jj) gives (size/bs)² independent tasks
	 *   → much better load balance on 8 HW threads.
	 * No race: each (ii,jj) tile owns its C region; kk is sequential.
	 */
	#pragma omp parallel for schedule(static) collapse(2)
	for (int ii = 0; ii < size; ii += bs) {
		for (int jj = 0; jj < size; jj += bs) {
			int i_end = imin(ii + bs, size);
			int j_end = imin(jj + bs, size);
			for (int kk = 0; kk < size; kk += bs) {
				int k_end = imin(kk + bs, size);
				for (int i = ii; i < i_end; i++) {
					double * restrict Ci = C + (size_t)i * size;
					const double * restrict Ai = A + (size_t)i * size;
					for (int k = kk; k < k_end; k++) {
						const double a_ik = Ai[k];
						const double * restrict Bk = B + (size_t)k * size;
						#pragma GCC ivdep
						for (int j = jj; j < j_end; j++) {
							Ci[j] += a_ik * Bk[j];
						}
					}
				}
			}
		}
	}
}

void initMatrix(double * restrict A, double * restrict B, int size)
{
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++) {
			A[i * size + j] = 1;
			B[i * size + j] = 1;
		}
}

void checkMatrix(double * restrict C, int size)
{
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			if (C[i * size + j] != size) {
				printf("Error : value of result[%d][%d] is %f instead of %d !\n",
				       i, j, C[i * size + j], size);
				return;
			}
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <size> [block_size] [threads]\n", argv[0]);
		return 1;
	}

	int size = atoi(argv[1]);
	int bs   = (argc >= 3) ? atoi(argv[2]) : 64;
	int nt   = (argc >= 4) ? atoi(argv[3]) : omp_get_max_threads();
	omp_set_num_threads(nt);

	size_t n     = (size_t)size * size;
	size_t bytes = ((n * sizeof(double) + ALIGN - 1) / ALIGN) * ALIGN;

	double *A, *B, *C;
	posix_memalign((void **)&A, ALIGN, bytes);
	posix_memalign((void **)&B, ALIGN, bytes);
	posix_memalign((void **)&C, ALIGN, bytes);

	initMatrix(A, B, size);

	double t0 = omp_get_wtime();
	matmul(A, B, C, size, bs);
	double t1 = omp_get_wtime();

	checkMatrix(C, size);

	double elapsed = t1 - t0;
	double gflops  = 2.0 * (double)size * size * size / (elapsed * 1e9);
	fprintf(stderr,
	        "Size=%d  Block=%d  Threads=%d  Time=%.4fs  %.2f GFLOPS\n",
	        size, bs, nt, elapsed, gflops);

	free(A); free(B); free(C);
	return 0;
}
