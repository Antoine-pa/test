#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#define ALIGN 64
#define MR   4
#define NR   8

static inline int imin(int a, int b) { return a < b ? a : b; }

/*
 * 4×8 micro-kernel — the hot inner loop.
 * Computes  C[i..i+3][j..j+7]  +=  A[i..i+3][ks..ke) · B[ks..ke)[j..j+7]
 *
 * 8 YMM regs hold the 4×8 C tile across the entire k-loop →
 *   ONE load at entry, ONE store at exit.
 * Per k iteration: 4 broadcasts (port 5) + 2 B loads (port 2/3) + 8 FMAs (port 0/1)
 *   → bottleneck = 4 cycles, throughput = 8 FMA / 4 cyc = 2 FMA/cyc = peak.
 */
static inline void __attribute__((always_inline))
kern_4x8(const double * restrict A, const double * restrict B,
         double * restrict C, int ld,
         int i, int j, int ks, int ke)
{
	double       *c0 = C + (size_t)i * ld + j;
	double       *c1 = c0 + ld;
	double       *c2 = c1 + ld;
	double       *c3 = c2 + ld;
	const double *a0 = A + (size_t)i * ld;
	const double *a1 = a0 + ld;
	const double *a2 = a1 + ld;
	const double *a3 = a2 + ld;

	__m256d C00 = _mm256_loadu_pd(c0);
	__m256d C01 = _mm256_loadu_pd(c0 + 4);
	__m256d C10 = _mm256_loadu_pd(c1);
	__m256d C11 = _mm256_loadu_pd(c1 + 4);
	__m256d C20 = _mm256_loadu_pd(c2);
	__m256d C21 = _mm256_loadu_pd(c2 + 4);
	__m256d C30 = _mm256_loadu_pd(c3);
	__m256d C31 = _mm256_loadu_pd(c3 + 4);

	for (int k = ks; k < ke; k++) {
		const double *bk = B + (size_t)k * ld + j;
		__m256d b0 = _mm256_loadu_pd(bk);
		__m256d b1 = _mm256_loadu_pd(bk + 4);

		__m256d a;
		a = _mm256_broadcast_sd(a0 + k);
		C00 = _mm256_fmadd_pd(a, b0, C00);
		C01 = _mm256_fmadd_pd(a, b1, C01);

		a = _mm256_broadcast_sd(a1 + k);
		C10 = _mm256_fmadd_pd(a, b0, C10);
		C11 = _mm256_fmadd_pd(a, b1, C11);

		a = _mm256_broadcast_sd(a2 + k);
		C20 = _mm256_fmadd_pd(a, b0, C20);
		C21 = _mm256_fmadd_pd(a, b1, C21);

		a = _mm256_broadcast_sd(a3 + k);
		C30 = _mm256_fmadd_pd(a, b0, C30);
		C31 = _mm256_fmadd_pd(a, b1, C31);
	}

	_mm256_storeu_pd(c0,     C00);
	_mm256_storeu_pd(c0 + 4, C01);
	_mm256_storeu_pd(c1,     C10);
	_mm256_storeu_pd(c1 + 4, C11);
	_mm256_storeu_pd(c2,     C20);
	_mm256_storeu_pd(c2 + 4, C21);
	_mm256_storeu_pd(c3,     C30);
	_mm256_storeu_pd(c3 + 4, C31);
}

/* scalar fallback for edge tiles that don't fill a full 4×8 */
static inline void
edge_scalar(const double * restrict A, const double * restrict B,
            double * restrict C, int ld,
            int is, int ie, int js, int je, int ks, int ke)
{
	for (int i = is; i < ie; i++) {
		double       * restrict Ci = C + (size_t)i * ld;
		const double * restrict Ai = A + (size_t)i * ld;
		for (int k = ks; k < ke; k++) {
			double aik = Ai[k];
			const double * restrict Bk = B + (size_t)k * ld;
			#pragma GCC ivdep
			for (int j = js; j < je; j++)
				Ci[j] += aik * Bk[j];
		}
	}
}

__attribute__((hot))
void matmul(double * restrict A, double * restrict B, double * restrict C,
            int size, int bs)
{
	memset(C, 0, (size_t)size * size * sizeof(double));

	#pragma omp parallel for schedule(static) collapse(2)
	for (int ii = 0; ii < size; ii += bs) {
		for (int jj = 0; jj < size; jj += bs) {
			int ie = imin(ii + bs, size);
			int je = imin(jj + bs, size);
			int ib = ii + ((ie - ii) / MR) * MR;
			int jb = jj + ((je - jj) / NR) * NR;

			for (int kk = 0; kk < size; kk += bs) {
				int ke = imin(kk + bs, size);

				for (int i = ii; i < ib; i += MR) {
					int j;
					for (j = jj; j < jb; j += NR)
						kern_4x8(A, B, C, size, i, j, kk, ke);
					if (j < je)
						edge_scalar(A, B, C, size, i, i + MR, j, je, kk, ke);
				}
				if (ib < ie)
					edge_scalar(A, B, C, size, ib, ie, jj, je, kk, ke);
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
