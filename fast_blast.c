#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>



void checkMatrix(double *result, int size);
void matmul_blas(double* A1, double* B1, double* result, int size);
void initMatrix(double* A1, double* B1, int size);

#define BS 64   // block size (à tuner selon CPU)


int main(int argc, char **argv){
	int size = atoi(argv[1]);
	
	double *A1 = malloc(size*size*sizeof(double));
	double *B1 = malloc(size*size*sizeof(double));
	double *result = malloc(size*size*sizeof(double));
	
	initMatrix(A1, B1, size);
	matmul_blas(A1, B1, result, size);
	checkMatrix(result, size);
	return 0;
}

void matmul_blas(double* A1, double* B1, double* result, int size)
{
    // C = A * B
    // A, B, C en row-major size x size

    const int n = size;
    const int lda = size;
    const int ldb = size;
    const int ldc = size;

    // C ← 1.0 * A * B + 0.0 * C
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                1.0,
                A1, lda,
                B1, ldb,
                0.0,
                result, ldc);
}


void initMatrix(double* A1, double* B1, int size){
	
	for (int i=0; i<size; i++){
		for (int j=0; j<size; j++){
			A1[i*size+j] = 1;
			B1[i*size+j] = 1;
		}
	}
}


void checkMatrix(double *result, int size){
	
	for (int i=0; i<size; i++){
		for (int j=0; j<size; j++){
			if (result[i*size+j] != size){
				printf("Error : value of result[%d][%d] is %f instead of %d !\n", i, j , result[i*size+j], size);
				return; 
			}
		}
	}
	
}
