#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void checkMatrix(double *result, int size);
void matmul(double* A1, double* B1, double *result, int size, int N);
void initMatrix(double* A1, double* B1, int size);

int main(int argc, char **argv){
	int size = atoi(argv[1]);
	
	double *A1 = malloc(size*size*sizeof(double));
	double *B1 = malloc(size*size*sizeof(double));
	double *result = malloc(size*size*sizeof(double));
	
	initMatrix(A1, B1, size);
	matmul(A1, B1, result, size, 8);
	checkMatrix(result, size);
	return 0;
}


void matmul(double* A1, double* B1, double *result, int size, int N){
	int i, j, k, a, b, c, sum, Na, Nb, Nk;
	for (i = 0; i < size; i += N) {
		Na = i + N <= size ? N : size - i;
		for (j = 0; j < size; j += N) {
			Nb = j + N <= size ? N : size - j;
			for (k = 0; k < size; k+=N) {
				Nk = k + N <= size ? N : size - k;
				// On fait le produit matriciel : A[i,k] * B[k, j] -> (Na * Nb) * (Nb * Na)
				// Ensuite on fait : C[i, j] = ...
				for (a = 0; a < Na; a++) {
					for (b = 0; b < Nb; b++) {
						// On  fait le produit : C[i, j][a, b] = A[i, k][a, c] * B[k, j][c, b]
						sum = 0;
						for (c = 0; c < Nk; c++) {
							sum += A1[(i + a) * size + k + c] * B1[(k + c) * size + j + b];
						}
						result[(i + a) * size + j + b] += sum;
					}
				}

			}
		}
	}
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
