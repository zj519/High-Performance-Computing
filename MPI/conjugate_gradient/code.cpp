#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include "cblas.h"
using namespace std;
// Generate a random n x n matrix
void FillMatrix(int n, double* H) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            H[i*n+j] = double(rand())/(RAND_MAX/2.0)-1.0;
        }
    }
}
// Generate a random vector
void FillVector(int n, double* f) {
    for (int i = 0; i < n; ++i) {
        f[i] = double(rand())/(RAND_MAX/2.0)-1.0;
    }
}
// Print a matrix
void PrintMatrix(int n, double* H) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << setw(10) << H[i*n+j] << " ";
        }
        cout << endl;
    }
}
// Print a vector
void PrintVector(int n, double* u) {
    for (int i = 0; i < n; ++i) {
        cout << u[i] << endl;
    }
}
// Solve the linear system using Conjugate Gradient method
void SolveConjugateGradient(int n_local, double* A_local, double* b_local, double* x_local) {
    double* r = new double[n_local];
    double* p = new double[n_local];
    double* t = new double[n_local]; //temp
    int k;
    double alpha;
    double beta;
    double eps;
    double tol = 0.00001;
    cblas_dcopy(n_local, b_local, 1, r, 1); // r_0 = b_local (i.e. b_local)
    cblas_dsymv(CblasRowMajor, CblasUpper, n_local, -1.0, A_local, n_local,
                    x_local, 1, 1.0, r, 1); // r_0 = b_local - A_local x_local
    cblas_dcopy(n_local, r, 1, p, 1);       // p_0 = r_0
    k = 0;
    do {
        cblas_dsymv(CblasRowMajor, CblasUpper, n_local, 1.0, A_local, n_local,
                    p, 1, 0.0, t, 1);        // t = A_local p_k
        alpha = cblas_ddot(n_local, t, 1, p, 1); // alpha = p_k^T A_local p_k
        alpha = cblas_ddot(n_local, r, 1, r, 1) / alpha; // compute alpha_k
        beta  = cblas_ddot(n_local, r, 1, r, 1); // r_k^T r_k
        cblas_daxpy(n_local, alpha, p, 1, x_local, 1); // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n_local, -alpha, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A_local p_k
        eps = cblas_dnrm2(n_local, r, 1);
        cout << "Iteration " << k << ": eps=" << eps << endl;
        if (eps < tol*tol) {
            break;
        }
        beta = cblas_ddot(n_local, r, 1, r, 1) / beta;
        cblas_dcopy(n_local, r, 1, t, 1);
        cblas_daxpy(n_local, beta, p, 1, t, 1);
        cblas_dcopy(n_local, t, 1, p, 1);
        k++;
    } while (k < 5000); // Set a maximum number of iterations
    delete[] r;
    delete[] p;
    delete[] t;
}
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int n = 8;
    const int n_local = n / size; // Assuming n is divisible by size
    const int offset = rank * n_local;
    double* M = new double[n*n];
    double* A = new double[n_local*n];
    double* x = new double[n];
    double* y = new double[n];
    double* b = new double[n_local];
    srand(time(0));
    if (rank == 0) {
        // Populate M and x
        FillMatrix(n, M);
        FillVector(n, x);
        cout << "Matrix M:" << endl;
        PrintMatrix(n, M);
        cout << "Initial vector:" << endl;
        PrintVector(n, x);
    }
    // Scatter M and x among processes
    MPI_Scatter(M, n*n_local, MPI_DOUBLE, A, n*n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Compute b = A*x locally
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n_local, n, 1.0, A, n, x, 1, 0.0, b, 1);
    // Solve system using the Conjugate Gradient method
    SolveConjugateGradient(n_local, A, b, y);
    // Gather y from all processes to y in rank 0
    MPI_Gather(y, n_local, MPI_DOUBLE, y, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "Solution vector:" << endl;
        PrintVector(n, y);
    }
    delete[] M;
    delete[] A;
    delete[] x;
    delete[] y;
    delete[] b;
    MPI_Finalize();
    return 0;
}


