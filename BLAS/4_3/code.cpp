#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;
#include "cblas.h"
// Generate a random n x n matrix
void FillMatrix(int n, double* H) {
    for (int i = 0; i < n; ++i) {           // index rows
        for (int j = 0; j < n; ++j) {     // index cols
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
// Solve the linear system
void SolveConjugateGradient(int n, double* A, double* b, double* x) {
    double* r = new double[n];
    double* p = new double[n];
    double* t = new double[n]; //temp
    int k;
    double alpha;
    double beta;
    double eps;
    double tol = 0.00001;
    cblas_dcopy(n, b, 1, r, 1);        // r_0 = b (i.e. b)
    cblas_dsymv(CblasRowMajor, CblasUpper, n, -1.0, A, n,
                    x, 1, 1.0, r, 1);  // r_0 = b - A x_0
    cblas_dcopy(n, r, 1, p, 1);        // p_0 = r_0
    k = 0;
    do {
        cblas_dsymv(CblasRowMajor, CblasUpper, n, 1.0, A, n,
                    p, 1, 0.0, t, 1);       // t = A p_k
        alpha = cblas_ddot(n, t, 1, p, 1);  // alpha = p_k^T A p_k
        alpha = cblas_ddot(n, r, 1, r, 1) / alpha; // compute alpha_k
        beta  = cblas_ddot(n, r, 1, r, 1);  // r_k^T r_k
        cblas_daxpy(n, alpha, p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n, -alpha, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k
        eps = cblas_dnrm2(n, r, 1);
        cout << "Iteration " << k << ": eps=" << eps << endl;
        if (eps < tol*tol) {
            break;
        }
        beta = cblas_ddot(n, r, 1, r, 1) / beta;
        cblas_dcopy(n, r, 1, t, 1);
        cblas_daxpy(n, beta, p, 1, t, 1);
        cblas_dcopy(n, t, 1, p, 1);
        k++;
    } while (k < 5000); // Set a maximum number of iterations
    delete[] r;
    delete[] p;
    delete[] t;
}
int main() {
    const int    n   = 8;
    double* M = new double[n*n];
    double* A = new double[n*n];
    double* x = new double[n];
    double* y = new double[n];
    double* b = new double[n];
    srand(time(0));
    // Populate M and x
    // M is symmetric
    FillMatrix(n, M);
    FillVector(n, x);
    // Compute A = M^T M
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                n, n, 1.0, M, n, 0.0, A, n);
    // Print matrices
    cout << "Matrix M:" << endl;
    PrintMatrix(n, M);
    cout << "Symmetric positive A = M*M:" << endl;
    PrintMatrix(n, A);
    cout << "Initial vector:" << endl;
    PrintVector(n, x);
    // Compute a RHS b = A*x
    cblas_dsymv(CblasRowMajor, CblasUpper, n, 1.0, A, n,
                x, 1, 0.0, b, 1);
    // Solve system using the Conjugate Gradient method
    cout << "Solving linear system:" << endl;
    SolveConjugateGradient(n, A, b, y);
    cout << "Solution vector:" << endl;
    PrintVector(n, y);
    // Compare the result y with the original vector x
    cblas_daxpy(n, -1.0, x, 1, y, 1);
    cout << "Error: " << cblas_dnrm2(n, y, 1) << endl;
    delete[] M;
    delete[] A;
    delete[] x;
    delete[] y;
    delete[] b;
    return 0;
}


