#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;
#define F77NAME(x) x##_
extern "C" {
    void   F77NAME(dgbmv)(const char& trans, const int& m, const int& n,
                const int& kl, const int& ku,
                const double& alpha, const double* A, const int& lda,
                const double* x, const int& incx, const double& beta,
                double* y, const int& incy);
    void   F77NAME(daxpy)(const int& n, const double& alpha,
                const double* x, const int& incx, double* y, const int& incy);
    double F77NAME(dnrm2)(const int& n, const double* x, const int& incx);
}
int main() {
    // ==============================================
    // 1. Generate vector f
    // Number of grid points
    const int n = 1001;
    // Matrix is general banded with 3 upper diagonals and 0 lower diagonals
    const int kl = 0;
    const int ku = 3;
    // BLAS banded storage does not require 'fill' rows
    const int lda = 1 + ku;
    // Length of domain 0 <= x <= 1
    const double L = 1.0;
    // Grid point spacing
    const double dx = (double)L/(n-1);
    // Allocate f and g
    double* f = new double[n];
    double* g = new double[n];
    // Populate the f and initialize g to zero
    for (int i = 0; i < n; ++i) {
        f[i] = sin(M_PI*i*dx);
        g[i] = 0.0;
    }
    // =============================================
    // 2. Generate matrix D and populate b
    // Allocate banded storage for A (upper banded with ku=3)
    //
    //   x    x    x   -1. ...  -1.  -1.  -1.
    //   x    x    4.   4. ...   4.   4.   4.
    //   x   -5.  -5.  -5. ...  -5.  -5.  -5.
    //   2.   2.   2.   2. ...   2.   2.   2.
    //
    double* D = new double[lda*n];
    // Calculate the entries for the matrix
    double p = 1.0/dx/dx;
    double s[lda] = {-1.0*p, 4.0*p, -5.0*p, 2.0*p};
    // Populate the matrix (ok to put values in 'x' locations for simplicity)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < lda; ++j) {
            D[i*lda + j] = s[j];
        }
    }
    // Calculate values for b vector and put in g
    g[n-3] =     -p*sin(M_PI*(L+dx));
    g[n-2] =  4.0*p*sin(M_PI*(L+dx)) -     p*sin(M_PI*(L+2.0*dx));
    g[n-1] = -5.0*p*sin(M_PI*(L+dx)) + 4.0*p*sin(M_PI*(L+2.0*dx))
                - p*sin(M_PI*(L+3.0*dx));
    // =============================================
    // 3. Use BLAS
    // Compute matrix operation using DSBMV [ g <- A * f  +  g ]
    F77NAME(dgbmv)('N', n, n, kl, ku, 1.0, D, lda, f, 1, 1.0, g, 1);
    // =============================================
    // 4. Evaluate exact solution and compute error
    // Compute exact solution
    for (int i = 0; i < n; ++i) {
        f[i] = -M_PI*M_PI*sin(M_PI*i*dx);
    }
    // Calculate the point-wise difference
    F77NAME(daxpy)(n, -1.0, f, 1, g, 1);
    // Calculate the L2 error
    cout << "Error: " << F77NAME(dnrm2)(n, g, 1) << endl;
    // Clean up storage
    delete[] D;
    delete[] f;
    delete[] g;
}

