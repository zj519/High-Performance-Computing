#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

#define F77NAME(x) x##_
extern "C" {
    void F77NAME(dsbev)(const char& jobz, const char& ul, const int& n,
                           const int& kd,
                           double* ab, const int& ldab, double* w,
                           double* z,  const int& ldz,
                           double* work, int* info );
    void F77NAME(dgbtrf)(const int& M, const int& N, const int& KL,
                    const int& KU, double* A, const int& lda,
                    int* ipiv, int* info);
    void F77NAME(dgbtrs)(const char& TRANS, const int& N, const int& KL,
                    const int& KU, const int& NRHS,
                    const double* A, const int& lda,
                    int* ipiv, double* B, const int& ldb, int* info);
}


int main() {
    const int n = 20;
    const int kl = 1;
    const int ku = 1;
    const int lda = 2*kl + ku + 1;
    double* A = new double[lda*n]();
    double* w = new double[n]();
    double* z = 0;
    double* work = new double[3*n - 2]();
    int info = 0;
    double nu = 0.4;  // < 0.5
    // ======== CHECK EIGENVALUES OF EXPLICIT OPERATOR ================
    // We set up a general banded matrix, as no symmetric-banded option
    // available for triangular factorisation and solve later on.
    for (int i = 0; i < n; ++i) {
        A[i*lda+1] = nu;
        A[i*lda+2] = 1.0 - 2.0*nu;
        A[i*lda+3] = nu;
    }
    // Offset ptr A by 2, as SBEV wants diagonal and lower-diagonal.
    // But we use LDA=4, as this is the memory layout.
    F77NAME(dsbev)('N','L',n,1,A+2,lda,w,z,1,work,&info);
    if (info) {
        cout << "Error in DSBEV: " << info << endl;
        return 1;
    }
    // Print all the EVs. For nu < 0.5, these should be |w_i| < 1.0.
    for (int i = 0; i < n; ++i) {
        if (fabs(w[i]) > 1.0) {
            cout << "EV " << i << " = " << w[i] << endl;
            return 1;
        }
    }
    cout << "All eigenvalues are within the unit circle." << endl;

    // ====== SOLVE THE HEAT EQUATION IMPLICITLY =====================
    double Nt = 100;
    const double dx = 1.0/(n-1);
    double* u = new double[n]();
    int* ipiv = new int[n]();
    cout << "Corresponding timestep for given nu value: dt = "
         << nu*dx*dx << endl;
    // Initial condition
    for (int i = 0; i < n; ++i) {
        u[i] = sin(double(i)/(n-1)*M_PI);
    }
    // Print out the initial condition
    cout << "Initial condition" << endl;
    for (int i = 0; i < n; ++i) {
        cout << u[i] << endl;
    }
    // Modify A to represent the implicit operator (note change of signs)
    for (int i = 0; i < n; ++i) {
        A[i*lda+1] = -nu;
        A[i*lda+2] = 1.0 + 2.0*nu;
        A[i*lda+3] = -nu;
    }
    // To impose Dirichlet boundary conditions, we zero the off-diagonal entries
    // in the first and last rows of the matrix
    A[lda+1] = 0.0;
    A[(n-2)*lda+3] = 0.0;
    // Pre-factor matrix
    F77NAME(dgbtrf)(n, n, kl, ku, A, lda, ipiv, &info);
    if (info) {
        cout << "Error in DGBTRF: " << info << endl;
        return 1;
    }
    // Time integrate
    for (int t = 0; t < Nt; t++) {
        F77NAME(dgbtrs)('N', n, kl, ku, 1, A, lda, ipiv, u, n, &info);
        if (info) {
            cout << "Error in DGBTRS: " << info << endl;
        }
    }
    // Print the final solution
    cout << "Final solution" << endl;
    for (int i = 0; i < n; ++i) {
        cout << u[i] << endl;
    }
    return 0;
}

