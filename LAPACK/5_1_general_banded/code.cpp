#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;
#include "cblas.h"
#define F77NAME(x) x##_
extern "C" {
    // Performs LU factorisation of general banded matrix
    void F77NAME(dgbtrf)(const int& N, const int& M, const int& KL, 
                        const int& KU, double* AB, const int& LDAB,
                        int* IPIV, int* INFO);
    // Solves pre-factored system of equations
    void F77NAME(dgbtrs)(const char& TRANS, const int& N, const int& KL, 
                        const int& KU,
                        const int& NRHS, double* AB, const int& LDAB,
                        int* IPIV, double* B, const int& LDB, int* INFO);
}
/**
 * @brief Populate Helmholtz matrix.
 *
 * LAPACK only provides general banded routines, not symmetric banded, so we
 * must provide both the lower, kl, and upper, ku, diagonals. In addition,
 * LAPACK requires a further kl rows of 'padding' at the top of the matrix. This
 * is needed during the LU factorisation step.
 *
 * @param   nsv     Leading dimension of matrix
 * @param   H       Pointer to matrix storage of size 2*nsv
 * @param   lam     Lambda coefficient
 * @param   dx      Grid spacing
 */
void FillHelmholtzMatrix(int nsv, double* H, double lam, double dx, int* ipiv) {
    const int ldh = 4;      // Diagonal and upper diagonal
    int info;
    const double oodx2 = 1.0/dx/dx;
    //H[1] = -lam - 2.0*oodx2;
    for (int i = 0; i < nsv; ++i) {
        H[i*ldh    ] = 0.0; // Top row 'padding' could be left unset
        H[i*ldh + 1] = oodx2;
        H[i*ldh + 2] = -lam - 2.0*oodx2;
        H[i*ldh + 3] = oodx2;
    }
    F77NAME(dgbtrf)(nsv, nsv, 1, 1, H, ldh, ipiv, &info);
    if (info) {
        cout << "Failed to LU factorise matrix" << endl;
    }
}
/**
 * @brief Fills the forcing vector with
 *        \f$ f = -(\lambda + \pi^2) \sin(\pi x) \f$
 *
 * @param   n       Vector dimension
 * @param   f       Pointer to vector storage of length n
 * @param   lam     Lambda coefficient
 * @param   dx      Grid spacing
 */
void FillForcingFunction1(int n, double* f, double lam, double dx) {
    for (int i = 0; i < n; ++i) {
        f[i] = -(lam + M_PI*M_PI)*sin(M_PI*i*dx);
    }
}
/**
 * @brief Fills the forcing vector with
 *        \f$ f = -(\lambda + \pi^2) \cos(\pi x) \f$
 *
 * @param   n       Vector dimension
 * @param   f       Pointer to vector storage of length n
 * @param   lam     Lambda coefficient
 * @param   dx      Grid spacing
 */
void FillForcingFunction2(int n, double* f, double lam, double dx) {
    for (int i = 0; i < n; ++i) {
        f[i] = -(lam + M_PI*M_PI)*cos(M_PI*i*dx);
    }
}
/**
 * @brief Enforces zero Dirichlet boundary conditions. 
 *
 * @param   n       Vector dimension
 * @param   f       Pointer to forcing term storage of length n
 * @param   u       Pointer to solution vector storage of length n
 * @param   lam     Lambda coefficient
 * @param   dx      Grid spacing
 */
void EnforceBoundaryConditions1(int n, double* f, double* u, double lam, double dx) {
    u[0] = sin(0);
    u[n-1] = sin(M_PI*(n-1)*dx);
    f[1] -= u[0]/dx/dx;
    f[n-2] -= u[n-1]/dx/dx;
}
/**
 * @brief Enforces zero Dirichlet boundary conditions. 
 *
 * @param   n       Vector dimension
 * @param   f       Pointer to forcing term storage of length n
 * @param   u       Pointer to solution vector storage of length n
 * @param   lam     Lambda coefficient
 * @param   dx      Grid spacing
 */
void EnforceBoundaryConditions2(int n, double* f, double* u, double lam, double dx) {
    u[0] = cos(0);
    u[n-1] = cos(M_PI*(n-1)*dx);
    f[1] -= u[0]/dx/dx;
    f[n-2] -= u[n-1]/dx/dx;
}
/**
 * @brief Computes the exact solution for first problem with
 *        \f$ u = \sin(\pi x) \f$
 *
 * @param   n       Vector dimension
 * @param   e       Pointer to solution vector storage of length n
 * @param   dx      Grid spacing
 */
void ExactSolution1(int n, double* e, double dx) {
    for (int i = 0; i < n; ++i) {
        e[i] = sin(M_PI*i*dx);
    }
}
/**
 * @brief Computes the exact solution for first problem with
 *        \f$ u = \cos(\pi x) \f$
 *
 * @param   n       Vector dimension
 * @param   e       Pointer to solution vector storage of length n
 * @param   dx      Grid spacing
 */
void ExactSolution2(int n, double* e, double dx) {
    for (int i = 0; i < n; ++i) {
        e[i] = cos(M_PI*i*dx);
    }
}
/**
 * @brief Prints a symmetric tri-banded matrix in column-major format.
 *
 * Note that for a symmetric matrix, only the upper diagonals or lower
 * diagonals need to be stored.
 *
 * @param   nsv     Matrix dimension
 * @param   H       Pointer to matrix storage of size 2*nsv
 */
void PrintMatrix(int nsv, double* H) {
    const int ldh = 4;
    cout.precision(4);
    for (int i = 0; i < ldh; ++i) {
        for (int j = 0; j < nsv; ++j) {
            cout << setw(6) << H[j*ldh+i] << " ";
        }
        cout << endl;
    }
    cout << endl;
}
/**
 * @brief Prints a vector
 *
 * @param   n       Vector dimension
 * @param   u       Pointer to vector storage of length n
 * @param   dx      Grid spacing
 */
void PrintVector(int n, double* u, double dx) {
    for (int i = 0; i < n; ++i) {
        cout << i*dx << "  " << u[i] << endl;
    }
    cout << endl;
}

/**
 * @brief Solve the matrix problem \f$ Hu=f \f$ using the LAPACK routine DGBSV
 *
 * @param   nsv     Dimension of matrix system
 * @param   H       Pointer to matrix storage of size ldh*nsv containing
 *                  the general banded matrix, including padding.
 * @param   f       Pointer to vector of length nsv containing forcing for the
 *                  unknown degrees of freedom.
 * @param   u       Pointer to solution vector of length nsv, for the unknown
 *                  degrees of freedom.
 */
void SolveLapack(int nsv, double* H, double* f, double* u, int* ipiv) {
    int info;
    int nrhs = 1;
    int ldh = 4;
    int ldb = nsv;
    int kl = 1;         // Number of lower diagonals
    int ku = 1;         // Number of upper diagonals
    cblas_dcopy(nsv, f, 1, u, 1);
    F77NAME(dgbtrs)('N', nsv, kl, ku, nrhs, H, ldh, ipiv, u, ldb, &info);
    if (info) {
        cout << "Error in solve: " << info << endl;
    }
}


/**
 * @brief Solves the Helmholtz problem for two different forcing terms.
 */
int main() {
    const int    n   = 21;          // Number of grid-points
    const int    nsv = n - 2;       // Number of unknown DOFs
    const int    ldh = 4;           // Leading dimension of H (2*L + 1 + U)
    const double lam = 1.0;         // Value of Lambda
    const double L   = 1.0;         // Length of domain
    const double dx  = L / (n - 1); // Grid-point spacing
    double* H = new double[ldh*nsv];// Helmholtz matrix storage
    double* u = new double[n];      // Solution vector
    double* f = new double[n];      // Forcing vector
    double* e = new double[n];      // Exact solution vector
    int* ipiv = new int[nsv];       // Pivot storage for LAPACK
    // Generate the Helmholtz matrix in symmetric banded storage.
    FillHelmholtzMatrix(nsv, H, lam, dx, ipiv);
    cout << "Helmholtz matrix (symmetric banded storage): " << endl;
    PrintMatrix(nsv, H);
    // Problem 1 with f = -(lambda + pi^2) sin(pi x)
    FillForcingFunction1(n, f, lam, dx);
    cblas_dscal(n, 1.0, u, 1);      // x_0 = 0
    EnforceBoundaryConditions1(n, f, u, lam, dx);
    SolveLapack(nsv, H, f+1, u+1, ipiv);
    cout << "Solution: " << endl;
    PrintVector(n, u, dx);
    cout << "Exact: " << endl;
    ExactSolution1(n, e, dx);
    PrintVector(n, e, dx);
    cblas_daxpy(n, -1.0, u, 1, e, 1);
    double err1 = cblas_dnrm2(n, e, 1);
    // Problem 2 with f = -(lambda + pi^2) cos(pi x)
    FillForcingFunction2(n, f, lam, dx);
    cblas_dscal(n, 0.0, u, 1);      // x_0 = 0
    EnforceBoundaryConditions2(n, f, u, lam, dx);
    SolveLapack(nsv, H, f+1, u+1, ipiv);
    cout << "Solution: " << endl;
    PrintVector(n, u, dx);
    cout << "Exact: " << endl;
    ExactSolution2(n, e, dx);
    PrintVector(n, e, dx);
    cblas_daxpy(n, -1.0, u, 1, e, 1);
    double err2 = cblas_dnrm2(n, e, 1);
    // Print the resulting errors
    cout << "Error (problem 1): " << err1 << endl;
    cout << "Error (problem 2): " << err2 << endl;
    // Clean up
    delete[] H;
    delete[] u;
    delete[] f;
    delete[] e;
    delete[] ipiv;
}

