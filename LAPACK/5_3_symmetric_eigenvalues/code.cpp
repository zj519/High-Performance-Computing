#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cblas.h> // Include cblas.h for the C interface to BLAS
extern "C" {
    // LAPACK function declarations
    void dsyev_(const char *jobz, const char *uplo, const int *n, double *a,
                const int *lda, double *w, double *work, const int *lwork,
                int *info);
}
using namespace std;
int main() {
    const int n = 10;
    int info = 0;
    int lwork = -1;
    double wkopt;
    double* A = new double[n * n];
    double* w = new double[n];
    double* work;
    // Initialize random seed
    srand(time(0));
    // Populate matrix A with random values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (double)rand() / RAND_MAX;
        }
    }
    // Optimal workspace size query
    dsyev_("V", "U", &n, A, &n, w, &wkopt, &lwork, &info);
    lwork = (int)wkopt;
    work = new double[lwork];
    // Compute eigenvalues
    dsyev_("V", "U", &n, A, &n, w, work, &lwork, &info);
    // Print eigenvalues
    cout << "Eigenvalues:" << endl;
    for (int i = 0; i < n; i++) {
        cout << "Eigenvalue " << i << ": " << w[i] << endl;
    }
    // Clean up
    delete[] A;
    delete[] w;
    delete[] work;
    return 0;
}

