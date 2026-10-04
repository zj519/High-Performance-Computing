#include <iostream>
#include <iomanip>
#include <cmath>
#include "cblas.h"
using namespace std;
extern "C"{
    void dsbev_(const char *jobz, const char *uplo, const int *n, int *kd, double*ab, const int *ldab, double* w, double*z, const int *ldz, double*work, int*info);
}

int main (){
    const int Nx = 6;
    double h = 0.01;
    int k = 2;  // symmetric banded so number of superdiag=number of subdiag=2
    const int ldh = k + 1; // banded storage leading dimension 
    double* D = new double[ldh * Nx]; // symmetric banded storage
    int size =max(1,3*Nx-2);
    // Generate matrix D using symmetric banded storage
    // Diagonal elements
    // x  x -1 -1 ...-1
    // x 16 16 ..    16
    // -30 -30 ..    30
    
    double p = 1.0 / (12.0 * h * h);
    double s[ldh] = {-1.0 * p, 16.0 * p, -30.0 * p};
    // Populate matrix D
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < ldh; j++) {
            D[i * ldh + j] = s[j];
        }
    }
    // Compute eigenvalues and eigenvectors
    int info = 0;
    //int lwork = -1;
    //double wkopt;
    double* w = new double[Nx]; // Eigenvalues array
    double* z = new double[Nx * Nx]; // Eigenvectors matrix
    double* work = new double[size];
    
    // Optimal workspace size query
    //ssbev_("V", "L", &Nx, &k, D, &ldh, w, z, &Nx, &wkopt,  &info);
    //lwork = (int) wkopt;
    //work = new double[lwork];
    dsbev_("V", "U", &Nx, &k, D, &ldh, w, z, &Nx, work, &info);
    
    cout<<"info:"<<info<<endl;
    // Print eigenvalues
    cout << "Eigenvalues:" << endl;
    for (int i = 0; i < Nx; i++) {
        cout << "Eigenvalue " << i << ": " << w[i] << endl;
    }
    
    
    delete[] D;
    delete[] w;
    delete[] work;
    delete[] z; // Don't forget to free eigenvectors array
    
    return 0;
}

