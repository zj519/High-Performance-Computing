
#include <iostream>
#include <iomanip>
#include <cmath>
#include "cblas.h"
using namespace std;
#define F77NAME(x) x##_
extern "C" {
                
    void dsyev_(const char*jobz, const char *uplo,const int *n, double *a, const int *lda, double *w, double *work, const int *lwork, int *info);
}

int main(){
    
    const int Nx=5;
    double h=0.01; //grid space same as dx
    const int ku=2;  //number of upper diagonal
    const int kl=2;  //number of lower diagonal
    //banded storage leading dimension = ku+kl+1
    const int ldh=ku+kl+1;
    
    double* f= new double[Nx];
    double*g= new double[Nx];
    double* D= new double[ldh*Nx];
    //double L=1.0;
    //populate f and g and initialize to 0
    for (int i=0;i<Nx;i++){
        f[i]=cos(M_PI*i*h);
        g[i]=0.0;
    }
    
    //generate matrix D using symmetric
    // x x -1 -1 ...-1
    //x 16 16 ..    16
    //-30 -30 ..    30
    //16 16 ...  16  x
    //-1 -1 ..-1 x   x
    double p=1.0/(12.0*h*h);
    //double s[ldh]={-1.0*p,16.0*p,-30.0*p,16.0*p, -1.0*p};
    //populate matrix D
    D[0]=-30.0*p;
    for (int i=0;i<Nx;i++){
        for (int j=0;j<Nx;j++){
        D[i*Nx+i-1]=16.0*p;
        D[i*Nx+i-2]=-1.0*p;
        D[i*Nx+i]=-30.0*p;
        
        }
    }
    
    int info =0;
    int lwork=-1;
    double wkopt;
    double* w= new double[Nx*Nx];
    double* work;
    // Optimal workspace size query
    dsyev_("V", "U", &Nx, D, &Nx, w, &wkopt, &lwork, &info);
    lwork = (int)wkopt;
    work = new double[lwork];
    // Compute eigenvalues
    dsyev_("V", "U", &Nx, D, &Nx, w, work, &lwork, &info);
    // Print eigenvalues
    cout << "Eigenvalues:" << endl;
    for (int i = 0; i < Nx; i++) {
        cout << "Eigenvalue " << i << ": " << w[i] << endl;
    }
    
    
}

