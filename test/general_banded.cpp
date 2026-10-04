
#include <iostream>
#include <iomanip>
#include <cmath>
#include "cblas.h"
using namespace std;
#define F77NAME(x) x##_
extern "C" {
    void   F77NAME(dgbmv)(const char& trans, const int& rows, const int& cols,
                const int& kl, const int& ku,
                const double& alpha, const double* D, const int& ldh,
                const double* x, const int& incx, const double& beta,
                double* b, const int& incy);
}



int main(){
    
    const int Nx=101;
    double h=0.01; //grid space same as dx
    const int ku=2;  //number of upper diagonal
    const int kl=2;  //number of lower diagonal
    //banded storage leading dimension = ku+kl+1
    const int ldh=ku+kl+1;
    
    
    double* f= new double[Nx];
    double*g= new double[Nx];
    double* D= new double[ldh*Nx];
    double L=1.0;
    //populate f and g and initialize to 0
    for (int i=0;i<Nx;i++){
        f[i]=cos(M_PI*i*h);
        g[i]=0.0;
    }
    
    //generate matrix D using genegral banded since it's easier and quicker to implement under time limit
    // x x -1 -1 ...-1
    //x 16 16 ..    16
    //-30 -30 ..    30
    //16 16 ...  16  x
    //-1 -1 ..-1 x   x
    double p=1.0/(12.0*h*h);
    double s[ldh]={-1.0*p,16.0*p,-30.0*p,16.0*p, -1.0*p};
    //populate matrix D
    for (int i=0;i<Nx;i++){
        for (int j=0;j<ldh;j++){
            D[i*ldh+j]=s[j];
        }
        
    }
    
    //calculate values for vector b and put in g
    g[0]=(16*cos(M_PI*(-h))-cos(M_PI*h*(-2.0)))*p;
    g[1]=(-cos(M_PI*h*(-1.0)))*p;
    g[Nx-2]=(-cos(M_PI*(L+h)))*p;
    g[Nx-1]=(16*cos(M_PI*(L+h))-cos(M_PI*(L+2.0*h)))*p;
    
    //use blas to compute g<--Df+b, where this vector b will be updted in g
    
    F77NAME(dgbmv)('N',Nx,Nx,kl,ku,1.0,D,ldh,f,1,1.0,g,1);
    
    //compute exact solution
    for (int i=0;i<Nx;i++){
        f[i]=-M_PI*M_PI*cos(M_PI*i*h);
        
    }
    
    //calculate the point-wize difference
    cblas_daxpy(Nx,-1.0,f,1,g,1);
    //calculate norm
    double error=cblas_dnrm2(Nx,g,1);
    
    cout<<"error is:"<<error<<endl;
    
    delete[] D;
    delete[] f;
    delete[] g;
    
    
    
}

