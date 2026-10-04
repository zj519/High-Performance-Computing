//dot product

#include <iostream>
using namespace std;

#include <mpi.h>
int main (int argc, char* argv[]){
    int myrank,p,i,k;
    MPI_Init(&argc, &argv);  //initialize MPI program
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); //get rank
    MPI_Comm_size(MPI_COMM_WORLD, &p); //get number of process
    
    const int n =64;
    int loc_n=n/p;  //assume p divides n
    
    double* x= new double[loc_n];
    double*y=new double[loc_n];
    
    srand(time(0)+myrank);  //seed using time and rank
    double loc_dot=0.0;
    for (int i=0;i<loc_n;i++){
        loc_dot+=x[i]*y[i];
    }
    
    double* loc_dots;
    if (myrank==0) loc_dots=new double[p];
    
    MPI_Gather(&loc_dot,1,MPI_DOUBLE,loc_dots,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    
    //first process compute results
    if (myrank==0){
        double dot=0.0;
        for (int k=0;k<p;k++){
            dot+=loc_dots[k];
        }
        
        cout<<"dot product:"<<dot<<endl;
        
    }
    
    MPI_Finalize();
    
    
    
}

