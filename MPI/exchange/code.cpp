#include <iostream>
using namespace std;

#include <mpi.h>
int main (int argc, char* argv[]){
    int a =0;
    int b=0;
    int rank,size, retval_rank, retval_size;
    MPI_Init(&argc, &argv);
    retval_rank=MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    retval_size=MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (retval_rank==MPI_ERR_COMM || retval_size==MPI_ERR_COMM){
        cout<<"invalid communicator"<<endl;
        return 1;
    }
    
    if (rank==0){
        a=2;
        //send a from 0 to process rank 1
        MPI_Send(&a,1,MPI_INT,1,0,MPI_COMM_WORLD);
        //rank 0 receive b from process rank 1
        MPI_Recv(&b,1,MPI_INT,1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    
    else {
        b=3;
        //rank 1 receive a from process rank 0
        MPI_Recv(&a, 1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //rank 1 send b to 0
        MPI_Send(&b,1,MPI_INT,0,1,MPI_COMM_WORLD);
        
    }
    
    cout<<"rank:"<<rank<<",a="<<a<<",b="<<b<<endl;
    
    MPI_Finalize();
    return 0;
    
}

