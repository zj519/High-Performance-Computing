#include <iostream>
#include <cmath>
using namespace std;
#include <cblas.h>
#include <mpi.h>
int main(int argc, char* argv[]) {
    int n    = 0;
    int rank = 0;
    int size = 0;
    // Initialise MPI.
    int err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS) {
        cout << "Failed to initialise MPI" << endl;
        return -1;
    }
    // Get the rank and comm size on each process.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Check number of processes is suitable
    if (1024 % size) {
        if (rank == 0) {
            cout << "Error: Number of processes must be 2^n" << endl;
        }
        MPI_Finalize();
        return 0;
    }
    // Generate vectors and populate with random numbers
    n = 1024 / size;
    double* x = new double[n];
    double* y = new double[n];
    srand(time(0)+rank);
    for (int i = 0; i < n; ++i) {
        x[i] = (double)rand()/RAND_MAX;
        y[i] = (double)rand()/RAND_MAX;
    }
    // Compute local dot products and norms^2
    double dot  = cblas_ddot(n, x, 1, y, 1);
    double nrmx = cblas_ddot(n, x, 1, x, 1);
    double nrmy = cblas_ddot(n, y, 1, y, 1);
    // Variables for reduced results
    double dot_r;
    double nrmx_r;
    double nrmy_r;
    // Reduce SUM to compute global dot and norms
    MPI_Reduce(&dot,  &dot_r,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nrmx, &nrmx_r, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nrmy, &nrmy_r, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // Print out results.
    if (rank == 0) {
        cout << "Dot product: " << dot_r << endl;
        cout << "Norm x:      " << sqrt(nrmx_r) << endl;
        cout << "Norm y:      " << sqrt(nrmy_r) << endl;
    }
    delete[] x;
    delete[] y;
    // Finalise MPI.
    MPI_Finalize();
    return 0;
}

