#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
using namespace std;
int main(int argc, char* argv[]) {
    int rank, size;
    const int n = 1000; // Size of the matrix and vector (2n)
    const int m = 2;    // Number of processes (2m)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2 * m) {
        if (rank == 0) {
            cout << "Error: Number of processes must be 2m." << endl;
        }
        MPI_Finalize();
        return -1;
    }
    // Initialize random seed
    srand(time(NULL) + rank);
    // Generate matrix A and vector x on the root process
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> x(n);
    if (rank == 0) {
        // Fill matrix A with random values
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] = (double)rand() / RAND_MAX;
            }
        }
        // Fill vector x with random values
        for (int i = 0; i < n; ++i) {
            x[i] = (double)rand() / RAND_MAX;
        }
    }
    // Broadcast matrix A and vector x to all processes
    MPI_Bcast(&A[0][0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Perform matrix-vector multiplication locally
    vector<double> y_local(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            y_local[i] += A[i][j] * x[j];
        }
    }
    // Gather results on the root process
    vector<double> y(n);
    MPI_Gather(&y_local[0], n, MPI_DOUBLE, &y[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Print result on the root process
    if (rank == 0) {
        cout << "Result vector y:" << endl;
        for (int i = 0; i < n; ++i) {
            cout << y[i] << " ";
        }
        cout << endl;
    }
    MPI_Finalize();
    return 0;
}


