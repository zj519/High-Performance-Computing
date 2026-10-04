#include <iostream>
#include <sstream>
using namespace std;
#include <mpi.h>
int main(int argc, char* argv[]) {
    int rank = 0;
    int size = 0;
    int num  = 0;
    // Initialise MPI
    int err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS) {
        cout << "Error initializing MPI." << endl;
        return -1;
    }
    // Get rank of process and size of communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Verify we are running in parallel!
    if (size == 1) {
        cout << "This program must be run in parallel." << endl;
        return -1;
    }
    // Get a number from the user on rank 0.
    if (rank == 0) {
        string input;
        cout << "Enter a number to send around the ring: ";
        cin >> input;
        stringstream S(input);
        S >> num;
    }
    else {
        // Other processes rank > 0 wait for a number to be received first.
        MPI_Recv(&num, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        cout << "Process " << rank << " received " << num
             << " from rank " << rank - 1 << endl;
    }
    // On rank 0, we send the number. Other ranks send after first receiving.
    MPI_Send(&num, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
    // Only rank 0 now needs to receive the number from the last process.
    if (rank == 0) {
        MPI_Recv(&num, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        cout << "Process " << rank << " received " << num
             << " from rank " << size - 1 << endl;
    }
    // Finalize MPI
    MPI_Finalize();
    return 0;
}


