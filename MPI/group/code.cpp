#include<iostream>
#include<mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    // Get the rank and size in the original communicator
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 16) {
        if (world_rank == 0) {
            std::cout << "This example requires 16 ranks." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // We are going to select 7 ranks
    const int n = 7;

    // The ranks we select are those which are prime numbers
    int ranks[n] = {1, 2, 3, 5, 7, 11, 13};

    // Construct a group containing all of the prime ranks in the world_group
    MPI_Group prime_group;
    MPI_Group_incl(world_group, n, ranks, &prime_group);

    // Create a new communicator based on the group
    MPI_Comm prime_comm;
    MPI_Comm_create(MPI_COMM_WORLD, prime_group, &prime_comm);

    int prime_rank = -1, prime_size = -1;
    if (MPI_COMM_NULL != prime_comm) {
        MPI_Comm_rank(prime_comm, &prime_rank);
        MPI_Comm_size(prime_comm, &prime_size);
    }

    // -1 indicates this world rank is not present in the prime communicator.
    std::cout << "World:" << world_rank << "/" << world_size
              << " Prime:" << prime_rank << "/" << prime_size << std::endl;

    MPI_Finalize();
    return 0;
}

