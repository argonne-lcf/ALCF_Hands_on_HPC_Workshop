#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define COUNT (64*1024)

int main(void)
{
    int rank, size;
    double *sbufnorth, *sbufsouth, *sbufeast, *sbufwest;
    double *rbufnorth, *rbufsouth, *rbufeast, *rbufwest;
    int px = 3, py = 4; /* hardcoded 3x4 decomposition */
    int rx, ry;
    int north, south, east, west;

    /* initialize MPI envrionment */
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* determine my coordinates (x,y) -- rank=x*a+y in the 2d processor array */
    rx = rank % px;
    ry = rank / px;

    /* determine my four neighbors */
    north = (ry - 1) * px + rx;
    if (ry - 1 < 0)
        north = MPI_PROC_NULL;
    south = (ry + 1) * px + rx;
    if (ry + 1 >= py)
        south = MPI_PROC_NULL;
    west = ry * px + rx - 1;
    if (rx - 1 < 0)
        west = MPI_PROC_NULL;
    east = ry * px + rx + 1;
    if (rx + 1 >= px)
        east = MPI_PROC_NULL;

    //printf("north = %d, south = %d, east = %d, west = %d\n", north, south, east, west);

    sbufnorth = malloc(sizeof(double) * COUNT);
    sbufsouth = malloc(sizeof(double) * COUNT);
    sbufeast = malloc(sizeof(double) * COUNT);
    sbufwest = malloc(sizeof(double) * COUNT);

    rbufnorth = malloc(sizeof(double) * COUNT);
    rbufsouth = malloc(sizeof(double) * COUNT);
    rbufeast = malloc(sizeof(double) * COUNT);
    rbufwest = malloc(sizeof(double) * COUNT);

    MPI_Barrier(MPI_COMM_WORLD);

    if (south != MPI_PROC_NULL) {
        MPI_Send(sbufsouth, COUNT, MPI_INT, south, 0, MPI_COMM_WORLD);
    }
    if (north != MPI_PROC_NULL) {
        MPI_Recv(rbufnorth, COUNT, MPI_INT, north, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (east != MPI_PROC_NULL) {
        MPI_Send(sbufeast, COUNT, MPI_INT, east, 0, MPI_COMM_WORLD);
    }
    if (west != MPI_PROC_NULL) {
        MPI_Recv(rbufwest, COUNT, MPI_INT, west, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);        
    }
    if (north != MPI_PROC_NULL) {
        MPI_Send(sbufnorth, COUNT, MPI_INT, north, 0, MPI_COMM_WORLD);
    }
    if (south != MPI_PROC_NULL) {
        MPI_Recv(rbufsouth, COUNT, MPI_INT, south, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (west != MPI_PROC_NULL) {
        MPI_Send(sbufwest, COUNT, MPI_INT, west, 0, MPI_COMM_WORLD);
    }
    if (east != MPI_PROC_NULL) {
        MPI_Recv(rbufeast, COUNT, MPI_INT, east, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    free(sbufnorth);
    free(sbufsouth);
    free(sbufeast);
    free(sbufwest);

    free(rbufnorth);
    free(rbufsouth);
    free(rbufeast);
    free(rbufwest);

    MPI_Finalize();
    return 0;
}
