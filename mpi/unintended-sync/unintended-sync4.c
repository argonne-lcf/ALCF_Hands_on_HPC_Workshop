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
    MPI_Request reqs[8];

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

    MPI_Irecv(rbufnorth, COUNT, MPI_INT, north, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(rbufwest, COUNT, MPI_INT, west, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Irecv(rbufsouth, COUNT, MPI_INT, south, 0, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(rbufeast, COUNT, MPI_INT, east, 0, MPI_COMM_WORLD, &reqs[3]);

    MPI_Isend(sbufsouth, COUNT, MPI_INT, south, 0, MPI_COMM_WORLD, &reqs[4]);
    MPI_Isend(sbufeast, COUNT, MPI_INT, east, 0, MPI_COMM_WORLD, &reqs[5]);
    MPI_Isend(sbufnorth, COUNT, MPI_INT, north, 0, MPI_COMM_WORLD, &reqs[6]);
    MPI_Isend(sbufwest, COUNT, MPI_INT, west, 0, MPI_COMM_WORLD, &reqs[7]);

    MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

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
