/*
 * (C) 2014 The University of Chicago.
 */

/* COMPILE:
 *
 * cc -Wall warpdriveB.c -o warpdriveB
 */


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>

#define IOSIZE 200000

static int example1B(const char* dir, int rank, int nprocs);

int main(int argc, char **argv)
{
    char* dir;
    int ret;
    int rank;
    int nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if(argc != 2)
    {
        if(rank == 0)
        {
            fprintf(stderr, "Usage: warpdriveB <directory>\n");
            MPI_Finalize();
            return(-1);
        }
    }

    dir = strdup(argv[1]);

    /*****************/
    sleep(1);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) printf("  Running...\n");
    ret = example1B(dir, rank, nprocs);
    if(ret < 0)
    {
        if(rank == 0) fprintf(stderr, "Failure.\n");
        MPI_Finalize();
        return(-1);
    }
    if(rank == 0) printf("  Completed.\n");

    free(dir);
    MPI_Finalize();
    
    return(0);
}

static int example1B(const char* dir, int rank, int nprocs)
{
    char file_name[PATH_MAX];
    int ret;
    MPI_File fh;
    char msg[MPI_MAX_ERROR_STRING];
    int msg_len;
    char *buffer;
    MPI_Status status;
	MPI_Offset offset;
    int i;

    buffer = malloc(IOSIZE);

    sprintf(file_name, "%s/warpdriveB", dir);

    ret = MPI_File_open(MPI_COMM_WORLD, file_name, 
        MPI_MODE_CREATE |MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    if(ret != MPI_SUCCESS)
    {
        MPI_Error_string(ret, msg, &msg_len);
        fprintf(stderr, "Error: MPI_File_open: %s: %s\n", file_name, msg);
        free(buffer);
        return(-1);
    }

    for(i=0; i<512; i++)
    {
		/* skip ahead in file for each round of I/O */
		offset = (MPI_Offset)nprocs*i*IOSIZE;
		/* skip ahead to offset for this process' data */
		offset += (MPI_Offset)rank*IOSIZE;

        ret = MPI_File_write_at(fh, offset, buffer, IOSIZE, MPI_CHAR, 
            &status);
        if(ret != MPI_SUCCESS)
        {
            MPI_Error_string(ret, msg, &msg_len);
            fprintf(stderr, "Error: MPI_File_write_at: %s: %s\n", file_name, msg);
            free(buffer);
            return(-1);
        }
    }

    ret = MPI_File_close(&fh);
    if(ret != MPI_SUCCESS)
    {
        MPI_Error_string(ret, msg, &msg_len);
        fprintf(stderr, "Error: MPI_File_close: %s: %s\n", file_name, msg);
        free(buffer);
        return(-1);
    }

    free(buffer);
    return 0;
}


/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *
 * vim: ts=4
 * End:
 */ 


