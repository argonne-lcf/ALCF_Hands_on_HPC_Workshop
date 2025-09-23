/*
 * (C) 2014 The University of Chicago.
 */

/* COMPILE:
 *
 * cc -Wall fidgetspinnerB.c -o fidgetspinnerB
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
#include <assert.h>

static int example1A(const char* dir, int rank, int nprocs);

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
            fprintf(stderr, "Usage: fidgetspinnerB <directory>\n");
            MPI_Finalize();
            return(-1);
        }
    }

    dir = strdup(argv[1]);

    /*****************/
    sleep(1);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) printf("  Running...\n");
    ret = example1A(dir, rank, nprocs);
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

#define IOSIZE (1024*1024)

static int example1A(const char* dir, int rank, int nprocs)
{
    char file_name[PATH_MAX];
    int ret;
    int fd;
    char *buffer;
    off_t i;
    struct stat statbuf;
    int j;

    buffer = malloc(IOSIZE);

    sprintf(file_name, "%s/fidgetspinnerB-%d", dir, rank);

    fd = creat(file_name, O_WRONLY);
    if(fd < 0)
	{
		perror("creat");
		return(-1);
	}

	for(i=0; i<20; i++)
	{
		ret = write(fd, buffer, IOSIZE);
		if(ret < IOSIZE)
		{
			perror("write");
			return(-1);
		}
    }


	ret  = close(fd);
	if(ret < 0)
	{
		perror("close");
        	return(-1);
    	}

	MPI_Barrier(MPI_COMM_WORLD);

	if(rank == 0)
	{
		for(j=0; j<nprocs; j++)
		{
		    sprintf(file_name, "%s/fidgetspinnerB-%d", dir, j);
		    ret = stat(file_name, &statbuf);
		    if(ret < 0)
		    {
			perror("stat");
			return(-1);
		    }
		    assert(statbuf.st_size == (20 * IOSIZE));
		}
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


