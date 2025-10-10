// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

#ifdef USE_MPI
#include <mpi.h>
#endif

// Global variables, defined here
bool run_tracker = false;
bool run_mpitracker = false;
pid_t	thispid;
int	mpi_rank = -1;

hrtime_t starttime;

/*==================================================================*/
/* Routine to set up the run*/
/*	process arguments to extract values for nn, niter, and kkmax */
/*	check for environment variable "RUN_TRACKER" */
/*	If USE_MPI is defined, call MPI_Init */

static void Print_Usage(void);

static char hostname[1024];

void
tracker()
{
  int ret=0;

  if (run_tracker) {
    ret = system("tracker");

    /* check the return code */
    if (ret != 0 ) {
      fprintf(stderr, "ERROR: launch of \"system(tracker)\" failed, returning %d; errno = %d, msg = %s\n", ret, errno, strerror(errno) );
    } else {
       //fprintf(stdout, "tracker run done\n" );
    }
  }
}

void
mpitracker()
{
  int ret=0;

  if (run_mpitracker) {
    ret = system("mpitracker");

    /* check the return code */
    if (ret != 0 ) {
      fprintf(stderr, "ERROR: launch of \"system(mpitracker)\" failed, returning %d; errno = %d, msg = %s\n", ret, errno, strerror(errno) );
    } else {
      fprintf(stdout, "mpitracker run done\n" );
    }
  }
}


void
setup_run(int argcc, char **argvv)
{
  int		i, j, num;
  char	*p;

  starttime = gethrtime();

  if (argcc >=2)  /* run testcode with options */ {
    for (i = 1; i < argcc; i++) {
      j = i;

      if(argvv[i][0] != '-') {
  	Print_Usage();
      }

      if (strlen(argvv[i]) == 2) {
  	/* argument has blank separating key and number */
  	j++;
  	if (argcc > j) {
  	    p = argvv[j];
  	    num = atoll(p);
  	} else {
  	    Print_Usage();
  	}
      } else {
  	/* argument has no blank separating key and number */
  	p = argvv[i] + 2;
  	num = atoll(p);
      }
  
      switch (argvv[i][1]) {
  	case 'N':
  	    nn = num;
  	    break;
  
  	case 'I':
  	    niter = num;
  	    break;
  
  	case 'K':
	    // Iterations in kernel -- reset dividing by 200 for nooffload
  	    kscale = num;
  	    kkmax = num * 2000;
  	    break;
  
  	default:
  	    Print_Usage();
      }
      i = j;
    }
  }

  thispid = getpid();
  fprintf(stdout, "    [%d] minitest process started\n", thispid );

  // Check for environment variable to run the tracker binary
  // It is run before MPI_Init and after MPI_Fini
  char *s = getenv("RUN_TRACKER");
  if (s != NULL) { 
    fprintf(stdout, "    [%d] Running of tracker enabled\n", thispid );
    run_tracker = true;
  } else {
    // fprintf(stdout, "    [%d] Running of tracker disabled\n", thispid );
    run_tracker = false;
  }

  // run tracker before MPI_Init
  tracker();

#ifdef USE_MPI
  // Check for environment variable to run the the mpitracker binary
  // It is run after MPI_Init and again before MPI_Fini
  s = getenv("RUN_MPITRACKER");
  if (s != NULL) { 
    fprintf(stdout, "    [%d] Running of mpitracker enabled\n", thispid );
    run_mpitracker = true;
  } else {
    // fprintf(stdout, "    [%d] Running of mpitracker disabled\n", thispid );
    run_mpitracker = false;
  }

  //Initialize MPI
  int rex = MPI_Init(&argcc, &argvv);
  if (rex != MPI_SUCCESS) {
    fprintf(stderr, "    [%d] ERROR: MPI_Init failed, returning %d\n", thispid, rex);
    exit (-1);
  } else {
    fprintf(stdout, "    [%d] MPI_Init succeeded, returning %d\n", thispid, rex);
  }

  // Determine rank
  int res;
  MPI_Comm communicator = MPI_COMM_WORLD;
  res = MPI_Comm_rank (communicator, &mpi_rank);
  if (res != MPI_SUCCESS) {
    fprintf(stderr, "    [%d] ERROR: MPI_Comm_rank failed, returning %d\n", thispid, res);
    exit (0);
  } else {
    // fprintf(stdout, "    [%d] MPI_Comm_rank = %d \n", thispid, mpi_rank);
  }
  // invoke mpitracker
   mpitracker();
#endif

  int ret = gethostname (hostname, 1024);
  if ( ret != 0 ) {
    fprintf(stderr, "    [%d] ERROR: gethostname failed: %d, err = %s\n", thispid, ret, strerror(errno) ); 
    exit(-1);
  }

  if (mpi_rank == -1 ) {
    fprintf(stdout, "  [%d] Process %d, Host %s\n",
      thispid, thispid, hostname );
  } else {
    fprintf(stdout, "  [%d] Process %d, MPI Rank %d,  Host %s\n",
      thispid, thispid, mpi_rank, hostname );
  }
  // Measure the overhead of real-time accounting
  hrtime_t accttime = gethrtime();
  hrtime_t endtime = gethrtime();
  double  tempus =  (double) (endtime - accttime)  ;
  fprintf(stdout, "       [%d] overhead of real-time delta measurement: %10.3f ns.\n\n",
    thispid, tempus);
}

// Synchronize  multiple MPI processes  -- call MPI_BARRIER
void
mpisync()
{
#ifdef USE_MPI
  int res = MPI_Barrier(MPI_COMM_WORLD);
  if (res != MPI_SUCCESS) {
    fprintf(stderr, "    [%d] ERROR: MPI_Barrier failed for MPI Rank %d, returning %d\n", thispid, mpi_rank, res);
    exit (0);
  } else {
    // fprintf(stdout, "    [%d] MPI_Barrier succeeded for MPI Rank %d \n", thispid, mpi_rank);
  }
#endif
}

static void
Print_Usage(void)
{
  fprintf(stdout, "Usage: <test_name> [-N array_size] [-K kernel_iteration_multiplier] [-I iteration_count]\n");

  exit(-1);

}


/*==================================================================*/
/* Routine to tear down the run*/
/*	If USE_MPI is defined, call MPI_Finalize */
void
teardown_run(void)
{
#ifdef USE_MPI
  mpitracker();
  MPI_Finalize();
#endif

  tracker();

  hrtime_t teardowntime = gethrtime();
  double tempus =  (double) (teardowntime - starttime) / (double)1000000000.;
  fprintf(stdout, "    [%d]   minitest app exiting at timestamp %13.9f s.\n",
    thispid, tempus );
  fprintf(stdout, "======================================================\n");
}

/*==================================================================*/
/*	Various routines to allocate and initialize data */
/*	Also, routines to check data, and output results */

void
allocinitdata(int numthreads)
{
#if 0
  fprintf(stdout, "[%d] Allocating and initializing data for %d threads\n", thispid, numthreads );
#endif

  /* allocate pointer arrays for the threads */
  rptr = (double **) calloc(numthreads, sizeof(double *) );
  lptr = (double **) calloc(numthreads, sizeof(double *) );
  pptr = (double **) calloc(numthreads, sizeof(double *) );

  /* allocate the l, r, and p arrays for each thread */
  for ( int k = 0; k < numthreads; k++) {
#if 0
    fprintf(stdout, "  [%d] thread %d allocating and initializing data\n", thispid, k );
#endif

    /* allocate and initialize the l and r arrays */
    lptr[k] = (double *) malloc (nn * sizeof(double) );
    if(lptr[k] == NULL) {
      fprintf(stderr, "[%d] Allocation for lptr[%d] failed; aborting\n", thispid, k);
      abort();
    }
    init(lptr[k], nn);

    rptr[k] = (double *) malloc (nn * sizeof(double) );
    if(rptr[k] == NULL) {
      fprintf(stderr, "[%d] Allocation for rptr[%d] failed; aborting\n", thispid, k);
      abort();
    }
    init(rptr[k], nn);

    /* allocate and clear the result array */
    pptr[k] = (double *) calloc(nn, sizeof(double) );
    if(pptr[k] == NULL) {
      fprintf(stderr, "[%d] Allocation for pptr[%d] failed; aborting\n", thispid, k);
      abort();
    }
#if 0
    fprintf(stdout, "  [%d] thread %d finished allocating and initializing data\n", thispid, k );
#endif
  }

  // DEBUG -- print addresses and result contents
#if 0
  fprintf(stdout, "[%d] Initial allocation of arrays\n", thispid );
  for ( int k = 0; k < numthreads; k++) {
    fprintf(stdout,  "[%d] Thread %d,      lptr[%d] = %p; rptr[%d] = %p, pptr[%d] = %p\n",
      thispid, k, k, lptr[k], k, rptr[k], k, pptr[k] );
  }

  for ( int k = 0; k < omp_num_t; k++) {
    /* write out the last element in each thread's result array */
    // output(k, lptr[k], nn, "initial l array");
    // output(k, rptr[k], nn, "initial r array");
    // output(k, pptr[k], nn, "initial p array");
  }
  fprintf(stdout, "\n");
#endif
}

/* initialize a double array with each element set to its index */
void
init(double *pp, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    pp[i] = (double) (i+1);
  }
}

/* write out various elements from a double array, with a label */
void
output( int threadnum, double *p, size_t size, const char *label)
{
  size_t i = size -1;
  size_t j = size/8;
  size_t k = size/16;
  fprintf(stdout, "    [%d] %s -- t %d, p[%zu]=%g; p[%zu]=%g; p[%zu]=%g; p[%zu]=%g; p[%zu]=%g\n",
    thispid, label, threadnum, 0UL, p[0], 1UL, p[1], k, p[k], j, p[j], i, p[i]);
}

/* check the elements of the p array */
void
checkdata(int threadnum, double *p, size_t size)
{
  int cnt = 0;
  for( int m = 0; m < size; m++) {
    /* check that the elements of the p array are all the same */
    if (p[m] != p[0])  {
      if ( cnt < 5) {
        fprintf(stderr, "    [%d] ERROR: thread %d: p[%d] (=%g) != p[0] (=%g)\n",
          thispid, threadnum, m, p[m], p[0]);
      }
      cnt++;
    }
  }
  // print the count of errors
  if (cnt != 0) {
    fprintf(stderr, "      [%d] ERROR: thread %d: erroneous data items %d; good data items = %d\n",
      thispid, threadnum, cnt, (int)size - cnt);
  } else {
    fprintf(stdout, "      [%d] SUCCESS: thread %d: good data items = %d\n",
      thispid, threadnum, (int)size - cnt);
  }
}

void
spacer(int timems, bool spin )
{
  if (spin == false ){ 
    // convert the integer millisecond argument to a timespec
    const struct timespec tspec = {0, (long) timems * 10000000 };

    // sleep for that amount of time
    int ret = nanosleep( &tspec, NULL);
    if (ret != 0) {
      fprintf(stdout, "[%d] nanosleep interrupted\n", thispid );
    }

  } else {
    // burn CPU which will be visible in the traced callstacks

    int	j,k;	/* temp values for loops */
    volatile float	x;	/* temp variable for f.p. calculation */
    long long count = 0;

    for (k= 0; k < 50; k++) {
      x = 0.0;
      for(j=0; j<1000000; j++) {
        x = x + 1.0;
      }
      count++;
    }
  }
}

/* =============================================================== */
/*  Routines for high-resolution timers */

hrtime_t
gethrvtime(void)
{
  int r;
  struct timespec tp;
  hrtime_t rc = 0;
  
  r =clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tp);
  if (r == 0) {
      rc = ((hrtime_t)tp.tv_sec)*1000000000 + (hrtime_t)tp.tv_nsec; 
  }

  return rc;
}

/* generic gethrtime() -- using clock_gettime(CLOCK_MONOTONIC, ...), and reformatting */
/*
 *  CLOCK_MONOTONIC
 *  Clock that cannot be set and represents monotonic time since some
 *           unspecified starting point.
 */

hrtime_t
gethrtime(void)
{
    int r;
    struct timespec tp;
    hrtime_t rc = 0;

    r =clock_gettime(CLOCK_MONOTONIC, &tp);
    if (r == 0) {
        rc = ((hrtime_t)tp.tv_sec)*1000000000 + (hrtime_t)tp.tv_nsec; 
    }

    return rc;
}

#if 0
static	char	*prhrdelta(hrtime_t);
static	char	*prhrvdelta(hrtime_t);

/* hrtime routines */
int
whrvlog(hrtime_t delta, hrtime_t vdelta, char *event, char *string)
{
	char	buf[1024];
	int	bytes;

	if(string == NULL) {
		sprintf(buf,
			"  %s wall-secs., %s CPU-secs., in %s\n",
			prhrdelta(delta),
			prhrvdelta(vdelta),
			event);
	} else {
		sprintf(buf,
			"  %s wall-secs., %s CPU-secs., in %s\n\t%s\n",
			prhrdelta(delta),
			prhrvdelta(vdelta),
			event, string);
	}

	bytes = fprintf(stdout, "%s", buf);
	return bytes;
}


/*	prhrdelta (hrtime_t delta)
 *		returns a pointer to a static string in the form:
 *		sec.nanosecs
 *		  1.123456789
 *		0123456789012
 *
 *	prhrvdelta is the same, but uses a different static buffer
 */

static	char	*
prhrdelta(hrtime_t delta)
{
	static	char	cvdbuf[26];
	double	tempus;

	/* convert to seconds */
	tempus = ( (double) delta) / (double)1000000000.;
	sprintf(cvdbuf, "%13.9f", tempus);
	return(cvdbuf);
}

static	char	*
prhrvdelta(hrtime_t delta)
{
	static	char	cvdbuf[26];
	double	tempus;

	/* convert to seconds */
	tempus = ( (double) delta) / (double)1000000000.;
	sprintf(cvdbuf, "%13.9f", tempus);
	return(cvdbuf);
}

#endif
