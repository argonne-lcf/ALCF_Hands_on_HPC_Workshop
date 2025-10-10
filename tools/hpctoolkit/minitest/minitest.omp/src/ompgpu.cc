// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

#include <string.h>
#include <omp.h>
#include "minitest.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

int checkxfers();

int numdev;

void
twork( int iter, int threadnum)
{
  double *d_l1 = lptr[threadnum];
  double *d_r1 = rptr[threadnum];
  double *d_p1 = pptr[threadnum];
  int kernmax = kkmax;

#if 0
  fprintf(stdout, "[%d] Iteration %3d,   d_l1[%d] = 0x%016llx;   d_r1[%d] = 0x%016llx;   d_p1[%d] = 0x%016llx\n",
    thispid, iter, threadnum, d_l1, threadnum, d_r1, threadnum, d_p1 );
#endif
  hrtime_t starttime = gethrtime();

  // Distribute work across GPU devices using round-robin assignment
  // If MPI is used, consider both thread number and MPI rank; otherwise, use only thread number
  int rank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  int gpu_index = (threadnum + rank) % numdev;

  // Set the target device for this thread's computations.
  // This ensures that the subsequent OpenMP target region executes on the selected GPU.
  omp_set_default_device(gpu_index);

  // int threadsPerBlock = 256;
  // int  blocksPerGrid = ( nn + threadsPerBlock -1 ) / threadsPerBlock;
  // #pragma omp teams num_teams(blocksPerGrid) thread_limit(threadsPerBlock)

  #pragma omp target map(to:d_l1[0:nn], d_r1[0:nn]) map(tofrom: d_p1[0:nn])
  #pragma omp teams
  {
    #pragma omp distribute parallel for
    for (size_t i = 0; i < nn; ++i) {
      size_t nelements = nn;

#include "compute.h"

    }
  }

  hrtime_t endtime = gethrtime();
  double  tempus =  (double) (endtime - starttime) / (double)1000000000.;
#if 1
  fprintf(stdout, "    [%d] Completed iteration %d, thread %d on GPU %d in %13.9f s.\n",
    thispid, iter, threadnum, gpu_index, tempus);
#endif
  spacer (50, true);
}

void
initgpu()
{
  hrtime_t initstart = gethrtime();
  double  tempus =  (double) (initstart - starttime) / (double)1000000000.;
#if 1
  fprintf(stdout, "    [%d] Started initgpu() at timestamp %13.9f s.\n",
    thispid, tempus);
#endif

  /* determine number of GPU's */
  numdev = omp_get_num_devices();
  fprintf (stdout, "    [%d] Machine has %d GPU device%s\n", thispid, numdev, (numdev==1 ? "" : "s") );
  if ( numdev == 0 ){
    fprintf (stderr, "    [%d] ERROR: ompoffloading, but no GPU present", thispid );
    exit  (-1);
  }

#if 0
// We don't understand when this is supposed to to work
  /* Test if GPU is available */
  int	idev = omp_is_initial_device();

  int runningOnGPU = -1;
  #pragma omp target map(from:runningOnGPU)
  {
    runningOnGPU = omp_is_initial_device();
  }

  /* If still running on CPU, GPU must not be available */
  if (runningOnGPU != 0) {
#ifndef IGNORE_BAD_INITIAL_DEVICE
    fprintf(stderr, " [%d] ERROR: bad initial device! idev = %d, runningOnGpU -- omp_is_initial_device() = %d; exiting\n",
      thispid, idev, runningOnGPU);
    exit(1);
#else
    fprintf(stdout, " [%d] ignoring error bad initial device! idev = %d, runningOnGpU -- omp_is_initial_device() = %d; trying anyway\n",
      thispid, dev, runningOnGPU );
#endif
  } else {
    fprintf(stdout, "   [%d] gputest is able to use the GPU! idev = %d, runningOnGpU -- omp_is_initial_device() = %d\n",
      thispid, idev, runningOnGPU );
  }
#endif

  int ret = checkxfers();
  if (ret != 0 ) {
    fprintf(stdout, "[%d] Return from checkxfers = %d\n", thispid, ret);
  }

  hrtime_t initdone = gethrtime();
  tempus =  (double) (initdone - starttime) / (double)1000000000.;
#if 1
  fprintf(stdout, "    [%d] Leaving initgpu() at timestamp %13.9f s.\n",
    thispid, tempus);
#endif
}

int
checkxfers()
{
#if 0
  // define original host values
  int origto = 11;
  int origfrom = 13;
  int origtofrom = 17;

  //define values the gpu will set
  int gputo = 4;
  int gpufrom = 5;
  int gputofrom = 6; 

  int to = origto;
  int from = origfrom;
  int tofrom = origtofrom;

  fprintf(stdout, "[%d] ON HOST before: to = %02d, from = %02d, tofrom = %02d\n", thispid, to, from, tofrom);

  #pragma omp target map (to:to) map(from:from) map(tofrom:tofrom)
  {
    // Note that if this "printf(..." and the one below are changed to "fprintf(stdout, ...",
    // the compile fails with a link error.
    //    For now, disable this check, and rely on the HOST checks below.
    // printf("[%d] ON GPU: enter:  to = %02d, from = %02d, tofrom = %02d\n", thispid, to, from, tofrom); 

    to = gputo;
    from = gpufrom;
    tofrom = gputofrom; 

    // printf("[%d] ON GPU: exit:   to = %02d, from = %02d, tofrom = %02d\n", thispid, to, from, tofrom); 
  }

  fprintf(stdout, "[%d] ON HOST after:  to = %02d, from = %02d, tofrom = %02d\n", thispid, to, from, tofrom);
  fprintf(stdout, "[%d]  EXPECTED:      to = %02d, from = %02d, tofrom = %02d\n", thispid, origto, gpufrom, gputofrom);
#else
  fprintf(stdout, "[%d] checking of GPU transfers disabled\n", thispid);
#endif

  return 0;
}
