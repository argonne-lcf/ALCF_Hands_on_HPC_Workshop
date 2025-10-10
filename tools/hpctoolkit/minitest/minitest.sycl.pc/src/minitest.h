/*
 * SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//  This file defines the interface between the various front-ends to minitest
//	and the backends providing the computations, and/or GPU offloading
//  It also defines various timing routines

#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <time.h>

extern size_t nn;
extern int kkmax;
extern int kscale;
extern int omp_num_t;
extern pid_t thispid;
extern int mpi_rank;

extern double **lptr;
extern double **rptr;
extern double **pptr;

extern bool run_post_rept;

extern void setup_run(int argc, char** argv);
extern void mpisync(void);
extern void teardown_run(void);
 
extern void allocinitdata(int numthreads);
extern void init(double *pp, size_t size);
extern void output( int threadnum, double *p, size_t size, const char *label );
extern void checkdata( int threadnum, double *p, size_t size );
extern void spacer(int time, bool spin);

/* routine to determine if GPU is available, and how many devices */
void initgpu();

/* routine to do the off-loaded work on the GPU */
void twork(int iter, int threadnum);

/* Timing routines */
typedef long long  hrtime_t;
extern hrtime_t gethrtime();
extern hrtime_t gethrvtime();

extern hrtime_t starttime;
