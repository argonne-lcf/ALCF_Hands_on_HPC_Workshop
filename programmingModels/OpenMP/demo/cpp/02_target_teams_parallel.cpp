#include <stdio.h>
#include <omp.h>

int main( int argv, char** argc ) {
#pragma omp target teams distribute parallel for simd
  for(int i=0;i<10000;i++)
    {
      printf( "Thread %d out of %d threads in team %d out of %d teams is using index %d\n", 
	      omp_get_thread_num(), omp_get_num_threads(), 
	      omp_get_team_num(), omp_get_num_teams(), i );
    }

return 0;
}
