program  main
  use omp_lib
  implicit none
  integer i
!$omp target teams distribute parallel do thread_limit(2) num_teams(2)
    do i=1,8
       write(*,*) "Thread", omp_get_thread_num(), &
            "out of", omp_get_num_threads() ,&
            "threads in team", omp_get_team_num(), &
            "out of", omp_get_num_teams(), &
            "teams is using index" , i
     end do
!$omp end target teams distribute parallel do
end program main
