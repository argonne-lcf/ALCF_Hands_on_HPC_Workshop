program  main
  use omp_lib
  implicit none

  write(*,*) "Number of devices:", omp_get_num_devices()

  !$omp target
   if( .not. omp_is_initial_device() ) then
     write(*,*) "Hello world from accelerator"
   else
     write(*,*) "Hello world from host"
  endif
  !$omp end target

end program main
