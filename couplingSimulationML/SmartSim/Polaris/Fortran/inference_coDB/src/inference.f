      module ssim

      use smartredis_client, only : client_type      

      type(client_type) :: client
      integer nnDB, ppn

      end module ssim

c ==================================================

      subroutine init_client(myrank)

      use iso_c_binding
      use ssim
      implicit none
      include "mpif.h"

      integer myrank, err

c     Initialize SmartRedis clients 
      if (myrank.eq.0) write(*,*) 'Initializing SmartRedis clients ... '
      if (nnDB.eq.1) then
         err = client%initialize(.false.) ! NOT using a clustered database (DB on 1 node only)
      else
         err = client%initialize(.true.) ! using a clustered database (DB on multiple nodes)
      endif
      if (err.ne.0) 
     &      write(*,*) "ERROR: client%initialize failed on rank ",myrank


      end subroutine init_client

c ==================================================

      program  inference

      use iso_c_binding
      use fortran_c_interop
      use ssim
      use, intrinsic :: iso_fortran_env
      implicit none
      include "mpif.h"

      real*8, allocatable, dimension (:,:) :: inf_data
      real*8, allocatable, dimension (:,:) :: pred_data
      real*8, allocatable, dimension (:,:) :: truth_data
      real*8, allocatable, dimension (:) :: read_inputs
      real*8 xmin, xmax, x, rank_fact
      integer nSamples, seed, num_inputs
      integer its, numts, i, j, err
      integer myrank, comm_size, ierr, tag, status(MPI_STATUS_SIZE), 
     &        nproc, name_len
      character*255 inf_key, pred_key, fname
      logical im_exst, exlog
      character(len=255), dimension(1) :: inputs
      character(len=255), dimension(1) :: outputs
      character*(MPI_MAX_PROCESSOR_NAME) proc_name

c     Initialize MPI
      call MPI_INIT(ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, comm_size, ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
      call MPI_Get_processor_name( proc_name, name_len, ierr)
      nproc = comm_size
      write(*,100) 'Hello from rank ',myrank,'/',nproc,
     &           ' on node ',trim(proc_name)
100   format (A,I0,A,I0,A,A)
      call MPI_Barrier(MPI_COMM_WORLD,ierr)
      flush(OUTPUT_UNIT)

c     Initialize SmartRedis client
      inquire(file='input.config',exist=exlog)
      if(exlog) then
         open (unit=24, file='input.config', status='unknown')
         read(24,*) num_inputs
         allocate(read_inputs(num_inputs))
         read(24,*) (read_inputs(j), j=1,num_inputs)
         close(24)
         nnDB = int(read_inputs(1))
         ppn = int(read_inputs(2))
         deallocate(read_inputs)
      else
         if (myrank.eq.0) then
            write(*,*) 'Inputs not specified in input.config'
            write(*,*) 'Setting nnDB and ppn to 1'
         endif
         nnDB = 1
         ppn = 1
      endif
      call init_client(myrank)
      call MPI_Barrier(MPI_COMM_WORLD,ierr)
      if (myrank.eq.0) write(*,*) 'All SmartRedis clients initialized'

c     Load the model on the database
      if (mod(myrank,ppn).eq.0) then
         err = client%set_model_from_file("model", "model_jit.pt",
     &                                   "TORCH", "GPU")
         if (err.eq.0) then
            write(*,*) "Uploaded model to database from rank ", myrank
         else 
            write(*,*) "ERROR: client%set_model_from_file failed ",
     &                 "on rank ",myrank
         endif
      endif
      call MPI_Barrier(MPI_COMM_WORLD,ierr)


c     Set parameters for array of random numbers to be set as inference data
c     In this example we create inference data for a simple function
c     y=f(x), which has 1 input (x) and 1 output (y)
c     The domain for the function is from 0 to 10
c     Tha inference data is obtained from a uniform distribution over the domain
      nSamples = 1024
      allocate(inf_data(nSamples,1))
      allocate(pred_data(nSamples,1))
      allocate(truth_data(nSamples,1))
      seed = myrank+1
      call RANDOM_SEED(seed)
      rank_fact = real(myrank/ppn)
      if (rank_fact.lt.1.0) then
         xmin = 0.0
         xmax = 5.0
      else
         xmin = 5.0
         xmax = 10.0
      endif


c     Generate the key for the inference data
c     The key will be tagged with the rank ID
      inf_key = "y."
      pred_key = "p."
      write (inf_key, "(A2,I0)") trim(inf_key), myrank
      write (pred_key, "(A2,I0)") trim(pred_key), myrank


c     Open file to write predictions
      if (mod(myrank,ppn).eq.0) then
         write(fname, "(A16,I0,A4)") "predictions_node",
     &                              myrank/ppn+1,".dat"
         open(unit=10,file = fname)
      endif


c     Emulate integration of PDEs with a do loop
      numts = 5
      do its=1,numts
         ! sleep for a few seconds to emulate the time required by PDE integration
         call sleep (5)

         ! generate the inference data for the polynomial y=f(x)=x**2 + 3*x + 1
         ! place output in first column, input in second column
         do i=1,nSamples
            call RANDOM_NUMBER(x)
            x = xmin + (xmax-xmin)*x
            inf_data(i,1) = x
            truth_data(i,1) = x**2 + 3*x +1
         enddo

         ! Send the inference data
         if (myrank.eq.0) write(*,*) 
     &            'Sending inference data to database with key ',
     &            trim(inf_key), ' and shape ',
     &            shape(inf_data)
         err = client%put_tensor(trim(inf_key), inf_data, 
     &                          shape(inf_data))
         if (err.ne.0) 
     &      write(*,*) "ERROR: client%put_tensor failed on rank ",myrank
         call MPI_Barrier(MPI_COMM_WORLD,ierr)
         if (myrank.eq.0) write(*,*) 'Finished sending inference data'

         ! Evaluate the model on the database
         inputs(1) = inf_key
         outputs(1) = pred_key
         err = client%run_model('model', inputs, outputs)
         if (err.ne.0) 
     &      write(*,*) "ERROR: client%run_model failed on rank ",myrank
         call MPI_Barrier(MPI_COMM_WORLD,ierr)
         if (myrank.eq.0) write(*,*) 'Finished evaluating model'

         ! Retreive the predictions
         err = client%unpack_tensor(trim(pred_key), pred_data,
     &                             shape(pred_data))
         if (err.ne.0) 
     &      write(*,*) "ERROR: client%unpack_tensor failed on ",
     &                 "rank ",myrank
         call MPI_Barrier(MPI_COMM_WORLD,ierr)
         if (myrank.eq.0) write(*,*) 'Finished retreiving predictions'

         ! Write the inference, prediction and truth data to file for plotting
         if (mod(myrank,ppn).eq.0) then
            do i=1,nSamples
               write(10,*) inf_data(i,1), pred_data(i,1), 
     &                     truth_data(i,1)
            enddo
         endif
      enddo


c     Finilization stuff
      if (myrank.eq.0) write(*,*) "Exiting ... "
      if (myrank.eq.0) close(10)
      deallocate(inf_data)
      deallocate(pred_data)
      deallocate(truth_data)
      call MPI_FINALIZE(ierr)

      end program inference
