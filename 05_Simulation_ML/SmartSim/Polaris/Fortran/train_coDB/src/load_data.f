      module ssim

      use smartredis_client, only : client_type      

      type(client_type) :: client
      integer nnDB

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

      program  data_loader

      use iso_c_binding
      use fortran_c_interop
      use ssim
      use, intrinsic :: iso_fortran_env
      implicit none
      include "mpif.h"

      real*8, allocatable, dimension (:,:) :: sendArr
      real*8 xmin, xmax, arrMLrun(2), x
      integer nSamples, nInputs, nOutputs, seed
      integer its, numts, i, err
      integer stepInfo(2), arrInfo(6)
      integer myrank, comm_size, ierr, tag, status(MPI_STATUS_SIZE), 
     &        nproc
      character*255 rank_key, sendArr_key
      character*255 ssdb

c     Initialize MPI
      call MPI_INIT(ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, comm_size, ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
      nproc = comm_size
      call MPI_Barrier(MPI_COMM_WORLD,ierr)
      write(*,*) 'Hello from rank ',myrank
      flush(OUTPUT_UNIT)

c     Initialize SmartRedis clients
      nnDB = 1 
      call init_client(myrank)
      call MPI_Barrier(MPI_COMM_WORLD,ierr)
      if (myrank.eq.0) write(*,*) 'All SmartRedis clients initialized'

c     Write the DB IP address to file
      call get_environment_variable("SSDB", ssdb)
      if (myrank.eq.0) then
         open (unit=25, file='SSDB.dat', status='replace')
         write(25,'(a)',advance='no') trim(adjustl(ssdb))
         close(25)
      endif

c     Set parameters for array of random numbers to be set as training data
c     In this example we create training data for a simple function
c     y=f(x), which has 1 input (x) and 1 output (y)
c     The domain for the function is from 0 to 10
c     Tha training data is obtained from a uniform distribution over the domain
      nSamples = 64
      nInputs = 1
      nOutputs = 1
      allocate(sendArr(nSamples,nInputs+nOutputs))
      seed = myrank+1
      call RANDOM_SEED(seed)
      xmin = 0.0
      xmax = 10.0


c     Send array used to communicate whether to keep running data loader or ML
      if (myrank.eq.0) then
         arrMLrun = 1.0
         err = client%put_tensor("check-run", arrMLrun, shape(arrMLrun))
         if (err.ne.0) write(*,*)
     &            "ERROR: client%put_tensor failed on rank ",myrank
      endif


c     Send some information regarding the training data size
      if (myrank.eq.0) then
         arrInfo(1) = nSamples
         arrInfo(2) = nInputs+nOutputs
         arrInfo(3) = nInputs
         arrInfo(4) = nproc
         arrInfo(5) = nproc
         arrInfo(6) = myrank
         err = client%put_tensor("sizeInfo", arrInfo, shape(arrInfo))
         if (err.ne.0) then 
            write(*,*) "ERROR: client%put_tensor failed on rank ",myrank
         else
            write(*,*) "Sent size info of training data to database"
         endif
      endif


c     Generate first part of the key for the training data
c     The key will be tagged with the rank ID and the time step number
      rank_key = "y."
      if (myrank.lt.10) then
         write (rank_key, "(A2,I1)") trim(rank_key), myrank
      elseif (myrank.lt.100) then
         write (rank_key, "(A2,I2)") trim(rank_key), myrank
      elseif (myrank.lt.1000) then
         write (rank_key, "(A2,I3)") trim(rank_key), myrank
      endif


c     Emulate integration of PDEs with a do loop
      numts = 1000
      do its=1,numts
         ! sleep for a few seconds to emulate the time required by PDE integration
         call sleep (10)

         ! first off check if ML is done training, if so exit from loop
         err = client%unpack_tensor("check-run", arrMLrun, 
     &                              shape(arrMLrun))
         if (err.ne.0) write(*,*) 
     &           "ERROR: client%unpack_tensor failed on rank ",myrank
         if (arrMLrun(1).lt.0.5) exit

         ! generate the training data for the polynomial y=f(x)=x**2 + 3*x + 1
         ! place output in first column, input in second column
         do i=1,nSamples
            call RANDOM_NUMBER(x)
            x = xmin + (xmax-xmin)*x
            sendArr(i,2) = x
            sendArr(i,1) = x**2 + 3*x +1
         enddo

         ! Append the ts number to the key so data doesn't get overwritten in database
         if (myrank.lt.10) then
            write (sendArr_key, "(A3,A1,I0)") trim(rank_key),'.',its
         elseif (myrank.lt.100) then
            write (sendArr_key, "(A4,A1,I0)") trim(rank_key),'.',its
         elseif (myrank.lt.1000) then
            write (sendArr_key, "(A5,A1,I0)") trim(rank_key),'.',its
         endif
         ! send training data to database
         if (myrank.eq.0) write(*,*) 
     &            'Sending training data to database with key ',
     &            trim(sendArr_key), ' and shape ',
     &            shape(sendArr)
         err = client%put_tensor(trim(sendArr_key), sendArr, 
     &                           shape(sendArr))
         call MPI_Barrier(MPI_COMM_WORLD,ierr)
         if (err.ne.0) then 
            write(*,*) "ERROR: client%put_tensor failed on rank ",myrank
         else
            if (myrank.eq.0) write(*,*) 
     &                'All ranks finished sending training data'
         endif
         ! Send also the time step number, used by ML program to determine 
         ! when new training data is available
         if (myrank.eq.0) then
            stepInfo(1) = its
            stepInfo(2) = 0.0
            err = client%put_tensor("step", stepInfo, shape(stepInfo))
            if (err.ne.0) write(*,*)
     &             "ERROR: client%put_tensor failed on rank ",myrank
         endif
      enddo
c     Finilization stuff
      if (myrank.eq.0) write(*,*) "Exiting ... "
      deallocate(sendArr)
      call MPI_FINALIZE(ierr)
      end program data_loader
