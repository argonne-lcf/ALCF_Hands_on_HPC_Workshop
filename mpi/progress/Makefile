CC=mpicc
BINS=async1 async1-s async1-t

all: $(BINS)

async1-s: async1.c
	$(CC) $(CFLAGS) -DSLEEP -o async1-s async1.c

async1-t: async1.c
	$(CC) $(CFLAGS) -DTEST -o async1-t async1.c

clean:
	rm -f $(BINS)
