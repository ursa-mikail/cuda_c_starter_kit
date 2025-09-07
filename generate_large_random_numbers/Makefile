CC=nvcc
CFLAGS=-O3
LIBS=-lgmp

all: gen_large_rand

gen_large_rand: generate_large_random_numbers.cu
	$(CC) $(CFLAGS) generate_large_random_numbers.cu -o gen_large_rand $(LIBS)

clean:
	rm -f gen_large_rand

