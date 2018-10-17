# Makefile for 159.735 Assignment 3
#

CPP = g++

# Use this for your CUDA programs
NVCC = nvcc

# FLAGS for Linux
CFLAGS = -w -O3

# Locally compiled modules
OBJS = fitsfile.o lenses.o

# Link to CFITSIO libraries - modify these accordingly
#LIBP = -L/export/home/iabond/cfitsio
#INCP = -I/export/home/iabond/cfitsio

# Link to CFITSIO libraries - modify these paths accordingly
LIBP = -L/home/dale/Documents/parallel/parallel-a3/a3/cfitsio
INCP = -I/home/dale/Documents/parallel/parallel-a3/a3/cfitsio

LIBS = -lcfitsio -lm

MODS = $(INCP) $(LIBP) $(LIBS) $(OBJS) 

BINS = a4

all : $(BINS)

clean :
	rm -f $(BINS)
	rm -f *.o

a4 : $(OBJS)
	$(NVCC) -o main a4.cu $(MODS)

# Demo program. Add more programs by making entries similar to this
lens_demo : lens_demo.cpp $(OBJS)
	${CPP} $(CFLAGS) -o lens_demo lens_demo.cpp $(MODS)

# Modules compiled and linked separately
fitsfile.o : fitsfile.cpp fitsfile.h
	${CPP} $(CFLAGS) $(INCP) -c fitsfile.cpp

lenses.o : lenses.cpp lenses.h
	${CPP} $(CFLAGS) $(INCP) -c lenses.cpp

