NVCC=nvcc
# CXX=g++
CXX=$(NVCC)
LD=${CXX}
NVCCFLAGS += -O3 -arch=sm_70 -I${HDF5_ROOT}/include
CXXFLAGS+=-O3 -I${HDF5_ROOT}/include
LDFLAGS+=-lm $(CXXFLAGS) -L${HDF5_ROOT}/lib -lhdf5

OBJS= main.o swe.o xdmf_writer.o

all: swe

%.o:%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

swe: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f swe xdmf *.o *~
