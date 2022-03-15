build = build/
src = preprocess/

all: PreProcess

PreProcess: main.o wh5.o cudaKernel.o
	g++ -o PreProcess -Ilib/pugixml-1.11/src -Ilib/b64 `pkg-config --cflags hdf5-serial` main.o wh5.o cudaExtract.o file_link.o `pkg-config --libs hdf5-serial` -lhdf5_cpp lib/libbase64.a lib/libsqlitewrap.a -lsqlite3 -lcudadevrt -lcudart

main.o:
	g++ -c -std=gnu++17 -Ilib/sqlitewrap -Ilib/pugixml-1.11/src -Ilib/b64 main.cpp -llib/libbase64.a -llib/libsqlitewrap.a -o main.o

wh5.o: writeh5.cpp writeh5.h
	g++ -c `pkg-config --cflags hdf5-serial` writeh5.cpp `pkg-config --libs hdf5-serial` -lhdf5_cpp -o wh5.o

cudaKernel.o: cudaExtract.cu cudaExtract.cuh
	nvcc -rdc=true -c cudaExtract.cu
	nvcc -dlink -o file_link.o cudaExtract.o -lcudadevrt -lcudart

clean:
	rm *.o
	rm wvfm_params.db
	rm unparsable.txt

