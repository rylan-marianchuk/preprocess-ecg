all: PreProcess

PreProcess: main.cpp wh5.o cudaKernel.o
	g++ -std=gnu++17 -o PreProcess -Ilib/pugixml-1.11/src -Ilib/b64 `pkg-config --cflags hdf5-serial` main.cpp wh5.o cudaExtract.o file_link.o `pkg-config --libs hdf5-serial` -lhdf5_cpp lib/libbase64.a lib/libsqlitewrap.a -lsqlite3 -lcudadevrt -lcudart
	mv *.o build/

wh5.o: lib/writeh5.cpp
	g++ -c `pkg-config --cflags hdf5-serial` lib/writeh5.cpp `pkg-config --libs hdf5-serial` -lhdf5_cpp -o wh5.o

cudaKernel.o: lib/cudaExtract.cu
	nvcc -rdc=true -c lib/cudaExtract.cu
	nvcc -dlink -o file_link.o cudaExtract.o -lcudadevrt -lcudart

clean:
	rm build/*.o

