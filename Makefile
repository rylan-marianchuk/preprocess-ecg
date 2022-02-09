build = build/
src = preprocess/


#run: main.cpp writeh5.cpp writeh5.h writeSQL.cpp writeSQL.h
	#g++ -std=gnu++17 -Ilib/pugixml-1.11/src -Ilib/b64 `pkg-config --cflags hdf5-serial` main.cpp writeh5.cpp writeSQL.cpp `pkg-config --libs hdf5-serial` -lhdf5_cpp -lsqlite3 lib/libbase64.a  -o run
	#g++ -std=gnu++17 -Ilib/pugixml-1.11/src -Ilib/b64 main.cpp wh5.o lib/libbase64.a -o run

all: PreProcess

PreProcess: main.o wh5.o sqlWriting.o cudaKernel.o
	g++ -o PreProcess -Ilib/pugixml-1.11/src -Ilib/b64 `pkg-config --cflags hdf5-serial` main.o wh5.o sqlWriting.o cudaExtract.o file_link.o `pkg-config --libs hdf5-serial` -lhdf5_cpp lib/libbase64.a -lsqlite3 -lcudadevrt -lcudart

main.o:
	g++ -c -std=gnu++17 -Ilib/pugixml-1.11/src -Ilib/b64 main.cpp -llib/libbase64.a -o main.o

wh5.o: writeh5.cpp writeh5.h
	g++ -c `pkg-config --cflags hdf5-serial` writeh5.cpp `pkg-config --libs hdf5-serial` -lhdf5_cpp -o wh5.o

sqlWriting.o: writeSQL.cpp writeSQL.h
	g++ -c writeSQL.cpp -lsqlite3 -o sqlWriting.o

cudaKernel.o: cudaExtract.cu cudaExtract.cuh
	nvcc -rdc=true -c cudaExtract.cu
	nvcc -dlink -o file_link.o cudaExtract.o -lcudadevrt -lcudart

clean:
	rm build/*
