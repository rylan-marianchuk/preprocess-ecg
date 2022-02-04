build = build/
src = preprocess/


run: main.cpp
	g++ -std=gnu++17 -Ilib/pugixml-1.11/src -Ilib/b64 `pkg-config --cflags hdf5-serial` main.cpp `pkg-config --libs hdf5-serial` -lhdf5_cpp lib/libbase64.a -o run


clean:
	rm build/*
