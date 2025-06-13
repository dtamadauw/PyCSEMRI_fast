


EIGEN_PATH = ./include/Eigen/
LM_PATH = ./include/
CFLAGS=-I./  -I$(EIGEN_PATH) -I$(LM_PATH)
LIBS=

CC=g++ -O3 -s -DNDEBUG
TARGET=libuwwfs
all:$(TARGET)
$(TARGET):libuwwfs.o fwFit_ComplexLS_1r2star.o fwFit_MagnLS_1r2star.o fwFit_MixedLS_1r2star.o
	$(CC) $(CFLAGS) -shared -o $(TARGET).so libuwwfs.o fwFit_ComplexLS_1r2star.o fwFit_MagnLS_1r2star.o fwFit_MixedLS_1r2star.o  $(LIBS)
libuwwfs.o:libuwwfs.cpp
	$(CC) $(CFLAGS) -c -fPIC libuwwfs.cpp -o libuwwfs.o
main.o:main.cpp
	$(CC) $(CFLAGS) -c -fPIC main.cpp -o main.o
fwFit_MagnLS_1r2star.o:fwFit_MagnLS_1r2star.cpp 
	$(CC) $(CFLAGS) -c -fPIC fwFit_MagnLS_1r2star.cpp -o fwFit_MagnLS_1r2star.o
fwFit_ComplexLS_1r2star.o:fwFit_ComplexLS_1r2star.cpp 
	$(CC) $(CFLAGS) -c -fPIC fwFit_ComplexLS_1r2star.cpp -o fwFit_ComplexLS_1r2star.o
fwFit_MixedLS_1r2star.o:fwFit_MixedLS_1r2star.cpp 
	$(CC) $(CFLAGS) -c -fPIC fwFit_MixedLS_1r2star.cpp -o fwFit_MixedLS_1r2star.o


install: $(TARGET)
	mkdir -p ../lib
	cp $(TARGET).so ../lib


clean:
	echo CLEAN
	rm -f ./*.o
	rm -f ./*.so


#$(CC) $(CFLAGS) -o $(TARGET) main.o fwFit_ComplexLS_1r2star.o ap.o optimization.o alglibinternal.o linalg.o alglibmisc.o solvers.o $(LIBS)