
all: matrix/mynative.class libmynative.so libmynative.S matrix/Matrix.class matrix.jar

matrix/mynative.class: mynative.scala
	scalac mynative.scala

matrix_mynative.h: matrix/mynative.class
	javah matrix.mynative

libmynative.S: libmynative.so
	objdump -S libmynative.so >libmynative.S

libmynative.so: libmynative.c matrix_mynative.h
	gcc -Wall -O5 -march=native -std=c99 -fPIC -fopenmp -shared libmynative.c -o libmynative.so

matrix/Matrix.class: matrix.scala
	scalac matrix.scala

matrix.jar: matrix/Matrix.class matrix/mynative.class
	jar cvf matrix.jar matrix

clean:
	rm -f *.class matrix/*.class *.so matrix_mynative.h libmynative.S matrix.jar *~ *.log 
