CFLAGS=-O4 -g `pkg-config --cflags opencv`
LDFLAGS=`pkg-config --libs opencv`
PROJECT=encode

all:
	gcc -o $(PROJECT) main.c $(CFLAGS) $(LDFLAGS)

clean:
	rm -rf $(PROJECT)
