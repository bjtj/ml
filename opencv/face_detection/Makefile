PROJECT=detect

all:
	g++ -o $(PROJECT) main.cpp `pkg-config --cflags --libs opencv`

clean:
	rm -rf $(PROJECT)
