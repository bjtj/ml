PROJECT=decode
CXXFLAGS=`pkg-config --cflags opencv`
LDFLAGS=`pkg-config --libs opencv`

all:
	g++ -o $(PROJECT) src/main.cpp $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -rf $(PROJECT)
