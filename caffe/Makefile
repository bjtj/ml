CXXFLAGS=-I${CAFFE_ROOT}/include -I${CAFFE_ROOT}/build/src -I${C_INCLUDE_PATH}
LDFLAGS=-L${CAFFE_ROOT}/build/lib -lcaffe -lglog -lboost_system

.PHONY: all

all:
	g++ -g -Wall -o cpp_xor cpp_xor.cpp ${CXXFLAGS} ${LDFLAGS}

clean:
	rm -rf cpp_xor
