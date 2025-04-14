CC = nvcc
CFLAGS = -std=c++17 -O3
LDFLAGS = -lpng -ljpeg

SRC = src/main.cpp src/image_processor.cpp src/cuda_kernels.cu
OBJ = $(SRC:.cpp=.o)
EXEC = image_processor

all: $(EXEC)

$(EXEC): $(OBJ)
    $(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
    $(CC) $(CFLAGS) -c -o $@ $<

clean:
    rm -f $(OBJ) $(EXEC)
