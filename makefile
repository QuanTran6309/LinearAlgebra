CXX      := g++
NVCC     := nvcc

CXXFLAGS := -std=c++17 -O2 -fPIC -Iinclude
NVCCFLAGS:= -std=c++17 -O2 -Xcompiler -fPIC -Iinclude

LDFLAGS  := -shared
LIBS     := -lcublas -lcudart

SRC_DIR  := src
OBJ_DIR  := build
TARGET   := liblinearalgebra.so

# Recursive search
CPP_SRCS := $(shell find $(SRC_DIR) -name "*.cpp")
CU_SRCS  := $(shell find $(SRC_DIR) -name "*.cu")

# Convert src/... to build/... 
CPP_OBJS := $(CPP_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CU_OBJS  := $(CU_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

OBJS     := $(CPP_OBJS) $(CU_OBJS)

all: $(TARGET)

# Create directory tree automatically
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean


#g++ main.cpp -Iinclude -L. -llinearalgebra -o main
