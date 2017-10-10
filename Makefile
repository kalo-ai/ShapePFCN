PROJECT := mvfcn

CONFIG_FILE := Makefile.config
# Explicitly check for the config file, otherwise make -k will proceed anyway.
ifeq ($(wildcard $(CONFIG_FILE)),)
$(error $(CONFIG_FILE) not found.)
endif
include $(CONFIG_FILE)

ifeq ($(RELEASE_BUILD_DIR),)
	RELEASE_BUILD_DIR := $(BUILD_DIR)_release
endif
ifeq ($(DEBUG_BUILD_DIR),)
	DEBUG_BUILD_DIR := $(BUILD_DIR)_debug
endif

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	BUILD_DIR := $(DEBUG_BUILD_DIR)
else
	BUILD_DIR := $(RELEASE_BUILD_DIR)
endif

# All of the directories containing code.
SRC_DIRS := .

##############################
# Get all source files
##############################
# SRCS are the source files excluding the test ones.
SRCS := $(shell find . -maxdepth 1 -name "*.cpp")
# source files needed to register caffe layers, TODO
# SRCS += src/caffe/layer_factory.cpp 

OBJS := $(addprefix $(BUILD_DIR)/, ${SRCS:.cpp=.o})
BIN := $(BUILD_DIR)/$(PROJECT).bin

##############################
# Derive include and lib directories
##############################
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include

THEA_INCLUDE_DIR := ./Thea/Code/Source

THEA_DEPS_INCLUDE_DIR := ./TheaDepsUnix/Source/Installations/include

CAFFE_INCLUDE_DIR := ./caffe-ours/include

HDF5_INCLUDE_DIR := /usr/include/hdf5/serial

INCLUDE_DIRS := $(THEA_INCLUDE_DIR)
INCLUDE_DIRS += $(THEA_DEPS_INCLUDE_DIR)
INCLUDE_DIRS += $(OTHER_INCLUDE_DIRS)
INCLUDE_DIRS += $(CAFFE_INCLUDE_DIR)
INCLUDE_DIRS += $(HDF5_INCLUDE_DIR)
ifneq ($(CPU_ONLY), 1)
	INCLUDE_DIRS += $(CUDA_INCLUDE_DIR)
endif


CUDA_LIB_DIR :=
# add <cuda>/lib64 only if it exists
ifneq ("$(wildcard $(CUDA_DIR)/lib64)","")
	CUDA_LIB_DIR += $(CUDA_DIR)/lib64
endif
CUDA_LIB_DIR += $(CUDA_DIR)/lib

THEA_LIBRARY_DIR := ./Thea/Code/Build/Output/lib

THEA_DEPS_LIBRARY_DIR := ./TheaDepsUnix/Source/Installations/lib/

CAFFE_LIBRARY_DIR := ./caffe-ours/build/lib

HDF5_LIBRARY_DIR := /usr/lib/x86_64-linux-gnu/hdf5/serial

LIBRARIES := glog gflags protobuf m hdf5_hl hdf5 leveldb snappy lmdb opencv_core opencv_highgui opencv_imgproc openblas caffe
LIBRARIES += boost_system boost_filesystem boost_regex boost_thread
LIBRARIES += Thea dl 3ds freeimageplus
# LIBRARIES += glut GL
LIBRARIES += OSMesa
LIBRARIES += gomp

LIBRARY_DIRS := $(THEA_LIBRARY_DIR)
LIBRARY_DIRS += $(THEA_DEPS_LIBRARY_DIR)
LIBRARY_DIRS += $(CAFFE_LIBRARY_DIR)
LIBRARY_DIRS += $(HDF5_LIBRARY_DIR)
ifneq ($(CPU_ONLY), 1)
	LIBRARY_DIRS += $(CUDA_LIB_DIR)
	LIBRARY_DIRS += $(OTHER_LIB_DIRS)
	LIBRARIES += cudart cublas curand
endif

ifeq ($(USE_CUDNN), 1)
	LIBRARIES += cudnn
endif

##############################
# Configure build
##############################

CXX ?= /usr/bin/g++
GCCVERSION := $(shell $(CXX) -dumpversion | cut -f1,2 -d.)
# older versions of gcc are too dumb to build boost with -Wuninitalized
ifeq ($(shell echo | awk '{exit $(GCCVERSION) < 4.6;}'), 1)
	WARNINGS += -Wno-uninitialized
endif

# Automatic dependency generation
CXXFLAGS := -MMD -MP -std=c++11 -Wall -Wno-sign-compare -Wno-reorder -Wno-strict-aliasing -Wno-unused-local-typedefs
CXXFLAGS += -pthread -fPIC -fopenmp
CXXFLAGS += -DUSE_OPENCV
CXXFLAGS += -DUSE_LEVELDB
CXXFLAGS += -DUSE_LMDB
ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG -g -O0
else
	CXXFLAGS += -DNDEBUG -O2
endif
ifeq ($(USE_CUDNN), 1)
	CXXFLAGS += -DUSE_CUDNN
endif
ifeq ($(CPU_ONLY), 1)
	CXXFLAGS += -DCPU_ONLY
endif
CXXFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -DTHEA_GL_OSMESA

LINKFLAGS := -Wl,-soname,-rpath
LINKFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
		$(foreach librarydir,$(LIBRARY_DIRS),-Wl,-rpath,$(librarydir)) \
		$(foreach library,$(LIBRARIES),-l$(library))

WARNS_EXT := warnings.txt

##############################
# Define build targets
##############################
.PHONY: all clean mkbuilddir

all: mkbuilddir compile link

mkbuilddir:  
	mkdir -p $(BUILD_DIR)

compile: $(OBJS)

link: $(BIN)

$(BUILD_DIR)/%.o: %.cpp 
	@ echo CXX $<
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)	
	@ cat $@.$(WARNS_EXT)
	
$(BIN): $(OBJS)
	@ echo CXX/LD $(OBJS) -o  $(BIN) $(LINKFLAGS) 
	$(CXX)  $(OBJS) -o  $(BIN) $(LINKFLAGS)

		
clean:
	@- $(RM) -rf $(BUILD_DIR)


