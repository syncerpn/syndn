ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/

CC=gcc
CPP=g++
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
LDFLAGS= -lm -pthread `pkg-config --libs opencv` -lstdc++ -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn
COMMON= -Iinclude/ -Isrc/ -DOPENCV `pkg-config --cflags opencv` -DGPU -I/usr/local/cuda/include/ -DCUDNN
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -fopenmp -Ofast -DOPENCV -DGPU -DCUDNN

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o 		\
	activations.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o 						\
	softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o 		\
	option_list.o route_layer.o upsample_layer.o box.o 										\
	normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o 			\
	logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o 		\
	batchnorm_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o 					\
	yolo_layer.o iseg_layer.o image_opencv.o quantization_layer.o							\
	priorbox_layer.o multibox_layer.o diff_layer.o initializer.o 							\
	convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o \
	col2im_kernels.o blas_kernels.o crop_kernels.o dropout_kernels.o diff_kernels.o			\
	maxpool_kernels.o avgpool_kernels.o quantization_kernels.o batchnorm_kernels.o			\
	convolutional_weight_transform.o priorbox_kernels.o multibox_kernels.o yolo_kernels.o 	\
	channel_selective_layer.o channel_selective_kernels.o 									\
	detection_layer.o region_layer.o image_kernels.o 										\
	partial_layer.o partial_kernels.o 														\

EXECOBJA=lsd.o segmenter.o regressor.o classifier.o detector.o nightmare.o 					\
		instance-segmenter.o darknet.o depth_compress.o auto_colorize.o	slimmable_net.o		\

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

all: obj backup results info $(SLIB) $(ALIB) $(EXEC)

$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results
info:
	mkdir -p info

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*