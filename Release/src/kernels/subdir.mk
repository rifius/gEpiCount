################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/kernels/entropyD.cu \
../src/kernels/fisher_d.cu \
../src/kernels/kernel1.cu \
../src/kernels/misc1.cu 

CU_DEPS += \
./src/kernels/entropyD.d \
./src/kernels/fisher_d.d \
./src/kernels/kernel1.d \
./src/kernels/misc1.d 

OBJS += \
./src/kernels/entropyD.o \
./src/kernels/fisher_d.o \
./src/kernels/kernel1.o \
./src/kernels/misc1.o 


# Each subdirectory must supply rules for building sources it contributes
src/kernels/%.o: ../src/kernels/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.0/bin/nvcc -O3 -Xcompiler -fopenmp -Xptxas -v -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  -odir "src/kernels" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.0/bin/nvcc --device-c -O3 -Xcompiler -fopenmp -Xptxas -v -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=compute_32 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


