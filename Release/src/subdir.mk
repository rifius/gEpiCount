################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/gEpiCount.cpp \
../src/version.cpp 

OBJS += \
./src/gEpiCount.o \
./src/version.o 

CPP_DEPS += \
./src/gEpiCount.d \
./src/version.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.0/bin/nvcc -O3 -Xcompiler -fopenmp -Xptxas -v -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.0/bin/nvcc -O3 -Xcompiler -fopenmp -Xptxas -v --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


