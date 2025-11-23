#include <cstdlib>
#include <driver_types.h>
#include<iostream>

inline void cudaCheckImpl(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at "
                  << file << ":" << line << "\n"
                  << "Code: " << error << ", Reason: "
                  << cudaGetErrorString(error) << std::endl;

        throw cudaError(error);
    }
}

#define cudaCheck(call)  cudaCheckImpl((call), __FILE__, __LINE__)

int main()
{
    std::cout<<"Starting check_device_info example..." << std::endl;

    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count));

    if(device_count == 0){
        std::cout<<"No CUDA capable devices found!"<<std::endl;
        return -1;
    }
    else {
        std::cout<<"Found "<< device_count <<" CUDA capable device(s)." <<std::endl;
    }

    //选择计算能力最优的设备
    int best_device = 0;
    if(device_count > 1){
        int max_multiprocessors = 0;
        for(int dev=0; dev<device_count; dev++){
            cudaDeviceProp device_prop;
            cudaCheck(cudaGetDeviceProperties(&device_prop, dev));

            if(device_prop.multiProcessorCount > max_multiprocessors){
                max_multiprocessors = device_prop.multiProcessorCount;
                best_device = dev;
            }
        }
        std::cout<<"Using device "<< best_device << " as the best device."<<std::endl;
    }

    cudaCheck(cudaSetDevice(best_device));

    cudaDeviceProp device_prop;
    cudaCheck(cudaGetDeviceProperties(&device_prop, best_device));
    std::cout<<"Device "<< best_device << " name: "<< device_prop.name << std::endl;

    int driver_version = 0, runtime_version = 0;

    cudaCheck(cudaDriverGetVersion(&driver_version));
    cudaCheck(cudaRuntimeGetVersion(&runtime_version));

    std::cout<<"CUDA Driver Version / Runtime Version: "
             << driver_version/1000 << "."<<(driver_version%100)/10
             <<" / "<< runtime_version/1000 << "."<<(runtime_version%100)/10 << std::endl;

    std::cout<<"CUDA Capability Major/Minor version number: "
             << device_prop.major << "." << device_prop.minor << std::endl;

    std::cout<<"Total amount of global memory: "
             << static_cast<float>(device_prop.totalGlobalMem)/(1024*1024)
             <<" MB"<< std::endl;

    std::cout<<"Multiprocessor count: "
             << device_prop.multiProcessorCount << std::endl;
            
    std::cout<<"GPU Clock rate: "
             << device_prop.clockRate * 1e-3f << " MHz" << std::endl;

    std::cout<<"Memory Clock rate: "
             << device_prop.memoryClockRate * 1e-3f << " MHz" << std::endl;

    std::cout<<"Memory Bus Width: "
             << device_prop.memoryBusWidth << " bits" << std::endl;

    if(device_prop.l2CacheSize){
        std::cout<<"L2 Cache Size: "
                 << device_prop.l2CacheSize << " bytes" << std::endl;
    }

    std::cout<<"Max Texture Dimension Size (x,y,z) "
    "1D = "<< device_prop.maxTexture1D << " "
    "2D = ("<< device_prop.maxTexture2D[0] <<", "<< device_prop.maxTexture2D[1] <<") "
    "3D = ("<< device_prop.maxTexture3D[0] <<", "
              << device_prop.maxTexture3D[1] <<", "
              << device_prop.maxTexture3D[2] <<") "<< std::endl;

    std::cout<<"Max Layered Texture Size (dim) "
    "1D = "<< device_prop.maxTexture1DLayered[0] << " "
    "2D = ("<< device_prop.maxTexture2DLayered[0] <<", "
              << device_prop.maxTexture2DLayered[1] <<", "
              << device_prop.maxTexture2DLayered[2] <<") "<< std::endl;

    std::cout<<"Total amount of constant memory: "
             << device_prop.totalConstMem << " bytes" << std::endl;

    std::cout<<"Total amount of shared memory per block: "
             << device_prop.sharedMemPerBlock << " bytes" << std::endl;

    std::cout<<"Total number of registers available per block: "
             << device_prop.regsPerBlock << std::endl;

    std::cout<<"Warp size: "
             << device_prop.warpSize << std::endl;

    std::cout<<"Max number of threads per multiprocessor: "
                << device_prop.maxThreadsPerMultiProcessor << std::endl;

    std::cout<<"Max number of threads per block: "
                << device_prop.maxThreadsPerBlock << std::endl;

    std::cout<<"Max sizes of each dimension of a block: "
    "x = "<< device_prop.maxThreadsDim[0] << " "
    "y = "<< device_prop.maxThreadsDim[1] << " "
    "z = "<< device_prop.maxThreadsDim[2] << std::endl;

    std::cout<<"Max sizes of each dimension of a grid: "
    "x = "<< device_prop.maxGridSize[0] << " "
    "y = "<< device_prop.maxGridSize[1] << " "
    "z = "<< device_prop.maxGridSize[2] << std::endl;

    std::cout<<"Maxinum memory pitch: "
             << device_prop.memPitch << " bytes" << std::endl;

    exit(EXIT_SUCCESS);
}