///////////////////////////////////////////////////////////////////////////////
//  Build:
//    with hipcc:
//      hipcc copy_global_mem.cpp  -o copy_global_mem
///////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <vector>

#define HIP_ENABLE_PRINTF
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// Helper functions
#define GPU_ERR_CHECK(expr)                     \
    {                                           \
        gpu_assert((expr), __FILE__, __LINE__); \
    }

inline void gpu_assert(hipError_t e, const char* file, int line, bool abort = true)
{
    if(e)
    {
        const char* errName = hipGetErrorName(e);
        const char* errMsg  = hipGetErrorString(e);
        std::cerr << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": "
                  << std::endl
                  << errMsg << std::endl;
        exit(e);
    }
}

// Copy from global memory to global memory directly
template <typename T>
__global__ void copy_direct(const T* __restrict__ idata,
                            T* __restrict__ odata)
{
    int base = blockDim.x * blockIdx.x;
    int idx = base + threadIdx.x;

    odata[idx] = idata[idx];
}

template <typename T>
void execute_test(const int   len,
                  const int   batch,
                  const int   device_id = 0)
{
    size_t i_total_size  = len * batch;
    size_t o_total_size  = len * batch;
    size_t i_total_bytes = i_total_size * sizeof(T);
    size_t o_total_bytes = o_total_size * sizeof(T);
    size_t lds_bytes     = len * sizeof(T);

    GPU_ERR_CHECK(hipSetDevice(device_id));
    dim3 grid(batch);
    dim3 block(len);

    std::cout << "-------------------------------------------------"
              << "\nlen " << len << ", batch " << batch << std::endl;

    std::vector<T> in(i_total_size);
    std::vector<T> out(o_total_size);
    T *d_in, *d_out;

    std::cout << "Generate input...\n";
    for(auto i = 0; i < batch; i++)
        for(auto j = 0; j < len; j++)
        {
            in[i * len + j].x = i * len + j;
            std::cout << "input is " << in[i * len + j].x << std::endl;
        }

    for(size_t i = 0; i < o_total_size; i++)
    {
        out[i].x = -1;
    }

    GPU_ERR_CHECK(hipMalloc(&d_in, i_total_bytes));
    GPU_ERR_CHECK(hipMalloc(&d_out, o_total_bytes));
    GPU_ERR_CHECK(hipMemcpy(d_in, in.data(), i_total_bytes, hipMemcpyHostToDevice));
    GPU_ERR_CHECK(hipMemcpy(d_out, out.data(), o_total_bytes, hipMemcpyHostToDevice));

    copy_direct<T><<<grid, block>>>(d_in, d_out);

    GPU_ERR_CHECK(hipMemcpy(out.data(), d_out, o_total_bytes, hipMemcpyDeviceToHost));

    std::cout << "Verify output...";

    for(auto i = 0; i < batch; i++)
    {
        for(auto j = 0; j < len; j++)
        {
            auto idx = i * len + j;
            std::cout << "output is " << out[idx].x << std::endl;
            if((int)(out[idx].x) != i * len + j)
            {
                std::cerr << "failed at [" << idx << "]: " << out[idx].x
                        << ", expected " << i * len + j << std::endl;
                exit(0);
            }
        }
    }

    std::cout << "done.\n";

    GPU_ERR_CHECK(hipFree(d_in));
    GPU_ERR_CHECK(hipFree(d_out));
}


int main(int argc, char* argv[])
{
    std::string type = "float2";
    int length=5, batch=10, device_id=0;

    if (type.compare("float2") == 0) {
        execute_test<float2>(length, batch, device_id);
    } else if (type.compare("double2") == 0) {
        execute_test<double2>(length, batch, device_id);
    }

    return 0;
}

