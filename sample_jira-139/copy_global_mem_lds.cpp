///////////////////////////////////////////////////////////////////////////////
//  Build:
//    with hipcc:
//      hipcc copy_global_mem_lds.cpp  -o copy_global_mem_lds
//
///////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>

#define HIP_ENABLE_PRINTF
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>

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

// Copy from global memory to global memory through LDS
template <typename T>
__global__ void copy_lds(const T* __restrict__ idata,
                         T* __restrict__ odata)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shmem_ptr[];
    T* lds = reinterpret_cast<T*>(shmem_ptr);

    int base = blockDim.x * blockIdx.x;

    int idx  =  threadIdx.x;
    lds[idx] = idata[base + idx];

    __syncthreads();

    odata[base + idx] = lds[idx];
}

template <typename T>
void execute_test(const int len,
                  const int batch)
{
    size_t i_total_size  = len * batch;
    size_t o_total_size  = len * batch;
    size_t i_total_bytes = i_total_size * sizeof(T);
    size_t o_total_bytes = o_total_size * sizeof(T);
    size_t lds_bytes     = len * sizeof(T);

    dim3 grid(batch);
    dim3 block(len);

    std::cout << "------------------------------------------------"
              << "\nlen " << len << ", batch " << batch <<std::endl;
    std::cout << "lds bytes: " << lds_bytes << std::endl;

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

    copy_lds<T>
        <<<grid, block, lds_bytes>>>(d_in, d_out);

    GPU_ERR_CHECK(hipMemcpy(out.data(), d_out, o_total_bytes, hipMemcpyDeviceToHost));

    std::cout << "Verify output...";

    for(auto i = 0; i < batch; i++)
    {
        for(auto j = 0; j < len; j++)
        {
            auto idx = i * len  + j;
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
    int length=5, batch=10;

    if (type.compare("float2") == 0)
        execute_test<float2>(length, batch);
    else if (type.compare("double2") == 0)
        execute_test<double2>(length, batch);

    return 0;
}
