///////////////////////////////////////////////////////////////////////////////
//  Build:
//    with hipcc:
//       hipcc transpose.cpp  -o transpose
//
///////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <vector>

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


// Tiled transpose through LDS
template <typename T>
__global__ void transpose_lds(const T* __restrict__ idata,
                              T* __restrict__ odata,
                              const int n,
                              const int elem_per_thread,
                              const int rows,
                              const int padding)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shmem_ptr[];
    T*                                                   lds = reinterpret_cast<T*>(shmem_ptr);

    int o_base = blockIdx.x * n * rows;
    int i_base = o_base + blockIdx.x * padding * rows;

    for(int j = 0; j < rows; j++)
        for(int i = 0; i < elem_per_thread; i++)
        {
            int idx  = j * n + i * blockDim.x + threadIdx.x;
            lds[idx] = idata[i_base + j * padding + idx];
        }

    __syncthreads();

    for(int i = 0; i < elem_per_thread; i++)
        for(int j = 0; j < rows; j++)
        {
            int lds_idx = i * blockDim.x + (blockDim.x * j + threadIdx.x) % rows * n
                          + (blockDim.x * j + threadIdx.x) / rows;
            int gw_idx    = o_base + i * blockDim.x * rows + j * blockDim.x + threadIdx.x;
            odata[gw_idx] = lds[lds_idx];
        }
}

//   len             : the size of one row(the fast dimension)
//   batch           : the total size of how many rows(the second fast dimension)
//   elem_per_thread : element number handled by each thread per row
//   rows            : tile width
//   padding         : padding element number at the end of each row
//
template <typename T>
void execute_test(const int       len,
                      const int   batch,
                      const int   elem_per_thread,
                      const int   rows,
                      const int   padding)
{
    size_t i_total_size  = len * batch;
    size_t o_total_size  = len * batch;
    size_t i_total_bytes = i_total_size * sizeof(T);
    size_t o_total_bytes = o_total_size * sizeof(T);
    size_t lds_bytes     = len * rows * sizeof(T);

    size_t i_padded_size  = (len + padding) * batch;
    size_t o_padded_size  = i_padded_size;
    size_t i_padded_bytes = (len + padding) * batch * sizeof(T);
    size_t o_padded_bytes = i_padded_bytes;


    assert(batch % rows == 0);
    assert(len % elem_per_thread == 0);

    dim3 grid(batch / rows);
    dim3 block(len / elem_per_thread);

    std::cout << "--------------------------------------------------------------------------------"
              << "\nlen " << len << ", batch " << batch
              << ", tile len " << len / elem_per_thread << ", tile width " << rows
              << "\nelem_per_thread " << elem_per_thread << ", padding for each row " << padding << std::endl;
    std::cout << "lds bytes: " << lds_bytes << std::endl;

    std::vector<T> in(i_padded_size);
    std::vector<T> out(o_padded_size);
    T *d_in, *d_out;

    std::cout << "Generate input...\n";
    for(auto i = 0; i < batch; i++)
        for(auto j = 0; j < len; j++)
        {
            std::cout << "input is " << in[i * len + j].x << std::endl;
            in[i * (len + padding) + j].x = i * len + j;
        }

    for(size_t i = 0; i < o_padded_size; i++)
    {
        out[i].x = -1;
    }

    GPU_ERR_CHECK(hipMalloc(&d_in, i_padded_bytes));
    GPU_ERR_CHECK(hipMalloc(&d_out, o_padded_bytes));
    GPU_ERR_CHECK(hipMemcpy(d_in, in.data(), i_padded_bytes, hipMemcpyHostToDevice));
    GPU_ERR_CHECK(hipMemcpy(d_out, out.data(), o_padded_bytes, hipMemcpyHostToDevice));

            transpose_lds<T>
                <<<grid, block, lds_bytes>>>(d_in, d_out, len, elem_per_thread, rows, padding);

    GPU_ERR_CHECK(hipMemcpy(out.data(), d_out, o_padded_bytes, hipMemcpyDeviceToHost));

    std::cout << "Verify output...";

    for(auto i = 0; i < batch / rows; i++)
       {
            auto base = i * len * rows;
            for(auto j = 0; j < len * rows; j++) // check rows * len transpose
            {
                int value = base + (j % rows) * len + j / rows;
                std::cout << "output is " << out[i * len * rows + j].x << std::endl;
                if((int)(out[i * len * rows + j].x) != value)
                {
                    std::cerr << "failed at [" << base + j << "]: " << out[i * len * rows + j].x
                              << ", expected " << value
                              << std::endl;
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
    int length=5, batch=10, elem_per_thread_max=1, rows_max=1, padding_max=10;

        if (type.compare("float2") == 0)
            execute_test<float2>(length, batch, elem_per_thread_max, rows_max, padding_max);
        else if (type.compare("double2") == 0)
            execute_test<double2>(length, batch, elem_per_thread_max, rows_max, padding_max);

    return 0;
}
