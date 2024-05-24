#include <stdio.h>
#include <math.h>
/// q1:需要频繁将global_memory->local_memory
/// s1:将global_memory->shared_memory，然后在使用的时候将shared_memory->local_memory
/// note:same block same shared_memory
#define M 1000
#define N 500
#define K 1000
/// 使用统一内存变量（不用区分是host端还是device端）
__managed__ int a[M*N];
__managed__ int b[N*K];
__managed__ int c_cpu[M*K];
__managed__ int c_gpu[M*K];

#define BLOCK_SIZE 16

// matrix1:M*N
// matrix2:N*K
// matrix3:M*K
__global__ void gpu_matrix(int* a,int* b,int* c, int m,int n,int k)
{
    __shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sub_b[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockDim.x*blockIdx.x+threadIdx.x;
    int y = blockDim.y*blockIdx.y+threadIdx.y;
    int tmp = 0;
    for(int step = 0;step < n/BLOCK; step++)
    {
        /// @brief c(x,y)->拿到a中(threadIdx.x,y)，(BLOCK_SIZE+threadIdx.x,y)...存入到sharedA中
        int step_x = step*BLOCK_SIZE+x;
        int step_y = y;
        idx = step_y*n+step_x;
        if(step_x>=n||step_y>=m)
        {
            sub_a[y][x] = 0;
        }else
        {
            sub_a[y][x] = a[idx];
        }
        /// @brief c(x,y)->拿到B中(x,threadIdx.y)，(threadIdx.x,y+BLOCK_SIZE)...存入到sharedB中
        step_x = x;
        
        step_y = step * BLOCK_SIZE + y;
        idx = step_y*k + step_x;
        if(step_x>=k||step_y>=n)
        {
            sub_b[y][x] = 0;
        }else
        {
            sub_b[y][x] = b[idx];
        }
        __syncthreads();
        /// @brief 当一个block中的线程都进行了一次取数后，进行一次运算
        for(int i = 0;i<BLOCK_SIZE;i++)
        {
            tmp+=sub_a[y][i]*sub_b[i][x];
        }
        __syncthreads();
    }
    if(x < k && y < m)
    {
        c[y * k + x] = tmp;
    }
}

void cpu_matrix(int* a,int* b,int* c, int m,int n,int k)
{
    for(int y = 0; y < m;y++)
    {
        for(int x = 0;x < k; x++)
        {
            int tmp = 0;
            for(int step = 0;step < n; step++){
                tmp += a[y * n + step] * b[step * k+x];
            }
            c[y * k + x] = tmp;
        }
    }
}

int main()
{
    for(int y = 0;y < M; y++)
    {
        for(int x = 0; x < N;x++)
        {
            a[y * M+x] = rand()%1024;
        }
    }
    
    for(int y = 0;y < N; y++)
    {
        for(int x = 0; x < K;x++)
        {
            b[y*M+x] = rand()%1024;
        }
    }


    unsigned int grid_x = (K + BLOCK_SIZE -1)/BLOCK_SIZE;
    unsigned int grid_y = (M + BLOCK_SIZE -1)/BLOCK_SIZE;

    dim3 dimGrid(grid_x,grid_y);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    gpu_matrix<<<dimGrid,dimBlock>>>(a, b, c_gpu, M, N, K);
    cpu_matrix(a, b, c_gpu, M, N, K);
    bool flag = false;
    for(int y = 0;y<M;y++)
    {
        for(int x = 0; x < K; x++)
        {
            if(fabs(c_cpu[y*K + x])-c_gpu[y*K + x]>=1.0e-10)
            {
                errors = true;
            }
        }
    }
    printf("info:Result:%s\n",errors?"failed":"success");
    
    return 0;
}
