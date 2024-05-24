#include <stdio.h>
#include <math.h>
#define BLOCK_SIZE 16
#define N 1000;

__global__ void gpu_matrix_mult(int *a,int *b,int *c,int size)
{
    int x = blockDim.x*blockIdx.x+threadIdx.x;
    int y = blockDim.y*blockIdx.y+threadIdx.y;
    int tmp = 0;
    for(int step = 0;step < size; step++)
    {
        tmp+=a[x*size+step]*b[step*size+y];
    }
    c[x*size+tmp] = tmp;
}

void cpu_matrix_mult(int *a,int *b,int *c,int size)
{
    
    for(int x = 0; x < size; x++)
    {
        for(int y = 0; y < size; y++)
        {
            int tmp = 0;
            for(int step = 0;step < size; step++)
            {
                tmp += a[x*size + step]*b[step*size + y];
            }
            c[x * size + y]=tmp;
        }
    }
}
int main()
{
    const int matrix_size = N;
    int memsize = sizeof(int) * matrix_size * matrix_size;
    int *h_a,*h_b,*h_c,*h_cc;
    cudaMallocHost((void**)&h_a,memsize);
    cudaMallocHost((void**)&h_b,memsize);
    cudaMallocHost((void**)&h_c,memsize);
    cudaMallocHost((void**)&h_cc,memsize);
    for(int i = 0; i < matrix_size;i++)
    {
        for(int j = 0;j < matrix_size;j++)
        {
            h_a[i * matrix_size + j]=rand() % 1024;
        }
    }
    for(int i = 0; i < matrix_size;i++)
    {
        for(int j = 0;j < matrix_size;j++)
        {
            h_b[i * matrix_size + j]=rand() % 1024;
        }
    }

    int* d_a,*d_b,*d_c;
    cudaMalloc((void**)(&d_a),memsize);
    cudaMalloc((void**)(&d_b),memsize);
    cudaMalloc((void**)(&d_c),memsize);

    cudaMemcpy(d_a,h_a,memsize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,memsize,cudaMemcpyHostToDevice);

    unsigned int grid_rows = (matrix_size+BLOCK_SIZE-1)/BLOCK_SIZE;
    unsigned int grid_cols = (matrix_size+BLOCK_SIZE-1)/BLOCK_SIZE;

    dim3 dimGrid(grid_cols,grid_rows);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);/// warp为32,这32个线程共享一个指令
    gpu_matrix_mult<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,matrix_size);
    cudaMemcpy(h_c,d_c,memsize,cudaMemcpyDeviceToHost);
    cpu_matrix_mult(h_a,h_b,h_cc,matrix_size);
    
    bool errors = false;
    for(int i = 0;i < matrix_size;i++)
    {
        for(int j = 0;j < matrix_size;j++)
        {
            if(fabs(h_cc[i*matrix_size + j]-h_c[i*matrix_size+j])>(1.0e-10))
            {
                errors = true;
                cudaFreeHost(h_a);
                cudaFreeHost(h_b);
                cudaFreeHost(h_c);
                cudaFreeHost(h_cc);
                cudaFree(d_a);
                cudaFree(d_b);
                cudaFree(d_c);
                printf("info error!\n");
                return;
            }
        }
    }
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
