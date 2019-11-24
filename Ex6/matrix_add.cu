#include <stdio.h>
#include <stdlib.h>
int N = 8192;
int WIDTH = 16;

__global__ void MatAdd(double *A, double *B, double *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i * N + j] = A[i * N + j] + B[i * N + j];
}

double Get_CPU_add_time(double *a, double *b, double *c)
{
    clock_t start = clock();
    for (int i = 0; i < N * N; ++i)
        c[i] = a[i] + b[i];
    clock_t end = clock();
    return (end - start) * 1000 / CLOCKS_PER_SEC;
}

void Check_GPU_CPU_result(double *a, double *b, double *c)
{
    bool flag = true;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            if (a[i * N + j] + b[i * N + j] != c[i * N + j])
            {
                flag = false;
                printf("Fail at (%d,%d)\n", i, j);
                printf("Correct Answer :%lf , My Answer :%lf\n",
                       (i - 0.1 * j + 1) + (0.2 * j - 0.1 * i), c[i * N + j]);
            }
        }
    if (flag == true)
        printf("GPU Matrix add result: True\n");
}

int main()
{
    // malloc matrix a b c at host
    double *a = (double *)malloc(N * N * sizeof(double));
    double *b = (double *)malloc(N * N * sizeof(double));
    double *c = (double *)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            a[i * N + j] = i - 0.1 * j + 1;
            b[i * N + j] = 0.2 * j - 0.1 * i;
            c[i * N + j] = 0;
        }

    // malloc matrix A B C at device
    double *A, *B, *C;
    cudaMalloc((void **)&A, N * N * sizeof(double));
    cudaMalloc((void **)&B, N * N * sizeof(double));
    cudaMalloc((void **)&C, N * N * sizeof(double));

    // GPU calculate start
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Memcpy CPU -> GPU
    cudaMemcpy(A, a, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B, b, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize block, grid
    dim3 block(WIDTH, WIDTH);
    dim3 grid(N / block.x, N / block.y);
    MatAdd<<<grid, block>>>(A, B, C, N);

    // Memcpy result GPU -> CPU
    cudaMemcpy(c, C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // GPU calculate end
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // Check the result of GPU add
    Check_GPU_CPU_result(a, b, c);

    // print time of GPU,CPU
    printf("GPU ElapsedTime:%f ms \n", elapsedTime);
    printf("CPU ElapsedTime:%lf ms \n", Get_CPU_add_time(a, b, c));


    // free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(a);
    free(b);
    free(c);
}