#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int N = 16384;

__global__ void MatMulti(double *A, double *B, double *C, const int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0.0;
    if (row < N && col < N && row == col)
        for (int i = 0; i < N; i++)
            sum += A[row * N + i] * B[i];
    C[row] = sum;
}

int main()
{
    //cpu变量声明以及初始化
    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * sizeof(double));
    double *C = (double *)malloc(N * sizeof(double));
    double *answer = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            A[i * N + j] = i - 0.1 * j + 1;
        }
        B[i] = log(sqrt(i * i - i + 2));
        C[i] = 0.0;
    }


    //gpu变量声明以及传递初值
    double *dev_A, *dev_B, *dev_C;
    cudaEvent_t event_start, event_stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaEventRecord(event_start, 0);
    cudaMalloc((void **)&dev_A, N * N * sizeof(double));
    cudaMalloc((void **)&dev_B, N * sizeof(double));
    cudaMalloc((void **)&dev_C, N * sizeof(double));
    cudaMemcpy(dev_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, N * sizeof(double), cudaMemcpyHostToDevice);

    //运行kernel函数
    dim3 block(32, 32);
    dim3 grid(N / block.x, N / block.y);
    MatMulti<<<grid, block>>>(dev_A, dev_B, dev_C, N);
    //cudaThreadSynchronize();
    cudaMemcpy(C, dev_C, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&elapsedTime, event_start, event_stop);
    printf("cuda time = %lf ms\n", elapsedTime);
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);

    //cpu运行计算
    clock_t start= clock(); 
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            answer[i] += A[i * N + j] *B[j];

    //需要测试时间的代码
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("cpu time = %lf ms\n", time * 1000);

    //cpu与gpu比较
    bool flag = true;
    for (int i = 0; i < N; ++i)
    {
        float a = answer[i];
        float b = C[i];
        if (a != b)
        {
            flag = false;
            printf("wrong at %f %f\n",a,b);
        }    
    }
    if (flag)
        printf("correct\n");
    else
        printf("wrong\n");

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    free(A);
    free(B);
    free(C);
    free(answer);
}