#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int N = 16384;

__constant__ float B[16384];

__global__ void MatMulti(float *A, float *C,const int N)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    float sum=0.0;
    if(row<N && col<N && row==col){

        for(int i=0;i<N;i++){

            sum+=A[row*N+i]*B[i];
            __syncthreads();

        }

        C[row]=sum;
    }

}

int main()
{
    //cpu变量声明以及初始化
    float *a = (float *)malloc(N * N * sizeof(float));
    float b[16384];
    float *c = (float *)malloc(N * sizeof(float));
    float *testc=(float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i){

        for (int j = 0; j < N; ++j)
        {
            a[i * N + j] = i - 0.1 * j + 1;
        }
    }

    for( int i=0;i<N;i++){

        b[i]=log(sqrt(i*i-i+2));
        c[i]=0.0;
    }

    //gpu变量声明以及传递初值
    float *A, *C;
    cudaEvent_t event_start, event_stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaEventRecord(event_start, 0);
    cudaMalloc((void **)&A, N * N * sizeof(float));
    cudaMalloc((void **)&C, N * sizeof(float));
    cudaMemcpy(A, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(B, b,sizeof(float)*N);
    cudaMemcpy(C, c, N * sizeof(float), cudaMemcpyHostToDevice);

    //运行kernel函数
    dim3 block(32, 32);
    dim3 grid(N / block.x, N / block.y);
    MatMulti<<<grid, block>>>(A,C,N);
    //cudaThreadSynchronize();
    //cudaMemcpyFromSymbol(testb,B,N*sizeof(float));
    cudaMemcpy(c, C, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&elapsedTime, event_start, event_stop);
    printf("cuda time = %lf ms\n", elapsedTime);
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);

    //cpu运行计算
    clock_t start;
    clock_t end;
    start=clock();
    for (int i = 0; i < N; ++i){

        for (int j = 0; j < N; ++j)
         {
             testc[i]+=a[i*N+j]*b[j];
         }
    }

    //cpu与gpu比较
    bool flag = true;
    for (int i = 0; i < N; ++i){

        float a=testc[i];
        float b=c[i];
        if (a!=b)
        {
            flag = false;
        }

    }
    if (flag == true)
        printf("correct\n");
    else{
        printf("wrong\n");
    }



    //需要测试时间的代码
    end=clock();
    float time=(float)(end-start)/CLOCKS_PER_SEC;
    printf("cpu time = %lf ms\n", time*1000);

    cudaFree(A);
    cudaFree(C);
    free(a);
    free(c);
    free(testc);

}