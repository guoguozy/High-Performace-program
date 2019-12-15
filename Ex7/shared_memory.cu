#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int N =16384;

extern __shared__ double shareB[];
__global__ void MatMulti(double *A, double *B, double *C,const int N,int num)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    for(int i=0;i<4;i++)

        shareB[i]=B[i+num*4];

    __syncthreads();
    double sum=0.0;
    if(row<N && col<N && row==col){
        for(int i=0;i<4;i++){

            sum+=A[row*N+i+num*4]*shareB[i];
        }
        C[row]+=sum;
    }

}

int main()
{
    //cpu变量声明以及初始化
    double *a = (double *)malloc(N * N * sizeof(double));
    double *b = (double *)malloc(N * sizeof(double));
    double *c = (double *)malloc(N * sizeof(double));
    double *testc=(double *)malloc(N * sizeof(double));

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
    double *dev_A, *dev_B, *dev_C;
    cudaEvent_t event_start, event_stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaEventRecord(event_start, 0);
    cudaMalloc((void **)&dev_A, N * N * sizeof(double));
    cudaMalloc((void **)&dev_B, N * sizeof(double));
    cudaMalloc((void **)&dev_C, N * sizeof(double));
    cudaMemcpy(dev_A, a, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, b, N * sizeof(double), cudaMemcpyHostToDevice);


    for(int i=0;i<N/4;++i){
      cudaMemcpy(dev_C, c, N * sizeof(double), cudaMemcpyHostToDevice);
      dim3 block(32, 32);
      dim3 grid(N / block.x, N / block.y);
      //运行kernel函数
      MatMulti<<<grid, block,4*sizeof(double)>>>(dev_A, dev_B, dev_C, N,i);
      //cudaThreadSynchronize();
      cudaMemcpy(c, dev_C, N * sizeof(double), cudaMemcpyDeviceToHost);
    }

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
    //需要测试时间的代码
    end=clock();
    double time=(double)(end-start)/CLOCKS_PER_SEC;
    printf("cpu time = %lf ms\n", time*1000);

    //cpu与gpu比较
    bool flag = true;
    for (int i = 0; i < N; ++i){

        float a=testc[i];
        float b=c[i];
        if (a!=b)
        {
            printf("%lf %lf\n",testc[i],c[i]);
            flag = false;
        }

    }
    if (flag == true)
        printf("correct\n");
    else{
        printf("wrong\n");
    }

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    free(a);
    free(b);
    free(c);
    free(testc);
}