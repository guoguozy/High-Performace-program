#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int N = 16384;

typedef struct  {
    double A1;
    double A2;
    double A3;
    double A4;
}tmp;

__global__ void MatMulti(tmp *A, tmp *B, double *C,const int N)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    double sum=0.0;
    if(row<N && col<N && row==col){
        for(int i=0;i<N/4;i++){

            sum+=A[row*N/4+i].A1*B[i*4].A1+A[row*N/4+i].A2*B[i*4+1].A1+A[row*N/4+i].A3*B[i*4+2].A1+A[row*N/4+i].A4*B[i*4+3].A1;
        }
        C[row]=sum;
    }

}

int main()
{
    //cpu变量声明以及初始化
    tmp *a = (tmp *)malloc(N * N/4 * sizeof(tmp));
    tmp *b = (tmp *)malloc(N * sizeof(tmp));
    double *c = (double *)malloc(N * sizeof(double));
    double *testc=(double *)malloc(N * sizeof(double));


    for (int i = 0; i < N; ++i){

        for (int j = 0; j < N/4; ++j)
        {
            a[i*N/4+j].A1=i-0.1*j*4+1;
            a[i*N/4+j].A2=i-(0.1*j*4+1)+1;
            a[i*N/4+j].A3=i-(0.1*j*4+2)+1;
            a[i*N/4+j].A4=i-(0.1*j*4+3)+1;
        }
    }

    for( int i=0;i<N;i++){

        b[i].A1=log(sqrt(i*i-i+2));
        b[i].A2=0.0;
        b[i].A3=0.0;
        b[i].A4=0.0;
        c[i]=0.0;
    }

    //gpu变量声明以及传递初值
    tmp *dev_A, *dev_B;
    double *dev_C;
    cudaEvent_t event_start, event_stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaEventRecord(event_start, 0);
    cudaMalloc((void **)&dev_A, N * N/4 * sizeof(tmp));
    cudaMalloc((void **)&dev_B, N * sizeof(tmp));
    cudaMalloc((void **)&dev_C, N * sizeof(double));
    cudaMemcpy(dev_A, a, N * N/4 * sizeof(tmp), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, b, N * sizeof(tmp), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, c, N * sizeof(double), cudaMemcpyHostToDevice);

    //运行kernel函数
    dim3 block(32, 32);
    dim3 grid(N / block.x, N / block.y);
    MatMulti<<<grid, block>>>(dev_A, dev_B, dev_C, N);
    //cudaThreadSynchronize();
    cudaMemcpy(c, dev_C, N * sizeof(double), cudaMemcpyDeviceToHost);
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

        for (int j = 0; j < N/4; ++j)
         {
             testc[i]+=a[i*N/4+j].A1*b[j*4].A1+a[i*N/4+j].A2*b[j*4+1].A1+a[i*N/4+j].A3*b[j*4+2].A1+a[i*N/4+j].A4*b[j*4+3].A1;
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
            printf("%lf %lf\n",a,b);
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