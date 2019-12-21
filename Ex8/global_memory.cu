#include <iostream>
using namespace std;

__global__ void MatrixMulKernel(int m, int n, int k, float *A, float *B, float *C)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < m) && (Col < k))
    {
        float Cvalue = 0.0;
        for (int i = 0; i < n; ++i)
            Cvalue += A[Row * n + i] * B[Col + i * k];
        C[Row * k + Col] = Cvalue;
    }
}
#define TILE_WIDTH 16

int main()
{
    //这里将矩阵按照行优先转换成了一维的形式
    int m = 4096, n = 4096, k = 4096;
    float *A = (float *)malloc(m * n * sizeof(float));
    float *B = (float *)malloc(n * k * sizeof(float));
    float *C = (float *)malloc(m * k * sizeof(float));
    float *result = (float *)malloc(m * k * sizeof(float));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
        {
            A[i * m + j] = (i - 0.1 * j + 1) / (i + j + 1);
            B[i * m + j] = (j - 0.2 * i + 1) * (i + j + 1) / (i * i + j * j + 1);
            C[i * m + j] = 0.0;
        }

    //分配显存空间
    int size = sizeof(float);
    float *d_a;
    float *d_b;
    float *d_c;

    // GPU time calculate start
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&d_a, m * n * size);
    cudaMalloc((void **)&d_b, n * k * size);
    cudaMalloc((void **)&d_c, m * k * size);

    //把数据从Host传到Device
    cudaMemcpy(d_a, A, size * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, size * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, C, size * m * k, cudaMemcpyHostToDevice);

    //分配网格结构
    dim3 dimGrid((k - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    //调用内核函数
    MatrixMulKernel<<<dimGrid, dimBlock>>>(m, n, k, d_a, d_b, d_c);

    //将结果传回到主机端
    cudaMemcpy(C, d_c, size * m * k, cudaMemcpyDeviceToHost);

    // GPU time calculate end
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "GPU time: " << elapsedTime << " ms" << endl;

    //CPU计算正确结果
    clock_t begin = clock();
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            float sum = 0;
            for (int k = 0; k < m; ++k)
                sum += A[i * m + k] * B[k * m + j];
            result[i * m + j] = sum;
        }
    }
    clock_t end = clock();
    cout << "CPU time: " << (end - begin) * 1000 / CLOCKS_PER_SEC << " ms" << endl;

    //比较结果
    bool flag = true;
    for (int i = 0; i < m * k; ++i)
    {
        if (abs(result[i] - C[i]) > 0.001)
        {
            flag = false;
            cout << result[i] << "-" << C[i] << endl;
        }
    }
    if (flag)
        cout << "Check answer: Correct!" << endl;
    else
        cout << "Check answer: Error!" << endl;

    //释放显存空间
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(A);
    free(B);
    free(C);
    free(result);
    return 0;
}