#include <iostream>
using namespace std;
#define TILE_WIDTH 16

__global__ void MatrixMulKernel(int m, int n, int k, float *A, float *B, float *C)
{
    //申请共享内存，存在于每个block中
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    //简化坐标记法,出现下面6个表示的地方就是并行的地方。
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //确定结果矩阵中的行和列
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    //临时变量
    float Cvalue = 0;

    //循环读入A,B瓦片，计算结果矩阵，分阶段进行计算
    for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; ++t)
    {
        //将A,B矩阵瓦片化的结果放入shared memory中，每个线程加载相应于C元素的A/B矩阵元素
        if (Row < m && t * TILE_WIDTH + tx < n)              //越界处理，满足任意大小的矩阵相乘（可选）
                                                             //ds_A[tx][ty] = A[t*TILE_WIDTH + tx][Row];
            ds_A[tx][ty] = A[Row * n + t * TILE_WIDTH + tx]; //以合并的方式加载瓦片
        else
            ds_A[tx][ty] = 0.0;

        if (t * TILE_WIDTH + ty < n && Col < k)
            //ds_B[tx][ty] = B[Col][t*TILE_WIDTH + ty];
            ds_B[tx][ty] = B[(t * TILE_WIDTH + ty) * k + Col];
        else
            ds_B[tx][ty] = 0.0;

        //保证tile中所有的元素被加载
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            Cvalue += ds_A[i][ty] * ds_B[tx][i]; //从shared memory中取值

        //确保所有线程完成计算后，进行下一个阶段的计算
        __syncthreads();

        if (Row < m && Col < k)
            C[Row * k + Col] = Cvalue;
    }
}

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
    cudaMalloc((void **)&d_a, m * n * size);
    cudaMalloc((void **)&d_b, n * k * size);
    cudaMalloc((void **)&d_c, m * k * size);

    // GPU time calculate start
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //把数据从Host传到Device
    cudaMemcpy(d_a, A, size * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, size * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, C, size * m * k, cudaMemcpyHostToDevice);

    //分配网格结构
    dim3 dimGrid((k - 1) / TILE_WIDTH + 1, (m - 1) / TILE_WIDTH + 1, 1); //向上取整
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
    /*clock_t begin = clock();
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
        cout << "Correct!" << endl;
    else
        cout << "Error!" << endl;*/

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