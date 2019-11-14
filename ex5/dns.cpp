#include <iostream>
#include <math.h>
#include <mpi.h>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

int main(int argc, char *argv[])
{
    int power = strtol(argv[1], NULL, 10);
    int N = pow(2, power);
    int comm_sz, my_rank;

    MPI_Init(NULL, NULL);
    double start = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int ndims = 3, dims[3] = {0}, periods[3] = {0}, reorder = 0, coords[3] = {0}, othercomm_sz;
    int my_cartrank;
    MPI_Comm comm_cart;
    MPI_Comm othercomm = MPI_COMM_WORLD;
    othercomm_sz = comm_sz;
    MPI_Dims_create(othercomm_sz, ndims, dims); //计算各维大小
    //printf("%d %d %d\n", dims[0], dims[1], dims[2]);
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart); //创建笛卡尔拓扑
    MPI_Comm_rank(comm_cart, &my_cartrank);                                     //获取进程在笛卡尔通信域的进程号
    MPI_Cart_coords(comm_cart, my_rank, ndims, coords);
    //printf("MPI_COMM_WORLD: %d of %d;  coords: (%d,%d,%d), cart_comm_rank:%d.\n",
    //       my_rank, comm_sz, coords[0], coords[1], coords[2], my_cartrank);

    MPI_Comm comm_cart_i, comm_cart_j, comm_cart_k;
    int remain_dims[3] = {1, 0, 0}; //设定按行划分，保留列号作为进程号
    MPI_Cart_sub(comm_cart, remain_dims, &comm_cart_i);
    remain_dims[0] = 0, remain_dims[1] = 1, remain_dims[2] = 0;
    MPI_Cart_sub(comm_cart, remain_dims, &comm_cart_j);
    remain_dims[0] = 0, remain_dims[1] = 0, remain_dims[2] = 1;
    MPI_Cart_sub(comm_cart, remain_dims, &comm_cart_k);

    int i_size = dims[0];
    int j_size = dims[1];
    int k_size = dims[2];
    int i_block = N / i_size;
    int j_block = N / j_size;
    int k_block = N / k_size;
    int i_rank = coords[0];
    int j_rank = coords[1];
    int k_rank = coords[2];
    int local_i = i_rank * i_block;
    int local_j = j_rank * j_block;
    int local_k = k_rank * k_block;

    //块内变量声明
    double **A = new double *[i_block];
    for (int i = 0; i < i_block; i++)
        A[i] = new double[k_block];

    double **B = new double *[k_block];
    for (int i = 0; i < k_block; i++)
        B[i] = new double[j_block];

    double **y = new double *[i_block];
    for (int i = 0; i < i_block; i++)
        y[i] = new double[j_block];

    
    //将A的第 j 列发送到第 j 层的某一列进程中
    //将B的 第 i 行发送到第 i 层的某一行进程中
    //然后对同层 进行广播

    if (j_rank == 0)
        for (int i = local_i; i < local_i + i_block; i++)
            for (int j = local_k; j < local_k + k_block; j++)
                A[i - local_i][j - local_k] = (i - 0.1 * j + 1) / (i + j + 1);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < i_block; i++)
        for (int j = 0; j  < k_block; j++)
            MPI_Bcast(&A[i][j], 1, MPI_DOUBLE, 0, comm_cart_j);

    if (i_rank == 0)
        for (int i = local_k; i < local_k + k_block; i++)
            for (int j = local_j; j < local_j + j_block; j++)
                B[i - local_k][j - local_j] = (j - 0.2 * i + 1) * (i + j + 1) / (i * i + j * j + 1);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < k_block; i++)
        for (int j = 0; j < j_block; j++)
            MPI_Bcast(&B[i][j], 1, MPI_DOUBLE, 0, comm_cart_i);

    //块内乘法
    for (int i = 0; i < i_block; i++)
        for (int j = 0; j < j_block; j++)
            for (int k = 0; k < k_block; k++)
                y[i][j] += A[i][k] * B[k][j];

    //沿着第三维做加法归约，算出具体对应的矩阵C 的每一个元素
    double **sum = new double *[i_block];
    for (int i = 0; i < i_block; i++)
        sum[i] = new double[j_block];

    for (int i = 0; i < i_block; i++)
        for (int j = 0; j < j_block; j++)
            MPI_Reduce(&y[i][j], &sum[i][j], 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart_k);

    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime() - start;

    //结果
    if (k_rank == 0)
    {
        printf("%d\n", my_rank);
        for (int i = 0; i < i_block; i++)
        {
            for (int j = 0; j < j_block; j++)
                printf("%f ", sum[i][j]);
            printf("\n");
        }
    }
    

    MPI_Finalize();
    if (my_rank == 0)
        printf("Tp of dns %d core 2^%d matrix is %f\n", comm_sz, power, time);
    return 0;
}