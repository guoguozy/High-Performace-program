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
    int comm_sz;
    int my_rank;
    MPI_Init(NULL, NULL);
    double start = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int ndims = 2, dims[2] = {0}, periods[2] = {1, 1}, reorder = 0, coords[2] = {0}, othercomm_sz;
    int my_cartrank, my_coords2rank;
    MPI_Comm comm_cart;
    MPI_Comm othercomm = MPI_COMM_WORLD;
    othercomm_sz = comm_sz;
    MPI_Dims_create(othercomm_sz, ndims, dims);                                 //计算各维大小
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart); //创建笛卡尔拓扑
    MPI_Comm_rank(comm_cart, &my_cartrank);                                     //获取进程在笛卡尔通信域的进程号
    MPI_Cart_coords(comm_cart, my_rank, ndims, coords);                         //将全局进程号转为进程的笛卡尔坐标
    // MPI_Cart_rank(comm_cart, coords, &my_coords2rank);                          //将进程的笛卡尔坐标转为笛卡尔通信域的进程号
    // printf("MPI_COMM_WORLD: %d of %d;  coords: (%d,%d), cart_comm_rank:%d, coords to rank:%d.\n", my_rank, comm_sz,coords[0],coords[1],my_cartrank,my_coords2rank);

    int col_rank, row_rank;
    col_rank = coords[0];
    row_rank = coords[1];

    int i_size = dims[0];
    int j_size = dims[1];
    int i_block = N / i_size;
    int j_block = N / j_size;
    int local_i = (my_rank / j_size) * i_block;
    int local_j = (my_rank % j_size) * j_block;
    MPI_Status status;

    //块内变量声明
    double **A = new double *[i_block];
    for (int i = 0; i < i_block; i++)
        A[i] = new double[j_block];
    double **B = new double *[j_block];
    for (int i = 0; i < j_block; i++)
        B[i] = new double[i_block];
    double **y = new double *[i_block];
    for (int i = 0; i < i_block; i++)
        y[i] = new double[i_block];

    //初始分块
    for (int i = local_i; i < local_i + i_block; i++)
        for (int j = local_j; j < local_j + j_block; j++)
            A[i - local_i][j - local_j] = (i - 0.1 * j + 1) / (i + j + 1);

    for (int i = local_i; i < local_i + i_block; i++)
        for (int j = local_j; j < local_j + j_block; j++)
            B[i - local_i][j - local_j] = ((double)j - 0.2 * (double)i + 1) * ((double)i + (double)j + 1) / ((double)i * (double)i + (double)j * (double)j + 1);

    for (int i = 0; i < i_block; i++)
        for (int j = 0; j < i_block; j++)
            y[i][j] = 0.0;

    int rowSrc, rowDest, colSrc, colDest;
    //对矩阵A的所有子矩阵 A i,j 进行 j -左移位(带回绕)；
    //对矩阵 B 的所有子矩阵 B i,j 进行 i -上移(带回绕)
    if (coords[0] > 0)
    {
        MPI_Cart_shift(comm_cart, 1, -coords[0], &rowSrc, &rowDest);
        for (int i = 0; i < i_block; i++)
            for (int j = 0; j < j_block; j++)
                MPI_Sendrecv_replace(&A[i][j], 1, MPI_DOUBLE, rowDest, 0, rowSrc, 0, comm_cart, &status);
    }

    if (coords[1] > 0)
    {
        MPI_Cart_shift(comm_cart, 0, -coords[1], &colSrc, &colDest);
        for (int i = 0; i < j_block; i++)
            for (int j = 0; j < i_block; j++)
                MPI_Sendrecv_replace(&B[i][j], 1, MPI_DOUBLE, colDest, 0, colSrc, 0, comm_cart, &status);
    }

    //执行一次本地的子矩阵乘
    for (int i = 0; i < i_block; i++)
        for (int j = 0; j < i_block; j++)
            for (int k = 0; k < j_block; k++)
                y[i][j] += A[i][k] * B[k][j];

    MPI_Barrier(MPI_COMM_WORLD);

    //所有的 A i,j向左移一位， B i,j上移一位。
    //执行下一次乘法并累积结果。
    //重复上面的移位和乘法直至 𝑝个块都完成
    for (int num = 1; num < i_size; num++)
    {
        MPI_Cart_shift(comm_cart, 1, -1, &rowSrc, &rowDest); //循环向左平移1格
        for (int i = 0; i < i_block; i++)
            for (int j = 0; j < j_block; j++)
                MPI_Sendrecv_replace(&A[i][j], 1, MPI_DOUBLE, rowDest, 0, rowSrc, 0, comm_cart, &status);

        MPI_Cart_shift(comm_cart, 0, -1, &colSrc, &colDest); //循环向上平移1格
        for (int i = 0; i < j_block; i++)
            for (int j = 0; j < i_block; j++)
                MPI_Sendrecv_replace(&B[i][j], 1, MPI_DOUBLE, colDest, 0, colSrc, 0, comm_cart, &status);

        for (int i = 0; i < i_block; i++)
            for (int j = 0; j < i_block; j++)
                for (int k = 0; k < j_block; k++)
                    y[i][j] += A[i][k] * B[k][j];
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //结果
    for (int i = 0; i < i_block; i++)
        for (int j = 0; j < i_block; j++)
            printf("%d %d %d %f\n", my_rank, col_rank, row_rank, y[i][j]);
    MPI_Finalize();
    double time = MPI_Wtime() - start;
    if (my_rank == 0)
        printf("Tp of cannon %d core 2^%d matrix is %f\n", comm_sz, power, time);
    return 0;
}