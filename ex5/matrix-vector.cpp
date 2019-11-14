#include <math.h>
#include <mpi.h>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;
#include <iostream>
int main(int argc, char *argv[])
{
    int N = strtol(argv[1], NULL, 10);
    int comm_sz;
    int my_rank;
    MPI_Init(NULL, NULL);
    double start = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int i_size, j_size;
    //分块
    switch (comm_sz)
    {
    case 1:
        i_size = j_size = 1;
        break;
    case 2:
        i_size = 1, j_size = 2;
        break;
    case 4:
        i_size = j_size = 2;
        break;
    case 8:
        i_size = 2, j_size = 4;
        break;
    case 16:
        i_size = j_size = 4;
        break;
    case 32:
        i_size = 4, j_size = 8;
        break;
    case 64:
        i_size = j_size = 8;
        break;
    case 128:
        i_size = 8, j_size = 16;
        break;
    case 256:
        i_size = j_size = 16;
        break;
    case 512:
        i_size = 16, j_size = 32;
        break;
    default:
        break;
    }


    int i_block = N / i_size;
    int j_block = N / j_size;
    int local_i = (my_rank / j_size) * i_block;
    int local_j = (my_rank % j_size) * j_block;

    //划分通信域
    MPI_Comm col_comm, row_comm;
    int col = my_rank % j_size;
    int row = my_rank / j_size;
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    int col_rank, row_rank;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_rank(row_comm, &row_rank);

    //块内变量声明
    double *y = new double[i_block];
    for (int i = 0; i < i_block; i++)
        y[i] = 0.0;
    double *x = new double[j_block];
    double **A = new double *[i_block];
    for (int i = 0; i < i_block; i++)
        A[i] = new double[j_block];

    //列块广播
    bool   = false;
    for (int i = local_i; i < local_i + i_block; i++)
        for (int j = local_j; j < local_j + j_block; j++)
            if (i == j)
            {
                x[i % j_block] = ((double)i / ((double)i * (double)i + 1));
                index = true;
            }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(x, j_block, MPI_DOUBLE, local_j / i_block, col_comm);

    MPI_Barrier(MPI_COMM_WORLD);

    //块内乘法
    for (int i = local_i; i < local_i + i_block; i++)
        for (int j = local_j; j < local_j + j_block; j++)
            A[i - local_i][j - local_j] = ((double)i - 0.1 * (double)j + 1) / ((double)i + (double)j + 1) * x[j % j_block];

    MPI_Barrier(MPI_COMM_WORLD);

    double *sum = new double[i_block];

    //行块规约
    for (int i = 0; i < i_block; i++)
        for (int j = 0; j < j_block; j++)
            sum[i] += A[i][j];

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < i_block; i++)
        MPI_Reduce(&sum[i], &y[i], 1, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    MPI_Barrier(MPI_COMM_WORLD);

    double *result = new double[N];
    MPI_Gather(y, i_block, MPI_DOUBLE, result, i_block, MPI_DOUBLE, 0, col_comm);

    double time = MPI_Wtime() - start;
    if (my_rank == 0)
    {
        for (int i = 0; i < N; i++)
            printf("%f\n", result[i]);
        printf("Tp of matrix-vector %d core 2^%d matrix is %f\n", comm_sz, power, time);
    }
    MPI_Finalize();
    return 0;
}