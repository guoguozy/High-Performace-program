#include <bits/stdc++.h>
#include <iostream>
#include <mpi.h>
using namespace std;
double a, b, c, d;
const int eps = 1e-12;
int n;
double f(double x)
{
    return exp(c * x) / sqrt(1 - exp(-d * x));
}
/*double f(double x)
{
    return (right_h - left_h) / (b - a) * (x - a) + left_h;
} */
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len)
{
    if (left_endpt == 0)
        left_endpt = eps;
    double estimate, x;
    estimate = (f(left_endpt) + f(right_endpt)) / 2.0;
    for (int i = 1; i <= trap_count - 1; i++)
    {
        x = left_endpt + i * base_len;
        estimate += f(x);
    }
    estimate = estimate * base_len;
    return estimate;
}
int main(int argc, char *argv[])
{
    a = atof(argv[1]); //左边坐标
    b = atof(argv[2]); //右边坐标
    c = atof(argv[3]);
    d = atof(argv[4]);
    n = atoi(argv[5]); //分块数量
    int my_rank, comm_sz, local_n;
    double h, local_a, local_b;
    double local_int, total_int;
    int source;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    double startTime, endTime;
    MPI_Barrier(MPI_COMM_WORLD);
    startTime = MPI_Wtime();

    h = (b - a) / n;
    local_n = n / comm_sz;
    local_a = a + my_rank * local_n * h;            //计算区间左端点
    local_b = local_a + local_n * h;                //计算区间右端点
    local_int = Trap(local_a, local_b, local_n, h); //计算积分

    if (my_rank != 0)
        MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); //发送给0号进程
    else
    {
        total_int = local_int;
        for (source = 1; source < comm_sz; source++)
        {
            MPI_Recv(&local_int, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_int += local_int; //计算总积分
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    endTime = MPI_Wtime();
    if (my_rank == 0)
    {
        cout << "block number n= " << n << endl;
        cout << "integral from " << a << " to " << b << " =" << total_int << "\n";
        cout << " Tp:" << endTime - startTime << endl;
    }
    MPI_Finalize();
    return 0;
}