#include <bits/stdc++.h>
#include <iostream>
#include <mpi.h>
using namespace std;
double b, c, L, R;
int num_section = 0;
const int eps = 1e-12;
#define MAX_SECTION 10000
typedef double dl;
dl f(dl x)
{
    return exp(b * x) / sqrt(1 - exp(-c * x));
}
dl simpson(dl l, dl r)
{
    if (l == 0)
        l = eps;
    dl mid = (l + r) / 2;
    return (f(l) + 4 * f(mid) + f(r)) * (r - l) / 6;
}
dl solve(dl L, dl R, dl ans)
{
    if (L == 0)
        L = eps;
    num_section++;
    dl mid = (L + R) / 2, l = simpson(L, mid), r = simpson(mid, R);
    if (fabs(l + r - ans) <= eps || num_section > MAX_SECTION)
        return ans; //超过分块最大数，也返回不再进行划分
    return solve(L, mid, l) + solve(mid, R, r);
}

int main(int argc, char *argv[])
{
    b = atof(argv[1]);
    c = atof(argv[2]);
    L = atof(argv[3]);
    R = atof(argv[4]);
    int my_rank, comm_sz, section_size;
    dl local_L, local_R;
    long double local_int, total_int = 0;
    int source;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    double startTime, endTime;
    MPI_Barrier(MPI_COMM_WORLD);
    startTime = MPI_Wtime();

    section_size = (R - L) / comm_sz;
    int local_section = my_rank * section_size;

    local_L = L + my_rank * section_size;
    local_R = local_L + section_size;

    //调用solve递归求得积分
    local_int = solve(local_L, local_R, simpson(local_L, local_R));

    //发送结果给0号进程
    MPI_Reduce(&local_int, &total_int, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    endTime = MPI_Wtime();

    if (my_rank == 0)
    {
        cout << "integral from " << L << " to " << R << " =" << total_int << "\n";
        cout << " Tp:" << endTime - startTime << endl;
    }

    MPI_Finalize();
    return 0;
}
