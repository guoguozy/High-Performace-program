#include <bits/stdc++.h>
#include <omp.h>
#include <pthread.h>
using namespace std;

pthread_mutex_t mutex1, mutex2[1024][1024], mutex3, mutex4;

vector<vector<double> > res; //存放结果
vector<vector<vector<double> > > cube; //存放中间值

int counter = 0, counter1 = 0, counter_num = 0;
int n, num_threads;
int a_row_size, a_col_size, b_row_size, b_col_size;
int i_size, k_size, j_size;

struct Block
{
    Block(int i = 0, int k = 0, int j = 0) : i(i), k(k), j(j) {}
    int i, k, j;
};

void serial_work()
{
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                res[i][j] += (i - 0.1 * k + 1) / (i + k + 1) * ((j - 0.2 * k + 1) * (k + j + 1) / (k * k + j * j + 1));
}

void *parallel(void *node)
{
    Block *t = (Block *)node;
    int a_row = t->i * a_row_size;
    int a_col = t->k * a_col_size;
    int b_col = t->j * b_col_size;
    for (int x = 0; x < a_row_size; ++x)
        for (int y = 0; y < a_col_size; ++y)
            for (int z = 0; z < b_col_size; ++z)
            {
                int i = x + a_row;
                int k = y + a_col;
                int j = z + b_col;

                cube[i][k][j] = (i - 0.1 * k + 1) / (i + k + 1) * ((j - 0.2 * k + 1) * (k + j + 1) / (k * k + j * j + 1));
            }
    pthread_mutex_lock(&mutex1);
    counter++;
    pthread_mutex_unlock(&mutex1);
    while (counter < num_threads)
        ;
    return NULL;
}
void *add(void *node)
{
    Block *t = (Block *)node;
    int a_row = t->i * a_row_size;
    int a_col = t->k * a_col_size;
    int b_col = t->j * b_col_size;
    for (int x = 0; x < a_row_size; ++x)
        for (int y = 0; y < a_col_size; ++y)
            for (int z = 0; z < b_col_size; ++z)
            {
                int i = x + a_row;
                int k = y + a_col;
                int j = z + b_col;
                pthread_mutex_lock(&mutex2[i][j]);
                res[i][j] += cube[i][k][j];
                pthread_mutex_unlock(&mutex2[i][j]);
            }
    return NULL;
}

void malloc_for_multi()
{
    //malloc_for_cube
    cube.resize(n);
    for (int i = 0; i < n; i++)
        cube[i].resize(n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cube[i][j].resize(n, 0);

    //malloc_for_result_matrix
    res.resize(n);
    for (int i = 0; i < n; ++i)
        res[i].resize(n, 0);
}
int main(int arg_1, char **arg_2)
{
    num_threads = atoi(arg_2[1]), n = pow(2, atoi(arg_2[2]));

    cout << "num_threads: " << num_threads << "\nsize: " << n << "\n";

    malloc_for_multi();

    switch (num_threads)
    {
    case 1:
        i_size = k_size = j_size = 1;
        break;
    case 2:
        i_size = k_size = 1;
        j_size = 2;
        break;
    case 4:
        i_size = 1;
        k_size = j_size = 2;
        break;
    case 8:
        i_size = k_size = j_size = 2;
        break;
    case 16:
        i_size = k_size = 2;
        j_size = 4;
        break;
    case 32:
        i_size = 2;
        k_size = j_size = 4;
        break;
    case 64:
        i_size = 4;
        k_size = 4;
        j_size = 4;
        break;
    default:
        break;
    }

    pthread_t threads[i_size][k_size][j_size];
    cout << i_size << " " << k_size << " " << j_size << "\n";

    a_row_size = n / i_size;
    a_col_size = b_row_size = n / k_size;
    b_col_size = n / j_size;

    cout << a_row_size << " " << a_col_size << " " << b_col_size << "\n";

    double start1=omp_get_wtime();
    serial_work();
    double finish1=omp_get_wtime();
    double serial_elapsed = finish1 - start1;

    double start=omp_get_wtime();
    /* Multi */
    for (int i = 0; i < i_size; ++i)
        for (int k = 0; k < k_size; ++k)
            for (int j = 0; j < j_size; ++j)
            {
                Block node = Block(i, k, j);
                pthread_create(&threads[i][k][j], NULL, parallel, (void *)&node);
            }
    for (int i = 0; i < i_size; ++i)
        for (int k = 0; k < k_size; ++k)
            for (int j = 0; j < j_size; ++j)
                pthread_join(threads[i][k][j], NULL);
    /* Add */

    for (int i = 0; i < i_size; ++i)
        for (int k = 0; k < k_size; ++k)
            for (int j = 0; j < j_size; ++j)
            {
                Block node = Block(i, k, j);
                pthread_create(&threads[i][k][j], NULL, add, (void *)&node);
            }
    for (int i = 0; i < i_size; ++i)
        for (int k = 0; k < k_size; ++k)
            for (int j = 0; j < j_size; ++j)
                pthread_join(threads[i][k][j], NULL);
    double finish =omp_get_wtime();
    double elapsed = finish - start;

   
    /* Show  */
    cout << "serial time: " << serial_elapsed << " seconds\n";
    cout << "parallel time: " << elapsed << " seconds\n";

    cout << "Speedup:" << (double)serial_elapsed / elapsed << "\n";
    cout << "Efficiency: "
         << (double) serial_elapsed / elapsed / num_threads << "\n";

    pthread_mutex_destroy(&mutex1);
    for (int i = 0; i < 1024; ++i)
        for (int j = 0; j < 1024; ++j)
            pthread_mutex_destroy(&mutex2[i][j]);
    pthread_mutex_destroy(&mutex3);
    pthread_mutex_destroy(&mutex4);
    return 0;
}