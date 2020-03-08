#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <mpi.h>
#include <math.h>
#include <iostream>
#include <cmath>
using namespace std;

int *gen_linked_list_1(int N)
{
	int *list = NULL;
	if (NULL != list)
	{
		free(list);
		list = NULL;
	}
	if (0 == N)
	{
		printf("N is 0, exit\n");
		exit(-1);
	}
	list = (int *)malloc(N * sizeof(int));
	if (NULL == list)
	{
		printf("Can not allocate memory for output array\n");
		exit(-1);
	}
	int i;
	for (i = 0; i < N; i++)
		list[i] = i - 1;

	return list;
}

//i和j的后继元素交换位置
void swap(int *list, int i, int j)
{
	if (i < 0 || j < 0 || i == j)
		return;
	int p = list[i]; //保存i后继元素下标p
	int q = list[j]; //保存j后继元素下标q
	if (p == -1 || q == -1)
		return;			 //如果有一个没有后继元素
	int pnext = list[p]; //保存p的后继元素下标
	int qnext = list[q]; //保存q的后继元素下标

	//i,j的后继元素交换位置
	if (p == j)
	{ //j是i的后继
		list[i] = q;
		list[j] = list[q];
		list[q] = j;
	}
	else if (i == q)
	{ //i是j的后继
		list[j] = p;
		list[i] = list[p];
		list[p] = i;
	}
	else
	{
		list[i] = q;	 //i的后继改为q
		list[j] = p;	 //j的后继改为p
		list[p] = qnext; //p的后继元素改为原来q的后继
		list[q] = pnext; //q的后继元素改为原来p的后继
	}
}

int *gen_linked_list_2(int N)
{
	int *list;
	list = gen_linked_list_1(N);
	int p = N / 5;
	int i;
	for (i = 0; i < N; i += 2)
	{
		int k = (i + i + p) % N;
		swap(list, i, k);
	}
	return list;
}

int main()
{
	double start, end;
	int comm_sz, my_rank;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int N = 10000000;
	int part = N / comm_sz;
	int *qq = NULL, *re = NULL;
	int *buf = (int *)malloc(sizeof(int) * (N + 1));
	memset(buf, 0, sizeof(int) * (N + 1));
	if (my_rank == 0)
	{
		qq = gen_linked_list_2(N);
		re = (int *)malloc(sizeof(int) * (N + 1));
	}
	else
	{
		qq = (int *)malloc(sizeof(int) * N);
	}
	MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
	start = MPI_Wtime();
	MPI_Bcast(qq, N, MPI_INT, 0, MPI_COMM_WORLD);
	for (int n = part * my_rank; n < part * (my_rank + 1); n++)
	{
		if (qq[n] != -1)
			buf[qq[n]] = n;
		else
			buf[N] = n;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(buf, re, N + 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	if (my_rank == 0)
	{
		for (int n = 0; n < N; n++)
		{
			int hold = re[re[N]];
			re[re[N]] = n;
			re[N] = hold;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
	end = MPI_Wtime();
	free(buf);
	if (my_rank == 0)
	{
		printf("List ranking mpi time = %f\n", end - start);
		free(qq);
		free(re);
	}
	MPI_Finalize();
}
