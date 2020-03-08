#include <stdio.h>
#include <cmath>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>

__host__ int *gen_linked_list_1(int N)
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
__host__ void swap(int *list, int i, int j)
{
	if (i < 0 || j < 0 || i == j)
		return;

	int p = list[i]; //保存i后继元素下标p
	int q = list[j]; //保存j后继元素下标q

	if (p == -1 || q == -1)
		return; //如果有一个没有后继元素

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

__host__ int *gen_linked_list_2(int N)
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
__global__ void my_order(int N, int *a, int *b, int allnum)
{
	int part = N / allnum;
	for (int n = part * threadIdx.y; n < part * (threadIdx.y + 1); n++)
	{
		if (a[n] != -1)
			b[a[n]] = n;
		else
			b[N] = n;
	}
}

int main(void)
{

	int N = 10000000;
	int *a, *b;
	int *qq = NULL;
	qq = gen_linked_list_2(N);
	int *result = (int *)malloc(sizeof(int) * (N + 1));

	cudaMalloc((void **)&a, sizeof(int) * N);
	cudaMalloc((void **)&b, sizeof(int) * (N + 1));

	int allnum = 1 * 1 * 1 * 100;
	
	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);//计时开始
	
	dim3 grid(1, 1);
	dim3 block(1, 100);
	cudaMemcpy(a, qq, sizeof(int) * N, cudaMemcpyHostToDevice);
	my_order<<<grid, block>>>(N, a, b, allnum);
	cudaDeviceSynchronize();//同步
	cudaMemcpy(result, b, sizeof(int) * (N + 1), cudaMemcpyDeviceToHost);
	for (int n = 0; n < N; n++)
	{
		int hold = result[result[N]];
		result[result[N]] = n;
		result[N] = hold;
	}
	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);


	float totaltime = 0.0f;
	cudaEventElapsedTime(&totaltime, start1, stop1);
	printf("List ranking cuda time = %f\n", totaltime);

	cudaFree(a); 
	cudaFree(b);
	free(result);
	free(qq);
	return 0;
}