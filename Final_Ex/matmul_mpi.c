#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
int N = 1 << 13;

struct matrix
{
	int x, y;
	float matrix;
};
	
int main()
{
	int comm_sz, my_rank;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	float *matrix_A = (float *)malloc(sizeof(float) * N * N);
	float *matrix_B = (float *)malloc(sizeof(float) * N * N);
	memset(matrix_A, 0, N * N);
	memset(matrix_B, 0, N * N);
	if (my_rank == 0)
	{
		FILE *file;
		file = fopen("/public/home/st17341046/read_matrix.txt", "rb");
		while (!feof(file))
		{
			struct matrix c;
			fread(&c, sizeof(struct matrix), 1, file);

			matrix_A[c.x * N + c.y] = c.matrix;
			matrix_B[c.x * N + c.y] = c.matrix;
		}
		fclose(file);
	}
	MPI_Bcast(matrix_A, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(matrix_B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if (comm_sz == 1 || comm_sz == 4 || comm_sz == 16 || comm_sz == 64 || comm_sz == 256)
	{

		int a = sqrt(comm_sz);
		int sub_N = N / a;

		int col, row;
		MPI_Comm col_comm, row_comm;
		col = my_rank % a;
		row = my_rank / a;
		MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);
		MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);

		float *A = (float *)malloc(sizeof(float) * sub_N * sub_N);
		float *B = (float *)malloc(sizeof(float) * sub_N * sub_N);
		int i, j;
		int i_index, j_index;
		for (i = 0; i < sub_N; i++)
		{
			i_index = row * sub_N + i;
			for (j = 0; j < sub_N; j++)
			{
				j_index = col * sub_N + j;
				A[i * sub_N + j] = matrix_A[i_index * N + j_index];
				B[i * sub_N + j] = matrix_B[i_index * N + j_index];
			}
		}
		float *t_A = (float *)malloc(sizeof(float) * sub_N * N);
		MPI_Allgather(A, sub_N * sub_N, MPI_FLOAT, t_A, sub_N * sub_N, MPI_FLOAT, row_comm);

		float *t_B = (float *)malloc(sizeof(float) * sub_N * N);
		MPI_Allgather(B, sub_N * sub_N, MPI_FLOAT, t_B, sub_N * sub_N, MPI_FLOAT, col_comm);
		float *sub_C = (float *)malloc(sizeof(float) * sub_N * sub_N);
		for (i = 0; i < sub_N; i++)
			for (j = 0; j < sub_N; j++)
			{
				sub_C[i * sub_N + j] = 0;
			}
		int k, count;
		for (count = 0; count < a; count++)
		{
			for (i = 0; i < sub_N; i++)
				for (k = 0; k < sub_N; k++)
				{
					if (t_A[count * sub_N * sub_N + i * sub_N + k] == 0)
						continue;
					for (j = 0; j < sub_N; j++)
					{
						if (t_B[count * sub_N * sub_N + k * sub_N + j] == 0)
							continue;
						sub_C[i * sub_N + j] += t_A[count * sub_N * sub_N + i * sub_N + k] * t_B[count * sub_N * sub_N + k * sub_N + j];
					}
				}
		}
		float *C = (float *)malloc(sizeof(float) * N * N);
		float *C1 = (float *)malloc(sizeof(float) * N * N);
		for (i = 0; i < sub_N; i++)
			for (j = 0; j < sub_N; j++)
			{
				int index = (row * sub_N + i) * N + col * sub_N + j;
				C[index] = sub_C[i * sub_N + j];
			}
		MPI_Reduce(C, C1, N * N, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

		if (my_rank == 0)
		{

			FILE *file;
			file = fopen("/public/home/st17341056/write_matrix.txt", "wb");
			for (i = 0; i < N; i++)
				for (j = 0; j < N; j++)
				{
					if (C[i * N + j] == 0)
						continue;
					else
					{
						struct matrix c;
						c.x = i,
						c.y = j;
						c.matrix = C[i * N + j];

						fwrite(&c, sizeof(struct matrix), 1, file);
					}
				}
			fclose(file);
		}
		free(matrix_A);
		free(t_A);
		free(A);
		free(matrix_B);
		free(t_B);
		free(B);
		free(sub_C);
		free(C1);
		free(C);
	}
	MPI_Finalize();
	return 0;
}