#include <string.h>
#include <stdio.h>
#include <time.h>

static const int M = 8192;
static const int N = 8192;
int block_x=16;
int block_y=16;
int grid_x=512;
int grid_y=512;
struct data{
	int x, y;
	float data;
};

__global__ void matrix_multi(float *A,float *B,float *C,int pitch)
{
    
    int Row=blockIdx.y*blockDim.y+threadIdx.y;
	int Col=blockIdx.x*blockDim.x+threadIdx.x;
	
	int thread_id=Row*blockDim.x*gridDim.x+Col; 
    int i, j;
    if(thread_id<M)
    {   
    	for(i=0;i<N;i++) 
    	{ 
			float sum=0;
			for(j=0;j<M;j++)
			{
				int h=pitch/sizeof(float);
				float *addr_a=&A[0]+j*h+thread_id;
				float *addr_b=&B[0]+j*h+i;
				sum+=(*addr_a)*(*addr_b);
			}
    	    C[thread_id*N+i]=sum;
		}
    }
}

int main()
{   
    float *h_a =(float *)malloc(M*N*sizeof(float));
    float *h_b =(float *)malloc(M*N*sizeof(float));
    float *h_c =(float *)malloc(M*N*sizeof(float));
    float *serial_c=(float *)malloc(M*N*sizeof(float));
    memset(h_a,0,M*N);
    memset(h_b,0,M*N);
	memset(serial_c,0,M*N);
	FILE* file;
    file = fopen("/public/home/st17341046/read_data.txt", "rb");
    while(!feof(file))
	{
		struct data c;
		fread(&c,sizeof(struct data),1,file);
		h_b[c.x*N+c.y]=c.data;
		h_a[c.y*N+c.x]=c.data;
	}
	fclose(file);
    int i,j,k;
	 
	 
    float *dev_b ;  
    float *dev_c ;  
	float *dev_a ;  
	size_t pitch=0;
	cudaMallocPitch((void**)&dev_a,&pitch,N*sizeof(float),M);  
	cudaMemcpy2D(dev_a, pitch, h_a, N* sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice); 
	cudaMallocPitch((void**)&dev_b,&pitch,N*sizeof(float),M);  
	cudaMemcpy2D(dev_b, pitch, h_b, N* sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice); 	
	cudaMalloc((void**)(&dev_c), M*N*sizeof(float));
 
    dim3 block(block_x,block_y);
    dim3 grid(grid_x,grid_y); 
	 
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    matrix_multi<<<grid,block>>>(dev_a,dev_b,dev_c,pitch);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("时间 %f ms\n",elapsedTime);
    cudaMemcpy((void*)(h_c), (void*)(dev_c), M*N*sizeof(float), cudaMemcpyDeviceToHost);


    file = fopen("/public/home/st17341056/write_data.txt", "wb");
	for(i=0;i<M;i++)
	    for(j=0;j<N;j++)
		{
			if(h_c[i*N+j]==0) continue;
			else
			{
				struct data c;
				c.x=i,
				c.y=j;
				c.data=h_c[i*N+j];
				fwrite(&c,sizeof(struct data),1,file);
			}
		}
	fclose(file);

    cudaFree((void*)dev_a);
    cudaFree((void*)dev_b);
    cudaFree((void*)dev_c);
    free(h_a);
    free(h_c);
    free(h_b);
    return 0;
}
