#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <time.h>
#include <device_functions.h>


__global__ void d_convolve_GPU(float *f, float *k , float *c,  int l_f, int l_k, int l_half)
{
	// get index and check if thread is in
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < l_f)
	{
		// clean the convolved value
		float val = 0.0;
		int check;
		// Cycle elements in kernel
		for (int j = 0; j < l_k; j++)
		{
			check = idx - l_half + j; // this is needed to ensure we dont attempt to index
			if (check > 0 && check < l_f)
			{
				val = val + f[idx - l_half + j]*k[j];
			}	
		}
		c[idx] = val;
	}
}

void convolve_CPU(float *f, float *k , float *c,  int l_f, int l_k, int l_half)
{
	float val;
	int check;
	// Cycle elements in array
	for (int i = 0; i< l_f; i++)
	{
		// clean the convolved value
		val = 0.0;
		
		// Cycle elements in kernel
		for (int j = 0; j < l_k; j++)
		{
			check = i - l_half + j; // this is needed to ensure we dont attempt to index
			if (check > 0 && check < l_f)
			{
				val = val + f[i - l_half + j]*k[j];
			}	
		}
		c[i] = val;
	}
}



int main()
{

	int n = 100;
	float f[n];
	float k[50];
	float c[n];
	float c_[n];

	////////////////////////////////
	// Initialise the flux arrays //
	////////////////////////////////
	for (int i = 0; i<n; i++)
	{
		f[i] = 1.0;
		c[i] = 0.0;
		c_[i] = 0.0;
	}

	///////////////////////////
	// Initialise the kernel //
	///////////////////////////
	
	for (int i = 0; i<50; i++)
	{
		k[i] = 0.02;
	}


	int l_f = sizeof(f)/sizeof(f[0]);
	int l_k = sizeof(k)/sizeof(k[0]);
	int l_half = l_k/2;

	////////////////////////////////////////
	// Make the convolution call with CPU //
	////////////////////////////////////////
	clock_t start = clock(), diff;
	for (int y = 0; y < 100; y++)
	{
		convolve_CPU(f, k , c, l_f, l_k, l_half);
	}

	diff = clock() - start;
	int msec_CPU = diff * 1000 / CLOCKS_PER_SEC;


	////////////////////////////////////////
	// Make the convolution call with GPU //
	////////////////////////////////////////

	float *d_f, *d_k, *d_c_;

	cudaMalloc( &d_f, l_f*sizeof(float));
	cudaMalloc( &d_k, l_k*sizeof(float));
	cudaMalloc( &d_c_, l_f*sizeof(float));

	cudaMemcpy( d_f, f, l_f*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( d_k, k, l_k*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy( d_c_, c_, l_f*sizeof(float), cudaMemcpyHostToDevice);
	clock_t start1 = clock(), diff1;
	for (int y = 0; y < 100; y++)
	{
		d_convolve_GPU <<< ceil(l_f / 256.0), 256 >>> (d_f, d_k , d_c_, l_f, l_k, l_half);
	}
	cudaMemcpy( c_, d_c_, l_f*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_f);
	cudaFree(d_k);
	cudaFree(d_c_);

	diff1 = clock() - start1;
	int msec_GPU = diff1 * 1000 / CLOCKS_PER_SEC;

	/////////////////////////////////////////
	// Now print out the first 100 results //
	/////////////////////////////////////////
	printf("Element     CPU     GPU");
	for (int i = 0; i<10; i++)
	{
		printf("c[%d] = %f  | %f\n", i, c[i], c_[i]);
	}
	printf("...\n--------------------------------\n");
	printf("CPU =  %d ms\n", msec_CPU%1000);
	printf("GPU =  %d ms\n", msec_GPU%1000);


	
	return 0;
}

