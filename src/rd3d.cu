 /*************************************************************************
 *   This file is part of Wavefuse                                        *
 *   https://gitlab.tue.nl/ifilot/Wavefuse                                *
 *                                                                        *
 *   Author: Ivo Filot <i.a.w.filot@tue.nl>                               *
 *                                                                        *
 *   Wavefuse is free software: you can redistribute it and/or modify     *
 *   it under the terms of the GNU General Public License as published    *
 *   by the Free Software Foundation, either version 3 of the License,    *
 *   or (at your option) any later version.                               *
 *                                                                        *
 *   Wavefuse is distributed in the hope that it will be useful,          *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty          *
 *   of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.              *
 *   See the GNU General Public License for more details.                 *
 *                                                                        *
 *   You should have received a copy of the GNU General Public License    *
 *   along with this program.  If not, see http://www.gnu.org/licenses/.  *
 *                                                                        *
 **************************************************************************/

#include "rd3d.h"

// constant variables on device
__device__ __constant__ float d_diffcon_a;
__device__ __constant__ float d_diffcon_b;
__device__ __constant__ float d_dt;
__device__ __constant__ unsigned int d_mx;
__device__ __constant__ unsigned int d_my;
__device__ __constant__ unsigned int d_mz;
__device__ __constant__ unsigned int d_pencils;
__device__ __constant__ unsigned int d_ncells;
__device__ __constant__ float d_f;
__device__ __constant__ float d_k;

RD3D::RD3D() {

}

/**
 * @brief      Check whether there was a cuda error and report error
 *
 * @param[in]  result  The result
 *
 * @return     cuda result
 */
inline cudaError_t checkCuda(cudaError_t result) {
    #if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    #endif

    return result;
}

/**
 * @brief      Calculate second derivative in x direction with periodic boundary conditions
 *
 * @param[in]  f     pointer with values
 * @param      df    pointer with second derivatives
 */
__global__ void derivative_x2_pbc(const float *f, float *df) {
    const int offset = 1;
    extern __shared__ float s_f[]; // 2-wide halo

    int i   = threadIdx.x;
    int j   = blockIdx.x * blockDim.y + threadIdx.y;
    int k   = blockIdx.y;
    int si  = i + offset;  // local i for shared memory access + halo offset
    int sj  = threadIdx.y; // local j for shared memory access

    int globalIdx = k * d_mx * d_my + j * d_mx + i;

    s_f[sj * (d_mx + 2 * offset) + si] = f[globalIdx];

    __syncthreads();

    // fill in periodic images in shared memory array
    if (i < offset) {
        s_f[sj * (d_mx + 2 * offset) + si - offset]  = s_f[sj * (d_mx + 2 * offset) + si + d_mx - offset];
        s_f[sj * (d_mx + 2 * offset) + si + d_mx] = s_f[sj * (d_mx + 2 * offset) + si];
    }

    __syncthreads();

    df[globalIdx] = s_f[sj * (d_mx + 2 * offset) + si + 1] - 2.0 * s_f[sj * (d_mx + 2 * offset) + si] + s_f[sj * (d_mx + 2 * offset) + si - 1];
}

/**
 * @brief      Calculate second derivative in y direction with periodic boundary conditions
 *
 * @param[in]  f     pointer with values
 * @param      df    pointer with second derivatives
 */
__global__ void derivative_y2_pbc(const float *f, float *df) {
    const int offset = 1;
    extern __shared__ float s_f[]; // 2-wide halo

    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int j  = threadIdx.y;
    int k  = blockIdx.y;
    int si = threadIdx.x;
    int sj = j + offset;

    int globalIdx = k * d_mx * d_my + j * d_mx + i;

    s_f[sj * d_pencils + si] = f[globalIdx];

    __syncthreads();

    // fill in periodic images in shared memory array
    if (j < offset) {
        s_f[(sj - offset) * d_pencils + si]  = s_f[(sj + d_my - offset) * d_pencils + si];
        s_f[(sj + d_my) * d_pencils + si] = s_f[sj * d_pencils + si];
    }

    __syncthreads();

    df[globalIdx] = s_f[(sj+1) * d_pencils + si] - 2.0 * s_f[sj * d_pencils + si] + s_f[(sj-1) * d_pencils + si];
}

/**
 * @brief      Calculate second derivative in z direction with periodic boundary conditions
 *
 * @param[in]  f     pointer with values
 * @param      df    pointer with second derivatives
 */
__global__ void derivative_z2_pbc(const float *f, float *df) {
    const int offset = 1;
    extern __shared__ float s_f[]; // 2-wide halo

    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int j  = blockIdx.y;
    int k  = threadIdx.y;
    int si = threadIdx.x;
    int sk = k + offset; // halo offset

    int globalIdx = k * d_mx * d_my + j * d_mx + i;

    s_f[sk * d_pencils + si] = f[globalIdx];

    __syncthreads();

    // fill in periodic images in shared memory array
    if (k < offset) {
        s_f[(sk - offset) * d_pencils + si]  = s_f[(sk + d_mz - offset) * d_pencils + si];
        s_f[(sk + d_mz) * d_pencils + si] = s_f[sk * d_pencils + si];
    }

    __syncthreads();

    df[globalIdx] = s_f[(sk+1) * d_pencils + si] - 2.0 * s_f[sk * d_pencils + si] + s_f[(sk-1) * d_pencils + si];
}

/**
 * @brief      Construct the Laplacian for component A
 *
 * @param      df    pointer to laplacian values
 * @param[in]  dfx   pointer to second derivative in x direction
 * @param[in]  dfy   pointer to second derivative in y direction
 * @param[in]  dfz   pointer to second derivative in z direction
 */
__global__ void construct_laplacian_a(float *df, const float *dfx, const float *dfy, const float *dfz) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < d_ncells; i += stride) {
        df[i] = d_diffcon_a * (dfx[i] + dfy[i] + dfz[i]);
    }
}

/**
 * @brief      Construct the Laplacian for component B
 *
 * @param      df    pointer to laplacian values
 * @param[in]  dfx   pointer to second derivative in x direction
 * @param[in]  dfy   pointer to second derivative in y direction
 * @param[in]  dfz   pointer to second derivative in z direction
 */
__global__ void construct_laplacian_b(float *df, const float *dfx, const float *dfy, const float *dfz) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < d_ncells; i += stride) {
        df[i] = d_diffcon_b * (dfx[i] + dfy[i] + dfz[i]);
    }
}

/**
 * @brief      Calculate gray-scott reaction rate
 *
 * @param[in]  fx    pointer to concentration of compound A
 * @param[in]  fy    pointer to concentration of compound B
 * @param      drx   pointer to reaction rate of compound A
 * @param      dry   pointer to reaction rate of compound B
 */
__global__ void reaction(const float *fx, const float *fy, float *drx, float *dry) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < d_ncells; i += stride) {
        float r = fx[i] * fy[i] * fy[i];
        drx[i] = -r + d_f * (1.0 - fx[i]);
        dry[i] = r - (d_f + d_k) * fy[i];
    }
}

/**
 * @brief      Perform time-step integration
 *
 * @param      x     pointer to concentration of A
 * @param      y     pointer to concentration of B
 * @param[in]  ddx   diffusion of component A
 * @param[in]  ddy   diffusion of component B
 * @param[in]  drx   reaction of component A
 * @param[in]  dry   reaction of component B
 */
__global__ void update(float *x, float *y, const float *ddx, const float *ddy, const float *drx, const float *dry) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < d_ncells; i += stride) {
        x[i] += (ddx[i] + drx[i]) * d_dt;
        y[i] += (ddy[i] + dry[i]) * d_dt;
    }
}

/**
 * @brief      Run time-integration on GPU
 */
void RD3D::run_cuda() {
    this->initialize_variables();

    std::cout << "Starting time-integration" << std::endl;

    std::cout << "Setting grids and blocks...           ";
    dim3 gridx(this->my / this->pencils, this->mz, 1);
    dim3 blockx(this->mx, this->pencils, 1);
    dim3 gridy(this->mx / this->pencils, this->mz, 1);
    dim3 blocky(this->pencils, this->my, 1);
    dim3 gridz(this->mx / this->pencils, this->my, 1);
    dim3 blockz(this->pencils, this->mz, 1);
    unsigned int block = this->mx;;
    unsigned int grid = (this->ncells + this->mx - 1) / this->mx;
    unsigned shared_mem_size = this->pencils * (this->mx + 2 * 1) * sizeof(float);
    std::cout << donestring << std::endl;

    // keep track of time
    cudaEvent_t startEvent, stopEvent;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );

    printf("+--------+-------------------------------|\n");
    printf("|  Step  |  Integration wall-clock time  |\n");
    printf("+--------+-------------------------------|\n");

    for(unsigned int t=0; t<this->timesteps; t++) {
        // start timer and execute
        checkCuda( cudaEventRecord(startEvent, 0) );

        for(unsigned int i=0; i<this->tsteps; i++) {
            // calculate laplacian for A
            derivative_x2_pbc<<<gridx,blockx,shared_mem_size>>>(d_a, d_dx2);
            derivative_y2_pbc<<<gridy,blocky,shared_mem_size>>>(d_a, d_dy2);
            derivative_z2_pbc<<<gridz,blockz,shared_mem_size>>>(d_a, d_dz2);
            construct_laplacian_a<<<grid,block>>>(d_da, d_dx2, d_dy2, d_dz2);

            // calculate laplacian for B
            derivative_x2_pbc<<<gridx,blockx,shared_mem_size>>>(d_b, d_dx2);
            derivative_y2_pbc<<<gridy,blocky,shared_mem_size>>>(d_b, d_dy2);
            derivative_z2_pbc<<<gridz,blockz,shared_mem_size>>>(d_b, d_dz2);
            construct_laplacian_b<<<grid,block>>>(d_db, d_dx2, d_dy2, d_dz2);

            // // calculate reaction
            reaction<<<grid,block>>>(d_a, d_b, d_ra, d_rb);

            // // update
            update<<<grid,block>>>(d_a, d_b, d_da, d_db, d_ra, d_rb);
        }

        // stop timer
        float milliseconds;
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );

        // copy results back
        int bytes = this->ncells * sizeof(float);
        checkCuda( cudaMemcpy(this->a, this->d_a, bytes, cudaMemcpyDeviceToHost) );
        checkCuda( cudaMemcpy(this->b, this->d_b, bytes, cudaMemcpyDeviceToHost) );

        char buffer[50];
        sprintf(buffer, "data_%03i.bin", (t+1));
        this->write_binary(std::string(buffer), b);

        printf("|  %04i  |              %12.6f ms  |\n", (t+1), milliseconds);
    }
    printf("+--------+-------------------------------|\n");

    // clean up
    checkCuda( cudaEventDestroy(startEvent) );
    checkCuda( cudaEventDestroy(stopEvent) );

    std::cout << std::endl;
    this->cleanup_variables();
}

/**
 * @brief      Initialize all variables
 */
void RD3D::initialize_variables() {
    std::cout << "Loading device variables." << std::endl;

    // build initial concentrations
    std::cout << "Constructing initial concentrations...";
    this->a = new float[this->ncells];
    this->b = new float[this->ncells];
    this->build_input(a,b);
    std::cout << donestring << std::endl;

    // allocate size on device
    std::cout << "Allocating variables on GPU device... ";
    int bytes = this->ncells * sizeof(float);
    checkCuda( cudaMalloc((void**)&this->d_a, bytes) );
    checkCuda( cudaMalloc((void**)&this->d_b, bytes) );
    checkCuda( cudaMalloc((void**)&this->d_dx2, bytes) );
    checkCuda( cudaMalloc((void**)&this->d_dy2, bytes) );
    checkCuda( cudaMalloc((void**)&this->d_dz2, bytes) );
    checkCuda( cudaMalloc((void**)&this->d_ra, bytes) );
    checkCuda( cudaMalloc((void**)&this->d_rb, bytes) );
    checkCuda( cudaMalloc((void**)&this->d_da, bytes) );
    checkCuda( cudaMalloc((void**)&this->d_db, bytes) );
    std::cout << donestring << std::endl;

    // copy data to device
    std::cout << "Copying data to GPU device...         ";
    checkCuda( cudaMemcpy(this->d_a, this->a, bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(this->d_b, this->b, bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemset(this->d_dx2, 0, bytes) );
    checkCuda( cudaMemset(this->d_dy2, 0, bytes) );
    checkCuda( cudaMemset(this->d_dz2, 0, bytes) );
    checkCuda( cudaMemset(this->d_ra, 0, bytes) );
    checkCuda( cudaMemset(this->d_rb, 0, bytes) );
    checkCuda( cudaMemset(this->d_da, 0, bytes) );
    checkCuda( cudaMemset(this->d_db, 0, bytes) );
    std::cout << donestring << std::endl;

    // set constants
    std::cout << "Setting constant variables on GPU...  ";
    float _diffcon_a = this->Da / (this->dx * this->dx);
    float _diffcon_b = this->Db / (this->dx * this->dx);
    checkCuda( cudaMemcpyToSymbol(d_diffcon_a, &_diffcon_a, sizeof(float)) );
    checkCuda( cudaMemcpyToSymbol(d_diffcon_b, &_diffcon_b, sizeof(float)) );
    checkCuda( cudaMemcpyToSymbol(d_dt, &this->dt, sizeof(float)) );
    checkCuda( cudaMemcpyToSymbol(d_mx, &this->mx, sizeof(unsigned int)) );
    checkCuda( cudaMemcpyToSymbol(d_my, &this->my, sizeof(unsigned int)) );
    checkCuda( cudaMemcpyToSymbol(d_mz, &this->mz, sizeof(unsigned int)) );
    checkCuda( cudaMemcpyToSymbol(d_pencils, &this->pencils, sizeof(unsigned int)) );
    checkCuda( cudaMemcpyToSymbol(d_ncells, &this->ncells, sizeof(unsigned int)) );
    checkCuda( cudaMemcpyToSymbol(d_f, &this->f, sizeof(float)) );
    checkCuda( cudaMemcpyToSymbol(d_k, &this->k, sizeof(float)) );
    std::cout << donestring << std::endl;

    std::cout << "All ready for time-integration." << std::endl << std::endl;
}

/**
 * @brief      Clean-up all variables
 */
void RD3D::cleanup_variables() {
    std::cout << "Cleaning Integration variables...     ";
    checkCuda( cudaFree(this->d_a) );
    checkCuda( cudaFree(this->d_b) );
    checkCuda( cudaFree(this->d_ra) );
    checkCuda( cudaFree(this->d_rb) );
    checkCuda( cudaFree(this->d_da) );
    checkCuda( cudaFree(this->d_db) );
    checkCuda( cudaFree(this->d_dx2) );
    checkCuda( cudaFree(this->d_dy2) );
    checkCuda( cudaFree(this->d_dz2) );

    delete [] this->a;
    delete [] this->b;

    std::cout << donestring << std::endl;
    std::cout << std::endl;
}

/**
 * @brief      Build random input
 *
 * @param      a     Concentration of a
 * @param      b     Concentration of b
 */
void RD3D::build_input(float* a, float* b) {
    // initialize with random data
    const float delta = 0.05f;
    for(unsigned int i=0; i < this->ncells; i++) {
        a[i] = 1.0 + uniform_dist() * delta;
        b[i] = 0.0 + uniform_dist() * delta;
    }

    const unsigned int cbsz = 5;
    for(unsigned int z=this->mz/2-cbsz; z<this->mz/2+cbsz; z++) {
        for(unsigned int y=this->my/2-cbsz; y<this->my/2+cbsz; y++) {
            for(unsigned int x=this->mx/2-cbsz; x<this->mx/2+cbsz; x++) {
                a[z * this->mx * this->my + y * this->mx + x] = 0.5f  + uniform_dist() * delta;
                b[z * this->mx * this->my + y * this->mx + x] = 0.25f  + uniform_dist() * delta;
            }
        }
    }
}

/**
 * @brief      Write 3D concentration profile as binary file
 *
 * @param[in]  filename  The filename
 * @param[in]  vals      Concentration data
 */
void RD3D::write_binary(const std::string filename, const float *vals) {
    std::ofstream out(filename, std::ios::binary);

    // write data size
    uint16_t dim = 0;

    // write size
    dim = this->mx;
    out.write((char*)&dim, sizeof(uint16_t));
    dim = this->my;
    out.write((char*)&dim, sizeof(uint16_t));
    dim = this->mz;
    out.write((char*)&dim, sizeof(uint16_t));
    dim = sizeof(float);
    out.write((char*)&dim, sizeof(uint16_t));

    // write values
    out.write((const char*)vals, sizeof(float) * this->ncells);

    out.close();
}
