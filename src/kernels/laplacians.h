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

#ifndef _LAPLACIANS_H
#define _LAPLACIANS_H

// constant variables on device
__device__ __constant__ float d_diffcon_a;
__device__ __constant__ float d_diffcon_b;
__device__ __constant__ unsigned int d_mx;
__device__ __constant__ unsigned int d_my;
__device__ __constant__ unsigned int d_mz;
__device__ __constant__ unsigned int d_pencils;
__device__ __constant__ unsigned int d_ncells;

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
 * @brief      Calculate second derivative in x direction with zero-flux boundary conditions
 *
 * @param[in]  f     pointer with values
 * @param      df    pointer with second derivatives
 */
__global__ void derivative_x2_zeroflux(const float *f, float *df) {
    extern __shared__ float s_f[];

    int i   = threadIdx.x;
    int j   = blockIdx.x * blockDim.y + threadIdx.y;
    int k   = blockIdx.y;
    int sj  = threadIdx.y; // local j for shared memory access

    int globalIdx = k * d_mx * d_my + j * d_mx + i;

    s_f[sj * d_mx + i] = f[globalIdx];

    __syncthreads();

    if(i == 0) {
        df[globalIdx] = s_f[sj * d_mx + i + 1] - s_f[sj * d_mx + i];
    } else if(i == (d_mx - 1)) {
        df[globalIdx] = s_f[sj * d_mx + i - 1] - s_f[sj * d_mx + i];
    } else {
        df[globalIdx] = s_f[sj * d_mx + i + 1] - 2.0 * s_f[sj * d_mx + i] + s_f[sj * d_mx + i - 1];
    }
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
 * @brief      Calculate second derivative in y direction with zero-flux  boundary conditions
 *
 * @param[in]  f     pointer with values
 * @param      df    pointer with second derivatives
 */
__global__ void derivative_y2_zeroflux(const float *f, float *df) {
    extern __shared__ float s_f[];

    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int j  = threadIdx.y;
    int k  = blockIdx.y;
    int si = threadIdx.x;

    int globalIdx = k * d_mx * d_my + j * d_mx + i;

    s_f[j * d_pencils + si] = f[globalIdx];

    __syncthreads();

    if(j == 0) {
        df[globalIdx] = s_f[(j+1) * d_pencils + si] - s_f[j * d_pencils + si];
    } else if(j == (d_my - 1)) {
        df[globalIdx] = s_f[(j-1) * d_pencils + si] - s_f[j * d_pencils + si];
    } else {
        df[globalIdx] = s_f[(j+1) * d_pencils + si] - 2.0 * s_f[j * d_pencils + si] + s_f[(j-1) * d_pencils + si];
    }
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
 * @brief      Calculate second derivative in z direction with zero-flux boundary conditions
 *
 * @param[in]  f     pointer with values
 * @param      df    pointer with second derivatives
 */
__global__ void derivative_z2_zeroflux(const float *f, float *df) {
    extern __shared__ float s_f[]; // 2-wide halo

    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int j  = blockIdx.y;
    int k  = threadIdx.y;
    int si = threadIdx.x;

    int globalIdx = k * d_mx * d_my + j * d_mx + i;

    s_f[k * d_pencils + si] = f[globalIdx];

    __syncthreads();

    if(k == 0) {
        df[globalIdx] = s_f[(k+1) * d_pencils + si] - s_f[k * d_pencils + si];
    } else if(k == (d_mz - 1)) {
        df[globalIdx] = s_f[(k-1) * d_pencils + si] - s_f[k * d_pencils + si];
    } else {
        df[globalIdx] = s_f[(k+1) * d_pencils + si] - 2.0 * s_f[k * d_pencils + si] + s_f[(k-1) * d_pencils + si];
    }
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

#endif // _LAPLACIAN_H
