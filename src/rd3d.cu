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

// add auxiliary cuda functions
#include "check_cuda.h"
#include "cuda_events.h"

// include kernels
#include "kernels/laplacians.h"
#include "kernels/reaction_gray_scott.h"
#include "kernels/update.h"

// other includes
#include "rd3d.h"

RD3D::RD3D() {

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

    unsigned shared_mem_size = 0;
    if(this->zeroflux) {
        shared_mem_size = this->pencils * this->mx * sizeof(float);
    } else {
        shared_mem_size = this->pencils * (this->mx + 2) * sizeof(float);
    }
    std::cout << donestring << std::endl << std::endl;

    // keep track of time
    cudaEvent_t startEvent, stopEvent;
    cudaEvent_t startEventKernel, stopEventKernel;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    checkCuda( cudaEventCreate(&startEventKernel) );
    checkCuda( cudaEventCreate(&stopEventKernel) );

    for(unsigned int t=0; t<this->timesteps; t++) {
        // start timer and execute
        checkCuda( cudaEventRecord(startEvent, 0) );

        // keep track of time
        float milliseconds = 0;
        float laplacian_x_times = 0;
        float laplacian_y_times = 0;
        float laplacian_z_times = 0;
        float reaction_times = 0;
        float update_times = 0;
        float laplacian_summation_times = 0;

        for(unsigned int i=0; i<this->tsteps; i++) {

            // calculate laplacian for A
            if(this->zeroflux) {
                // x2 derivative
                start_event(&startEventKernel);
                derivative_x2_zeroflux<<<gridx,blockx,shared_mem_size>>>(d_a, d_dx2);
                laplacian_x_times += stop_event(&startEventKernel, &stopEventKernel);

                // y2 derivative
                start_event(&startEventKernel);
                derivative_y2_zeroflux<<<gridy,blocky,shared_mem_size>>>(d_a, d_dy2);
                laplacian_y_times += stop_event(&startEventKernel, &stopEventKernel);

                // z2 derivative
                start_event(&startEventKernel);
                derivative_z2_zeroflux<<<gridz,blockz,shared_mem_size>>>(d_a, d_dz2);
                laplacian_z_times += stop_event(&startEventKernel, &stopEventKernel);;
            } else {
                // x2 derivative
                start_event(&startEventKernel);
                derivative_x2_pbc<<<gridx,blockx,shared_mem_size>>>(d_a, d_dx2);
                laplacian_x_times += stop_event(&startEventKernel, &stopEventKernel);;

                // y2 derivative
                start_event(&startEventKernel);
                derivative_y2_pbc<<<gridy,blocky,shared_mem_size>>>(d_a, d_dy2);
                laplacian_y_times += stop_event(&startEventKernel, &stopEventKernel);;

                // z2 derivative
                start_event(&startEventKernel);
                derivative_z2_pbc<<<gridz,blockz,shared_mem_size>>>(d_a, d_dz2);
                laplacian_z_times += stop_event(&startEventKernel, &stopEventKernel);;
            }

            // sum all three derivative components
            start_event(&startEventKernel);
            construct_laplacian_a<<<grid,block>>>(d_da, d_dx2, d_dy2, d_dz2);
            laplacian_summation_times += stop_event(&startEventKernel, &stopEventKernel);;

            // calculate laplacian for B
            if(this->zeroflux) {
                // x2 derivative
                start_event(&startEventKernel);
                derivative_x2_zeroflux<<<gridx,blockx,shared_mem_size>>>(d_b, d_dx2);
                laplacian_x_times += stop_event(&startEventKernel, &stopEventKernel);;

                // y2 derivative
                start_event(&startEventKernel);
                derivative_y2_zeroflux<<<gridy,blocky,shared_mem_size>>>(d_b, d_dy2);
                laplacian_y_times += stop_event(&startEventKernel, &stopEventKernel);;

                // z2 derivative
                start_event(&startEventKernel);
                derivative_z2_zeroflux<<<gridz,blockz,shared_mem_size>>>(d_b, d_dz2);
                laplacian_z_times += stop_event(&startEventKernel, &stopEventKernel);;
            } else {
                // x2 derivative
                start_event(&startEventKernel);
                derivative_x2_pbc<<<gridx,blockx,shared_mem_size>>>(d_b, d_dx2);
                laplacian_x_times += stop_event(&startEventKernel, &stopEventKernel);;

                // y2 derivative
                start_event(&startEventKernel);
                derivative_y2_pbc<<<gridy,blocky,shared_mem_size>>>(d_b, d_dy2);
                laplacian_y_times += stop_event(&startEventKernel, &stopEventKernel);;

                // z2 derivative
                start_event(&startEventKernel);
                derivative_z2_pbc<<<gridz,blockz,shared_mem_size>>>(d_b, d_dz2);
                laplacian_z_times += stop_event(&startEventKernel, &stopEventKernel);;
            }

            // sum all three derivative components
            start_event(&startEventKernel);
            construct_laplacian_b<<<grid,block>>>(d_db, d_dx2, d_dy2, d_dz2);
            laplacian_summation_times += stop_event(&startEventKernel, &stopEventKernel);;

            // calculate reaction
            start_event(&startEventKernel);
            reaction_gray_scott<<<grid,block>>>(d_a, d_b, d_ra, d_rb);
            reaction_times += stop_event(&startEventKernel, &stopEventKernel);;

            // update
            start_event(&startEventKernel);
            update<<<grid,block>>>(d_a, d_b, d_da, d_db, d_ra, d_rb);
            update_times += stop_event(&startEventKernel, &stopEventKernel);;;
        }

        // stop timer
        checkCuda( cudaEventRecord(stopEvent, 0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&milliseconds, startEvent, stopEvent) );

        // copy results back
        int bytes = this->ncells * sizeof(float);
        checkCuda( cudaMemcpy(this->a, this->d_a, bytes, cudaMemcpyDeviceToHost) );
        checkCuda( cudaMemcpy(this->b, this->d_b, bytes, cudaMemcpyDeviceToHost) );

        auto start = std::chrono::system_clock::now();
        char buffer[50];
        sprintf(buffer, "data_%03i.bin", (t+1));
        this->write_binary(std::string(buffer), b);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        float other_times = milliseconds - laplacian_x_times - laplacian_y_times - laplacian_z_times -
                            reaction_times - update_times - laplacian_summation_times;

        printf("------------------ %04i ------------------\n", (t+1));
        printf("  Integration time: %12.6f ms\n", milliseconds);
        printf("  Laplacian x:      %12.6f ms\n", laplacian_x_times);
        printf("  Laplacian y:      %12.6f ms\n", laplacian_y_times);
        printf("  Laplacian z:      %12.6f ms\n", laplacian_z_times);
        printf("  Laplacian sum:    %12.6f ms\n", laplacian_summation_times);
        printf("  Reaction:         %12.6f ms\n", reaction_times);
        printf("  Time step:        %12.6f ms\n", update_times);
        printf("  Other:            %12.6f ms\n\n", other_times);
        printf("  Writing results to: %s\n", buffer);
        printf("  Writing data:     %12.6f ms\n", elapsed_seconds.count() * 1000);
        printf("------------------------------------------\n");
        printf("\n");
    }

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
