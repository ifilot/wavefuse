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

#ifndef _RD3D_H
#define _RD3D_H

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <random>
#include <fstream>

class RD3D {
private:
    unsigned int mx = 256;
    unsigned int my = 256;
    unsigned int mz = 256;
    unsigned int ncells = mx * my * mz;

    unsigned int pencils = 4;

    float *a;   //!< concentration of component A
    float *b;   //!< concentration of component B

    // device variables
    float *d_a, *d_b, *d_dx2, *d_dy2, *d_dz2, *d_ra, *d_rb, *d_da, *d_db;

    // reaction settings of kinetic system
    float f = 0.0392;           //!< reactivity constant f
    float k = 0.0649;           //!< reactivity constant f
    float Da = 0.16;            //!< diffusion constant of A
    float Db = 0.08;            //!< diffusion constant of B
    float dt = 0.25;            //!< temporal discretization
    float dx = 0.5;             //!< spatial discretization

    unsigned int timesteps = 720;
    unsigned int tsteps = 100;

    std::string donestring = "           [DONE]";

    bool zeroflux = true;

public:
    /**
     * @brief      Constructs the object.
     */
    RD3D();

    /**
     * @brief      Run time-integration on GPU
     */
    void run_cuda();

    /**
     * @brief      Set unit cell dimensions
     *
     * @param[in]  _mx   dimensionality for x
     * @param[in]  _my   dimensionality for y
     * @param[in]  _mz   dimensionality for z
     */
    void set_dimensions(unsigned int _mx, unsigned int _my, unsigned int _mz) {
        this->mx = _mx;
        this->my = _my;
        this->mz = _mz;
        this->ncells = this->mx * this->my * this->mz;
    }

    /**
     * @brief      Sets the integration variables.
     *
     * @param[in]  _dt         Time discretization
     * @param[in]  _dx         Distance discretization
     * @param[in]  _timesteps  Number of output timesteps
     * @param[in]  _tsteps     Number of internal timesteps
     */
    void set_integration_variables(float _dt, float _dx, unsigned int _timesteps, unsigned int _tsteps) {
        this->dt = _dt;
        this->dx = _dx;
        this->timesteps = _timesteps;
        this->tsteps = _tsteps;
    }

    /**
     * @brief      Sets the kinetic variables.
     *
     * @param[in]  _f    Gray-Scott parameter f
     * @param[in]  _k    Gray-Scott parameter k
     */
    void set_kinetic_variables(double _f, double _k) {
        this->f = _f;
        this->k = _k;
    }

    /**
     * @brief      Sets the kinetic variables.
     *
     * @param[in]  _f    Gray-Scott parameter f
     * @param[in]  _k    Gray-Scott parameter k
     */
    void set_diffusion_parameters(double _Da, double _Db) {
        this->Da = _Da;
        this->Db = _Db;
    }

    /**
     * @brief      Set zeroflux boundary conditions
     *
     * @param[in]  _zeroflux  The zeroflux boundary condition
     */
    void set_zeroflux(bool _zeroflux) {
        this->zeroflux = _zeroflux;
    }

private:
    /**
     * @brief      Build random input
     *
     * @param      a     Concentration of a
     * @param      b     Concentration of b
     */
    void build_input(float* a, float* b);

    /**
     * @brief      Initialize all variables
     */
    void initialize_variables();

    /**
     * @brief      Clean-up all variables
     */
    void cleanup_variables();

    /**
     * @brief      Write 3D concentration profile as binary file
     *
     * @param[in]  filename  The filename
     * @param[in]  vals      Concentration data
     */
    void write_binary(const std::string filename, const float *vals);

    static float uniform_dist() {
        static std::mt19937 rng;
        // center at zero and scale is 0.05
        static std::uniform_real_distribution<> nd(0.0, 1.0);

        return nd(rng);
    }
};


#endif // _RD3D_H
