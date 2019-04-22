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
    unsigned int dim = 256;
    unsigned int mx = dim;
    unsigned int my = dim;
    unsigned int mz = dim;
    unsigned int ncells = dim * dim * dim;

    unsigned int pencils = 4;

    float *a;   //!< concentration of component A
    float *b;   //!< concentration of component B

    // device variables
    float *d_a, *d_b, *d_dx2, *d_dy2, *d_dz2, *d_ra, *d_rb, *d_da, *d_db;

    // reaction settings of kinetic system
    const float f = 0.0416;     //!< reactivity constant f
    const float k = 0.0625;     //!< reactivity constant f
    const float Da = 0.16;      //!< diffusion constant of A
    const float Db = 0.08;      //!< diffusion constant of B
    const float dt = 0.25;      //!< temporal discretization
    const float dx = 0.5;       //!< spatial discretization

    unsigned int timesteps = 360;
    unsigned int tsteps = 100;

    std::string donestring = "           [DONE]";

public:
    /**
     * @brief      Constructs the object.
     */
    RD3D();

    /**
     * @brief      Run time-integration on GPU
     */
    void run_cuda();

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
