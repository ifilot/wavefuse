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

#ifndef _REACTION_BRUSSELATOR
#define _REACTION_BRUSSELATOR

/**
 * @brief      Calculate Brusselator reaction rate
 *
 * @param[in]  fx    pointer to concentration of compound A
 * @param[in]  fy    pointer to concentration of compound B
 * @param      drx   pointer to reaction rate of compound A
 * @param      dry   pointer to reaction rate of compound B
 */
__global__ void reaction_brusselator(const float *fx, const float *fy, float *drx, float *dry) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < d_ncells; i += stride) {
        float r = fx[i] * fx[i] * fy[i];
        drx[i] = d_c1 - (d_c2 + 1.0) * fx[i] + r;
        dry[i] = d_c2 * fx[i] - r;
    }
}

#endif // _REACTION_BRUSSELATOR
