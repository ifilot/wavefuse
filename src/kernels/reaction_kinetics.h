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

#ifndef _REACTION_KINETICS_H
#define _REACTION_KINETICS_H

// constant variables on device
__device__ __constant__ float d_c1;
__device__ __constant__ float d_c2;
__device__ __constant__ float d_c3;
__device__ __constant__ float d_c4;

#include "reaction_brusselator.h"
#include "reaction_barkley.h"
#include "reaction_gray_scott.h"
#include "reaction_fitzhugh_nagumo.h"
#include "reaction_lotka_volterra.h"

#endif // _REACTION_KINETICS_H
