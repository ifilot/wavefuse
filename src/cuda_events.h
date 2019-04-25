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

#ifndef _CUDA_EVENTS_H
#define _CUDA_EVENTS_H

#include "check_cuda.h"

/**
 * @brief      Starts an event.
 *
 * @param      event  The event
 */
void start_event(cudaEvent_t* event) {
    checkCuda( cudaEventRecord(*event, 0) );
}

/**
 * @brief      Stops an event.
 *
 * @param      event  The event
 *
 * @return     Return number of milliseconds
 */
float stop_event(cudaEvent_t* start_event, cudaEvent_t* stop_event) {
    checkCuda( cudaEventRecord(*stop_event, 0) );
    checkCuda( cudaEventSynchronize(*stop_event) );
    float milliseconds = 0;
    checkCuda( cudaEventElapsedTime(&milliseconds, *start_event, *stop_event) );
    return milliseconds;
}

#endif // _CUDA_EVENTS_H
