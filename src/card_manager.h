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

#ifndef _CARD_MANAGER_H
#define _CARD_MANAGER_H

#include <iostream>
#include <vector>

class CardManager {
private:
    std::vector<std::string> gpu_names;

public:

    /**
     * @brief      Constructs the object.
     */
    CardManager();

    /**
     * @brief      Gets the gpu names.
     *
     * @return     The gpu names.
     */
    const auto& get_gpu_names() const {
        return this->gpu_names;
    }

    /**
     * @brief      Probe the available GPU units
     */
    void probe_cards();
private:
    /**
     * @brief      Get the total number of cores on the GPU
     *
     * @param      devProp  Card properties
     *
     * @return     The number cores.
     */
    int get_number_cores(void* devProp);
};

#endif // _CARD_MANAGER_H
