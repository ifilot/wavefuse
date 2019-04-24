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

#ifndef _INPUTREADER_H
#define _INPUTREADER_H

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "rd3d.h"

class InputReader {
private:
    const std::vector<std::string> double_vars = {"Da", "Db", "dt", "dx", "f", "k"};
    const std::vector<std::string> uint_vars = {"timesteps", "tsteps", "mx", "my", "mz", "pencils"};

    std::unordered_map<std::string, double> double_values;
    std::unordered_map<std::string, unsigned int> uint_values;

public:
    /**
     * @brief      Constructs the object.
     */
    InputReader();

    /**
     * @brief      Builds an integrator.
     *
     * @param[in]  inputfile  The inputfile
     *
     * @return     The integrator.
     */
    RD3D build_integrator(const std::string& inputfile);

private:
    /**
     * @brief      Get unsigned int value for variable
     *
     * @param[in]  name  Variable name
     *
     * @return     value
     */
    unsigned int get_uint(const std::string& name);

    /**
     * @brief      Get double value for variable
     *
     * @param[in]  name  Variable name
     *
     * @return     value
     */
    unsigned int get_double(const std::string& name);
};

#endif // _INPUTREADER_H
