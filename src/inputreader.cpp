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

#include "inputreader.h"

InputReader::InputReader() {

}

/**
 * @brief      Builds an integrator.
 *
 * @param[in]  inputfile  The inputfile
 *
 * @return     The integrator.
 */
RD3D InputReader::build_integrator(const std::string& inputfile) {
    std::ifstream file(inputfile);

    if(file.is_open()) {
        std::string line;

        static const boost::regex regex_comment("^#.*");
        static const boost::regex regex_empty_line("^\\s*");
        static const boost::regex regex_variable("^\\s*([A-Za-z0-9]+)\\s+=\\s+([A-Za-z0-9._-]+)\\s*$");


        while(std::getline(file, line)) {
            boost::smatch what;
            if(boost::regex_match(line, what, regex_comment)) {
                continue;
            }

            if(boost::regex_match(line, what, regex_empty_line)) {
                continue;
            }

            if(boost::regex_match(line, what, regex_variable)) {
                std::vector<std::string> pieces;
                boost::split(pieces, line, boost::is_any_of("="));

                // trim pieces
                boost::trim(pieces[0]);
                boost::trim(pieces[1]);

                // store doubles
                if(std::find(this->double_vars.begin(), this->double_vars.end(), pieces[0]) != this->double_vars.end()) {
                    this->double_values.emplace(pieces[0], boost::lexical_cast<double>(pieces[1]));
                    continue;
                }

                // store strings
                if(std::find(this->string_vars.begin(), this->string_vars.end(), pieces[0]) != this->string_vars.end()) {
                    this->string_values.emplace(pieces[0], pieces[1]);
                    continue;
                }

                // store unsigned integers
                if(std::find(this->uint_vars.begin(), this->uint_vars.end(), pieces[0]) != this->uint_vars.end()) {
                    this->uint_values.emplace(pieces[0], boost::lexical_cast<unsigned int>(pieces[1]));
                    continue;
                }

                // store boolean integers
                if(std::find(this->bool_vars.begin(), this->bool_vars.end(), pieces[0]) != this->bool_vars.end()) {
                    if(pieces[1] == "true") {
                        this->bool_values.emplace(pieces[0], true);
                    } else if(pieces[1] == "false") {
                        this->bool_values.emplace(pieces[0], false);
                    } else {
                        throw std::runtime_error("Invalid boolean value encountered: " + pieces[1]);
                    }
                    continue;
                }
            }
        }

        RD3D rd3d;
        rd3d.set_reaction_type(this->get_string("type"));
        rd3d.set_dimensions(this->get_uint("mx"), this->get_uint("my"), this->get_uint("mz"));
        rd3d.set_integration_variables(this->get_double("dt"), this->get_double("dx"), this->get_uint("timesteps"), this->get_uint("tsteps"));
        rd3d.set_kinetic_variables(this->get_double("c1"), this->get_double("c2"));
        rd3d.set_diffusion_parameters(this->get_double("Da"), this->get_double("Db"));
        rd3d.set_zeroflux(this->get_bool("zeroflux"));

        return rd3d;

    } else {
        throw std::runtime_error("Cannot open file: " + inputfile);
    }
}

/**
 * @brief      Get unsigned int value for variable
 *
 * @param[in]  name  Variable name
 *
 * @return     value
 */
unsigned int InputReader::get_uint(const std::string& name) {
    auto got = this->uint_values.find(name);
    if(got != this->uint_values.end()) {
        return got->second;
    } else {
        throw std::runtime_error("Cannot find variable " + name);
    }
}

/**
 * @brief      Get double value for variable
 *
 * @param[in]  name  Variable name
 *
 * @return     value
 */
double InputReader::get_double(const std::string& name) {
    auto got = this->double_values.find(name);
    if(got != this->double_values.end()) {
        return got->second;
    } else {
        throw std::runtime_error("Cannot find variable " + name);
    }
}

/**
 * @brief      Get bool value for variable
 *
 * @param[in]  name  Variable name
 *
 * @return     value
 */
bool InputReader::get_bool(const std::string& name) {
    auto got = this->bool_values.find(name);
    if(got != this->bool_values.end()) {
        return got->second;
    } else {
        throw std::runtime_error("Cannot find variable " + name);
    }
}

/**
 * @brief      Get bool value for variable
 *
 * @param[in]  name  Variable name
 *
 * @return     value
 */
const std::string& InputReader::get_string(const std::string& name) {
    auto got = this->string_values.find(name);
    if(got != this->string_values.end()) {
        return got->second;
    } else {
        throw std::runtime_error("Cannot find variable " + name);
    }
}
