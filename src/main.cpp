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

#include <iostream>
#include <chrono>

#include "config.h"
#include "rd3d.h"
#include "card_manager.h"

int main() {
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Executing "<< PROGRAM_NAME << " v." << PROGRAM_VERSION << std::endl;
    std::cout << "Author: Ivo Filot <i.a.w.filot@tue.nl>" << std::endl;
    std::cout << "Website: https://gitlab.tue.nl/ifilot/ftcs-cuda" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;

    auto start = std::chrono::system_clock::now();

    CardManager cm;
    cm.probe_cards();

    RD3D rd3d;
    rd3d.run_cuda();

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "Done execution in " << elapsed_seconds.count() << " seconds." << std::endl << std::endl;

    return 0;
}
