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
#include <tclap/CmdLine.h>

#include "config.h"
#include "rd3d.h"
#include "card_manager.h"
#include "inputreader.h"

int main(int argc, char* argv[]) {
    try {
        TCLAP::CmdLine cmd("Performs integration on 3D reaction-diffusion systems.", ' ', PROGRAM_VERSION);

        //**************************************
        // declare values to be parsed
        //**************************************

        // input filename
        TCLAP::ValueArg<std::string> arg_input_filename("i","input","Input file",true,"input","filename");
        cmd.add(arg_input_filename);

        cmd.parse(argc, argv);

        std::cout << "--------------------------------------------------------------" << std::endl;
        std::cout << "Executing "<< PROGRAM_NAME << " v." << PROGRAM_VERSION << std::endl;
        std::cout << "Author: Ivo Filot <i.a.w.filot@tue.nl>" << std::endl;
        std::cout << "Website: https://github.com/ifilot/wavefuse" << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;

        auto start = std::chrono::system_clock::now();

        CardManager cm;
        cm.probe_cards();

        InputReader ir;
        RD3D rd3d = ir.build_integrator(arg_input_filename.getValue());
        rd3d.run_cuda();

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        std::cout << "----------------------------------------------------------" << std::endl;
        std::cout << "Done execution in " << elapsed_seconds.count() << " seconds." << std::endl << std::endl;

        return 0;

    }  catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() <<
                     " for arg " << e.argId() << std::endl;
        return -1;
    }
}
