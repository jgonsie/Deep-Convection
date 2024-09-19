/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    pimpleFoam.C

Group
    grpIncompressibleSolvers

Description
    Transient solver for incompressible, turbulent flow of Newtonian fluids
    on a moving mesh.

    \heading Solver details
    The solver uses the PIMPLE (merged PISO-SIMPLE) algorithm to solve the
    continuity equation:

        \f[
            \div \vec{U} = 0
        \f]

    and momentum equation:

        \f[
            \ddt{\vec{U}} + \div \left( \vec{U} \vec{U} \right) - \div \gvec{R}
          = - \grad p + \vec{S}_U
        \f]

    Where:
    \vartable
        \vec{U} | Velocity
        p       | Pressure
        \vec{R} | Stress tensor
        \vec{S}_U | Momentum source
    \endvartable

    Sub-models include:
    - turbulence modelling, i.e. laminar, RAS or LES
    - run-time selectable MRF and finite volume options, e.g. explicit porosity

    \heading Required fields
    \plaintable
        U       | Velocity [m/s]
        p       | Kinematic pressure, p/rho [m2/s2]
        \<turbulence fields\> | As required by user selection
    \endplaintable

Note
   The motion frequency of this solver can be influenced by the presence
   of "updateControl" and "updateInterval" in the dynamicMeshDict.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include <sys/shm.h>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        " MY Transient solver for incompressible, turbulent flow"
        " of Newtonian fluids on a moving mesh."
    );
	// JGS: Arguments definition (not relevant)
	argList::validArgs.append("SM_pointer_input");
    argList::validArgs.append("SM_pointer_output");

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "initContinuityErrs.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createUfIfPresent.H"
    #include "CourantNo.H"
    #include "setInitialDeltaT.H"
	
    //JGS: My turbulence methods (not relevant)
    turbulence->nutRAM(nut_dict);
    turbulence->validate();

    runTime.setTime(startTime,0);
    runTime.setEndTime(endTime);

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    // MT: create the AD tape, could be in some global function
    label tapeSizeMB = mesh.solutionDict().subDict("PIMPLE").lookupOrDefault<label>("tapeSizeMB",1096);
    Info << "Creating Tape, size: " << tapeSizeMB << endl;
    AD::createGlobalTape(tapeSizeMB);

    //label nCells = U.size();
    label nFaces = Ufaces.size();

	// MT: Note: this will only work once. If we need to backprop multiple steps within the same binary we need some more boilerplate code

    // MT: single forward step, taped with AD, wrapped into lambda
    auto stepForward = [&](){
        #include "readDyMControls.H"
        #include "CourantNo.H"
        #include "setDeltaT.H"

        ++runTime;

        // MT: register inputs w.r.t. we want to diff.
        for(int i=0; i<nFaces; i++){
            AD::registerInputVariable(Ufaces[i][0]);
            AD::registerInputVariable(Ufaces[i][1]);
        }
		
		// MT: record time step
        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                // Do any mesh changes
                mesh.controlledUpdate();

                if (mesh.changing())
                {
                    MRF.update();

                    if (correctPhi)
                    {
                        // Calculate absolute flux
                        // from the mapped surface velocity
                        phi = mesh.Sf() & Uf();

                        #include "correctPhi.H"

                        // Make the flux relative to the mesh motion
                        fvc::makeRelative(phi, U);
                    }

                    if (checkMeshCourantNo)
                    {
                        #include "meshCourantNo.H"
                    }
                }
            }

            #include "UEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                laminarTransport.correct();
                turbulence->correct();
            }
        }
    };

    // MT: will use the captured variables + the global tape to perform Jacobian Vector Product during backprop
    auto stepReverse = [&](const std::vector<double> &U_bar) -> std::vector<double>{
        std::vector<double> jacVecProd(2*nFaces);
        // note that we hardcode dimensions X-Y here, no Z
        // "seed" outputs with vector dUx and dUy
        for(int i=0; i<nCells; i++){
            AD::derivative(U[i][0]) = U_bar[2*i+0];
            AD::derivative(U[i][1]) = U_bar[2*i+1];
        }
        AD::interpretTape(); // implicitly calculates product UF_bar = J^T*U_bar
        
        // MT: "harvest" Jacobian Vector Product from inputs
        for(int i=0; i<nFaces; i++){
            jacVecProd[i] = AD::derivative(Ufaces[i][0]);
			jacVecProd[i+nFaces] = AD::derivative(Ufaces[i][1]);
        }
        AD::resetTape(); // reset does not free memory
        return jacVecProd;
    };

    // MT: Run the lambdas
    stepForward();
	Info<< "I have finished forward call\n" << endl;

    // MT: call the backprop function "seed" with U_bar
    //std::vector<double> U_bar(2*nCells, 1.0); // some arbitrary vector here, this would come in from tensorFlow backprop
    std::vector<double> jacVecProd = stepReverse(U_bar);
	Info<< "I have finished backward call\n" << endl;

	// MT: destroy and free memory of tape
    //AD::removeGlobalTape(); //not working
	ADmode::tape_t::remove(ADmode::global_tape);
	
	// JGS: Here to the end I carry out some modifications for communication between softwares
    // Get the shared memory identifier from the command line argument
    int SM_output_id = atoi(argv[2]);

    // Attach the shared memory segment to a pointer for the return data
    double* output_dict = static_cast<double*>(shmat(SM_output_id, NULL, 0));
    if (output_dict == (double *)-1) {
        perror("shmat");
        exit(1);
    }
	
	for (int i = 0; i < nFaces; i++){
        output_dict[i] = jacVecProd[i];
		output_dict[i+nFaces] = jacVecProd[i+nFaces];
		Info<< "( " <<output_dict[i]<<", "<<output_dict[i+nFaces]<<")\n"<< endl;
    }
	
	Info<< "I have finished sending back data\n" << endl;
	
	shmdt(input_dict);
    shmdt(output_dict);
	
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
