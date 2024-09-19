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
	//DEFINE ARGUMENTS
	//argList::addArgument("argFields");
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

    //My turbulence methods
    turbulence->kRAM(k_dict);
    turbulence->OmegaRAM(Omega_dict);
    turbulence->nutRAM(nut_dict);
    turbulence->validate();

    runTime.setTime(startTime,0);
    runTime.setEndTime(endTime);

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readDyMControls.H"
        #include "CourantNo.H"
        #include "setDeltaT.H"

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;
		
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

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    // Get the shared memory identifier from the command line argument
    int SM_output_id = atoi(argv[2]);

    // Attach the shared memory segment to a pointer for the return data
    double* output_dict = static_cast<double*>(shmat(SM_output_id, NULL, 0));
    if (output_dict == (double *)-1) {
        perror("shmat");
        exit(1);
    }

    int ncells = mesh.nCells();
	int nfaces = phi.size();

    volScalarField nut (turbulence->nut());
	volScalarField k (turbulence->k()); //convert tmp volScalarField to regular
	volScalarField omega (turbulence->omega());
	
	// Loop over internalField of all fields
    for (int i = 0; i < ncells; i++){
        output_dict[i] = static_cast <double>(AD::value(k[i]));
		output_dict[i+ncells] = static_cast <double>(AD::value(nut[i]));
		output_dict[i+2*ncells] = static_cast <double>(AD::value(omega[i]));
		output_dict[i+3*ncells] = static_cast <double>(AD::value(p[i]));
		output_dict[i+4*ncells] = static_cast <double>(AD::value(U[i][0]));
		output_dict[i+5*ncells] = static_cast <double>(AD::value(U[i][1]));
    }
	
	int iterator = 6*ncells;
	for (int i = 0; i < nfaces; i++){
        output_dict[i+iterator] = static_cast <double>(AD::value(phi[i]));
	}
	iterator += nfaces;
	// Access the boundary information
    const auto& boundaryMesh = mesh.boundaryMesh();

    // Loop over all patch IDs
    forAll(boundaryMesh, patchi)
    {
        const auto& patch = boundaryMesh[patchi];
		if (patch.name() != "defaultFaces"){
			
			//Foam::label patchID = mesh.boundaryMesh().findPatchID(patch.name());
			int nfaces_patch = patch.size();
			//Info << "Processing boundary patch " << patch.name() <<" with ID "<< patchID << " with number of faces=" << patch.size() <<endl;
			for(int i = 0; i < nfaces_patch; i++){
				output_dict[i+iterator] = static_cast <double> (AD::value(k.boundaryField()[patchi][i]));
				output_dict[i+iterator+nfaces_patch] = static_cast <double>(AD::value(nut.boundaryField()[patchi][i]));
				output_dict[i+iterator+2*nfaces_patch] = static_cast <double>(AD::value(omega.boundaryField()[patchi][i]));
				output_dict[i+iterator+3*nfaces_patch] = static_cast <double>(AD::value(p.boundaryField()[patchi][i]));
				output_dict[i+iterator+4*nfaces_patch] = static_cast <double>(AD::value(U.boundaryField()[patchi][i][0]));
				output_dict[i+iterator+5*nfaces_patch] = static_cast <double>(AD::value(U.boundaryField()[patchi][i][1]));
				output_dict[i+iterator+6*nfaces_patch] = static_cast <double>(AD::value(phi.boundaryField()[patchi][i]));
				//Info << "Faces iterator: "<< i << " Face id: " << patch[i] << endl;
			}
			iterator += 7*nfaces_patch;
		}
    }
    shmdt(input_dict);
    shmdt(output_dict);

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
