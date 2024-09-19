/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2015 OpenFOAM Foundation
     \\/     M anipulation  |
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
    adjointMoveMesh

Group
    grpMeshManipulationUtilities

Description
    Solver for moving meshes.

\*---------------------------------------------------------------------------*/
#include "fvCFD.H"                          // lm

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "motionSolver.H"
#include "displacementMotionSolver.H"       // lm, overwrites original constructor
#include "pointMesh.H"                      // lm
#include "pointPatchField.H"                // lm
#include "pointFields.H"                    // lm
#include "PrimitivePatchInterpolation.H"    // lm

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    timeSelector::addOptions();
    #include "addDictOption.H"

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    instantList timeDirs = timeSelector::select0(runTime, args);

    // #include "createFields.H" // lm

    autoPtr<motionSolver> motionPtr = motionSolver::New(mesh);

    // get PointDisplacement field that will be used by the mesh movement
    pointVectorField& PointDisplacement = const_cast<pointVectorField&>
    (
        mesh.objectRegistry::lookupObject<pointVectorField>
        (
            "pointDisplacement"
        )
    );

    // lm: additional code for sensitivity mesh movement
    const Foam::wordList sensOutputPatches(
        mesh.solutionDict().subDict("SIMPLE").lookup("sensOutputPatches")
    );   

    // lm: get parameters
    const scalar lambda(motionPtr->lookupOrDefault<scalar>("lambda",1e-3));
    Info << "Selecting morphing scale factor: lambda = " << lambda << endl;
    const bool atanfilter(motionPtr->lookupOrDefault<bool>("atanfilter",true));
    Info << "Selecting atan filtering: atanfilter = " << atanfilter << endl;

    // store starting time
    scalar tStart = runTime.value();

    forAll(timeDirs, timeI)
    {
        runTime.setTime(timeDirs[timeI], timeI);

        surfaceScalarField surfSens
        (
            IOobject
            (
                "surfSens",
                runTime.timeName(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar("0", dimless, 0),
            "calculated"
        );


        forAll(sensOutputPatches,cI)
        {
            Info << "### Evaluating sensOutputPatch = " << sensOutputPatches[cI] << endl;
            Foam::label bi = mesh.boundaryMesh().findPatchID(sensOutputPatches[cI]);

            // calculate patch area for area-weighted average
            scalar patchArea = gSum(mesh.magSf().boundaryField()[bi])+SMALL;

            // debug output
            scalar sensMax = max(surfSens.boundaryField()[bi]);
            scalar sensMin = min(surfSens.boundaryField()[bi]);
            scalar sensMean = gSum(mesh.magSf().boundaryField()[bi]*mag(surfSens.boundaryField()[bi]))/patchArea;
            Info << "sensSurfaceVec (input):    min=" << sensMin << ", max=" << sensMax << ", avg=" << sensMean << endl;

            // bound sens with atan (c.f. Key2014)
            if(atanfilter) {
                forAll(mesh.boundary()[bi],i){
                    scalar sens = surfSens.boundaryField()[bi][i];
                    surfSens.boundaryFieldRef()[bi][i] = atan(sens/sensMean);
                }
                Info << "Filtered with atan filtering" << endl;
            }
            else
            {
                Info << "No filtering" << endl;
            }

            sensMax = max(surfSens.boundaryField()[bi]);
            sensMin = min(surfSens.boundaryField()[bi]);
            sensMean = gSum(mesh.magSf().boundaryField()[bi]*mag(surfSens.boundaryField()[bi]))/patchArea;
            Info << "sensSurfaceVec (filtered): min=" << sensMin << ", max=" << sensMax << ", avg=" << sensMean << endl;

            //- set-up interpolator
            PrimitivePatchInterpolation<primitivePatch> patchInterpolator(mesh.boundaryMesh()[bi]);

            //- Perform interpolation, and scale with lambda
            // scalar dx = 1e-3;
            vectorField interpSensPointVec = patchInterpolator.faceToPointInterpolate(
                lambda*mesh.boundary()[bi].nf()()*surfSens.boundaryField()[bi]
            );

            // lm: double == is necessary!
            PointDisplacement.boundaryFieldRef()[bi] == -interpSensPointVec;

            Info << "faceToPointInterpolate successful" << endl;
        }

        Info << "Time = " << runTime.timeName() << endl;

        mesh.movePoints(motionPtr->newPoints());
        mesh.write();

        Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
             << "  ClockTime = " << runTime.elapsedClockTime() << " s"
             << nl << endl;

        // lm: run only for one time step
        // break;
    }

    // write into starting time
    // Info<< "Writing morphed mesh into initial time = " << tStart << endl;
    // runTime.setTime(tStart,label(AD::passiveValue(tStart)));

    // runTime.write();        
    // mesh.write();

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
