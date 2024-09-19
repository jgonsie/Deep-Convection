/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2013 OpenFOAM Foundation
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
    simpleFoam

Description
    Steady-state solver for incompressible, turbulent flow

\*---------------------------------------------------------------------------*/

#include "profiling.H"

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"

#include "CheckInterface.H"
#include "CheckDict.H"
#include "cellSet.H"

#include "costFunctionLibrary.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scalar penalty(scalar x, scalar width, scalar height){
    const scalar pi = Foam::constant::mathematical::pi;
    return (x>scalar(0) && x<width) ?
        scalar(height*0.5*(sin((2*pi/width*x-0.5*pi))+1)) : scalar(0) ;
}

int main(int argc, char *argv[])
{
    #include "postProcess.H"

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "createMRF.H"
    #include "createFvOptions.H"
    #include "initContinuityErrs.H"

    #include "adjointSettings.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    CheckInterface check(runTime);
    CheckDict checkDict(runTime);
    CheckDatabase checkDB(runTime,checkDict);

    point searchPoint = simpleDict.lookupOrDefault<point>("derivPoint",point(0,0,0));
    label derivCellID = mesh.findCell(searchPoint);
    if(derivCellID >= 0){
        Pout << "nearest cell " << derivCellID << endl;
    }

    Info<< "\nStarting time loop\n" << endl;

    scalar oldSensSum = 0;

    ADmode::global_tape = ADmode::tape_t::create();
    bool firstStep = true;

    turbulence->validate();

    //ADmode::tape_t::position_t interpret_to = ADmode::global_tape->get_position();

    label NN = alpha.size();
    Foam::reduce(NN,sumOp<label>());

    scalar alphaMax = 1000;
    label nOptSteps = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<label>("nOptSteps",100);
    scalar penaltyScale = 1e-4;
    for(int optStep = 0; optStep < nOptSteps; optStep++){
        ADmode::global_tape->reset();
        ADmode::global_tape->switch_to_active();

        forAll(alpha,i){
            ADmode::global_tape->register_variable(alpha[i]);
        }
        ADmode::tape_t::position_t reset_to = ADmode::global_tape->get_position();
        scalar dSensSum = std::numeric_limits<double>::max();

        while (/*dSensSum > optEpsilon &&*/ simple.loop())
        {
            checkDB.registerAdjoints(); // store tape indices

            Info<< "Time = " << runTime.timeName() << nl << endl;

            // --- Pressure-velocity SIMPLE corrector
            {
                #include "UEqn.H"
                #include "pEqn.H"
            }

            laminarTransport.correct();

            // turbulence correct
            bool frozenTurbulence = 
                mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<bool>("frozenTurbulence",false);
            if(frozenTurbulence){
                Info << "\nWARNING: Calculating Adjoints with frozen Turbulence assumption\n" << endl;
                ADmode::global_tape->switch_to_passive();
            }

            turbulence->correct();

            if(frozenTurbulence){
                ADmode::global_tape->switch_to_active();
            }

            scalar J = CostFunction(mesh).eval();
            scalar scaleP = J/NN;
            scalar Jp = 0;
            forAll(alpha,i){
                Jp += penalty(alpha[i],alphaMax,penaltyScale*scaleP);
            }
            Foam::reduce(Jp,sumOp<scalar>());
            Info << "J: " << J << " " << Jp << " " << J+Jp << " " << gSum(alpha) << " " << penaltyScale << endl;
            J = J + Jp;
            if(Pstream::master()){
                AD::derivative(J)=1;
            }

            if(!firstStep){
                checkDB.restoreAdjoints();
            }else{
                firstStep = false;
            }

            ADmode::global_tape->interpret_adjoint(); //_to(interpret_to);
            checkDB.storeAdjoints();
            scalar norm2 = checkDB.calcNormOfStoredAdjoints();

            scalar sensSum = 0;

            forAll(designSpaceCells,i){
                const label j = designSpaceCells[i];
                sens[j] = AD::derivative(alpha[j])/mesh.V()[j];
                sensSum += std::abs(AD::passiveValue(sens[j]));
            }

            Foam::reduce(sensSum,sumOp<scalar>());
            Foam::reduce(norm2,sumOp<scalar>());
            dSensSum = abs(sensSum - oldSensSum)/NN;
            Info << "piggy: " << optStep << " " << runTime.timeName() << " " << sensSum << " " << dSensSum << " " << norm2 << " " << J << endl;
            oldSensSum = sensSum;

            ADmode::global_tape->reset_to(reset_to);
            ADmode::global_tape->zero_adjoints();

            runTime.write();

            Info<< "ExecutionTime = " << static_cast<scalar>(runTime.elapsedCpuTime()) << " s"
                << "  ClockTime = " <<   static_cast<scalar>(runTime.elapsedClockTime()) << " s"
                << nl << endl;

            if(runTime.timeIndex() % 50 == 0)
                break;
        }
        ADmode::global_tape->switch_to_passive();
        
        scalar lbda = 0.1*pow(1.05,optStep+1);
        lbda = min(static_cast<scalar>(100.0),lbda);
        penaltyScale *= 1.2;
        penaltyScale = min(10,penaltyScale);

        forAll(alpha,j){
            //const label j = designSpaceCells[i];
            //Pout << "cell " << j << endl;
            alpha[j] -= lbda*sens[j];
            alpha[j] = max(alpha[j],static_cast<scalar>(0.0));
            alpha[j] = min(alpha[j],alphaMax);
        }
        //Pout << "Enter gAverage" << endl;
        scalar avg = gAverage(alpha);
        //Pout << "Exit gAverage" << endl;
        Info << "lbda: " << lbda << " " << avg << endl;
    }


    ADmode::tape_t::remove(ADmode::global_tape);
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
