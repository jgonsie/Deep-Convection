/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2016 OpenFOAM Foundation
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

Group
    grpIncompressibleSolvers

Description
    Steady-state solver for incompressible flows with turbulence modelling.

    \heading Solver details
    The solver uses the SIMPLE algorithm to solve the continuity equation:

        \f[
            \div \vec{U} = 0
        \f]

    and momentum equation:

        \f[
            \div \left( \vec{U} \vec{U} \right) - \div \gvec{R}
          = - \grad p + \vec{S}_U
        \f]

    Where:
    \vartable
        \vec{U} | Velocity
        p       | Pressure
        \vec{R} | Stress tensor
        \vec{S}_U | Momentum source
    \endvartable

    \heading Required fields
    \plaintable
        U       | Velocity [m/s]
        p       | Kinematic pressure, p/rho [m2/s2]
        \<turbulence fields\> | As required by user selection
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"

#include "revolve.h"
#include "CheckInterface.H"
#include "CheckDict.H"
#include "CheckController.H"
#include "costFunctionLibrary.C"

#include <ceres.h>

template<typename FUN>
class piggyOpt : public ceres::FirstOrderFunction
{
private:
    Foam::label n;
    FUN& fun;
public:
    piggyOpt(Foam::label n, FUN& fun) : n(n), fun(fun) {}
    virtual bool Evaluate(
        const double* parameters,
        double* cost,
        double* gradient
    ) const
    {
      if (gradient != NULL) {
          fun(parameters, true, cost[0], gradient);
      }else{
          fun(parameters, false, cost[0], NULL);
      }
      return true;
    }

    virtual int NumParameters() const { return n; }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "postProcess.H"

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "createFvOptions.H"
    #include "initContinuityErrs.H"

    #include "adjointSettings.H"

    label NN = alpha.size();
    Foam::reduce(NN,sumOp<label>());

    bool frozenTurbulence = 
        mesh.solutionDict()
        .subDict("SIMPLE")
        .lookupOrDefault<bool>("frozenTurbulence",false);

    turbulence->validate();

    AD::createGlobalTape();
    AD::switchTapeToActive();

    CheckDict checkDict(runTime);
    CheckDatabase checkDB(runTime, checkDict);

    //piggyState.runSteps(true);
    //std::vector<double> parameters(alpha.size(), 0.0);
    double* parameters = new double[alpha.size()];
    for(int i=0; i<alpha.size(); i++){
        parameters[i] = 0.0;
    }

    label iter = 0;

    auto runSteps = [&](const double* params, bool calcGrad, double& cost, double* grad){
        bool firstStep = true;
        scalar piggyEps = 0.1; //1e-3;
        scalar costEps = 1e-3;
        scalar dSensSum = std::numeric_limits<double>::max();
        scalar dCost    = std::numeric_limits<double>::max();
        scalar oldCost = 0;
        scalar oldSensSum = 0;

        //label i = 0;

        // build local copys
        AD::resetTape();
        
        for(int i=0; i<alpha.size(); i++){
            alpha[i] = params[i];
        }

        if(calcGrad){
            AD::registerInputVariable(alpha.begin(), alpha.end());
        }else{
            AD::switchTapeToPassive();
        }

        auto reset_to = AD::getTapePosition();
        // run reverse accumulation until adjoint residual is small
        while ((dSensSum > piggyEps || dCost > costEps) && simple.loop())
        {
            if(calcGrad){
                checkDB.registerAdjoints(); // store tape indices
            }

            Info<< "Time = " << runTime.timeName() << nl << endl;

            // --- Pressure-velocity SIMPLE corrector
            {
                #include "UEqn.H"
                #include "pEqn.H"
            }

            laminarTransport.correct();
            
            bool wasActive = AD::isTapeActive();
            if(frozenTurbulence && wasActive)
                AD::switchTapeToActive();

            turbulence->correct();

            if(frozenTurbulence && wasActive)
                AD::switchTapeToActive();

            Info<< "ExecutionTime = " << runTime.elapsedCpuTime()   << " s"
                << "  ClockTime = "   << runTime.elapsedClockTime() << " s"
                << nl << endl;

            scalar J = CostFunction(mesh).eval();

            dCost = abs((J - oldCost)/J);
            oldCost = J;

            if(calcGrad){
                if(Pstream::master()){
                    AD::derivative(J)=1.0;
                }
                if(!firstStep)
                    checkDB.restoreAdjoints();
                else
                    firstStep = false;

                AD::interpretTape();
                checkDB.storeAdjoints();
                scalar sensSum = 0;
                forAll(alpha,i){
                    sens[i] = AD::derivative(alpha[i])/mesh.V()[i];
                    sensSum += std::abs(AD::passiveValue(sens[i]));
                }

                Foam::reduce(sensSum,sumOp<scalar>());
                dSensSum = abs(sensSum - oldSensSum)/NN;
                oldSensSum = sensSum;
                AD::resetTapeTo(reset_to);
                AD::zeroAdjointVector();
            }
            Info << "Piggy: " << iter << " " << J << " " << dCost << " " << dSensSum << endl;
            forAll(alpha,i){
                grad[i] = AD::passiveValue(sens[i]);
            }
        }
        runTime.write();
        iter++;
    };

    piggyOpt pOpt(alpha.size(), runSteps);
    ceres::GradientProblem problem(&pOpt);

    ceres::GradientProblemSolver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(options, problem, parameters, &summary);

    Info << summary.FullReport() << "\n";

    Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = " << runTime.elapsedClockTime() << " s"
        << nl << endl;

    return 0;
}


// ************************************************************************* //
