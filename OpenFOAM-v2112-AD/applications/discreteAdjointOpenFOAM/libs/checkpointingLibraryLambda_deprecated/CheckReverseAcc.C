#if defined(DAOF_AD_MODE_A1S)
#include <stack>

#include "CheckController.H"
#include "CheckDatabase.H"
#include "CheckReverseAcc.H"

CheckReverseAcc::CheckReverseAcc(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& runTime)
    :
    CheckMethod(cd, checkDict, runTime),
    firstRun(true),
    lastInterpretation(false),
    nDoneTapeSteps(0),
    revolve(nCheckpoints), /* online checkpointing */
    checkpointSettings(checkDict.checkpointDict())
{}

void CheckReverseAcc::run(CheckController& checkControl)
{
    ADmode::global_tape->switch_to_passive();
    int i = 0;
    do
    {
        if(i%2 == 0){
            checkData.replaceCheckpoint(0);
        }else{
            checkData.replaceCheckpoint(1);
        }
        i++;
        bool endReached = checkControl.runStep();
        if(endReached){
            break;
        }
    } while(true);

    Info << "Running one active step" << endl;
    checkData.restoreCheckpoint(i%2);
    auto pos1 = ADmode::global_tape->get_position();
    ADmode::global_tape->switch_to_active();
    checkData.registerAdjoints();
    auto pos2 = ADmode::global_tape->get_position();
    checkControl.runStep();
    checkData.registerAsOutput();

    scalar J = checkControl.calcCost();
    AD::derivative(J) = 1.0;
    ADmode::global_tape->switch_to_passive();

    label nReverseAccumulations = checkpointSettings.lookupOrDefault<label>("nReverseAccumulations",100); 
    //runTime.mesh().solutionDict().subDict("SIMPLE").lookupOrDefault<label>("nReverseAccumulations",100);

    for(int i=0; i<nReverseAccumulations; i++){
        if(i>0){
            checkData.restoreAdjoints();
        }
        ADmode::global_tape->interpretTapeTo(pos2);
        checkData.storeAdjoints();

        double a = 0;
        for(unsigned int i=1; i <= pos1.index(); i++){
            a += ADmode::global_tape->_adjoint(i);
        }
        double s = 0;
        for(unsigned int i=pos1.index()+1; i <= pos2.index(); i++){
            s += ADmode::global_tape->_adjoint(i);
        }

        //ADmode::global_tape->zeroAdjointVectorFromTo(pos2,pos1);
        ADmode::global_tape->zeroAdjointVectorTo(pos1);

        //double s = checkData.calcNormOfStoredAdjoints();
        Info << "Reverse acc: " << i << " adjoints: " << a << " adjoint residual: " << fabs(s) << endl;
    }
}
#endif
