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
    AD::switchTapeToPassive();
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
    auto pos1 = AD::getTapePosition();
    AD::switchTapeToActive();
    checkData.registerAdjoints();
    auto pos2 = AD::getTapePosition();
    checkControl.runStep();
    checkData.registerAsOutput();

    scalar J = checkControl.calcCost();
    AD::derivative(J) = 1.0;
    AD::switchTapeToPassive();

    label nReverseAccumulations = checkpointSettings.lookupOrDefault<label>("nReverseAccumulations",100); 

    for(int i=0; i<nReverseAccumulations; i++){
        if(i>0){
            checkData.restoreAdjoints();
        }
        AD::interpretTapeTo(pos2);
        checkData.storeAdjoints();

        //ADmode::global_tape->zeroAdjointVectorFromTo(pos2,pos1);
        AD::zeroAdjointVectorTo(pos1);

        double s = checkData.calcNormOfStoredAdjoints();
        Info << "Reverse acc: " << fabs(s) << endl;
        checkControl.postInterpret();
    }
    
    checkControl.postInterpret();
}
#endif
