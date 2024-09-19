//#define DAOF_AD_MODE_A1S
#if defined(DAOF_AD_MODE_A1S)
#include "CheckController.H"
#include "CheckDatabase.H"
#include "CheckEquidistant.H"

CheckEquidistant::CheckEquidistant(CheckDatabase& cd, CheckDict& checkDict,
                                   Foam::Time& runTime)
    : CheckMethod(cd, checkDict, runTime)
{}

void CheckEquidistant::run(CheckController& checkControl)
{
    AD::switchTapeToPassive();

    label checkDistance =
        max(nCalcSteps / nCheckpoints + (nCalcSteps % nCheckpoints != 0),1);

    Info << "CHECK: " << checkDistance << endl;
    label currCheck = 0;
    label nSteps = 0;

    std::pair<scalar,label> lastTimeIndex(0,0);

    // place checkpoint for initial step
    checkData.replaceCheckpoint(currCheck++);
    // run passive until completion
    while(!checkControl.runStep()){
        nSteps++;
        if ((runTime.timeIndex() - startTimeID) % checkDistance == 0)
        {
            Info << "CHECK: store Checkpoint " << currCheck << " ("
                 << runTime.timeIndex() << ", " << runTime.timeOutputValue()
                 << ")" << endl;
            checkData.replaceCheckpoint(currCheck++);
        }
    }
    lastTimeIndex = std::pair<scalar,label>(runTime.timeOutputValue(), runTime.timeIndex());

    targetTime = runTime.timeIndex();
    endTimeID = runTime.timeIndex();

    AD::switchTapeToActive();
    auto startPosition = AD::getTapePosition();
    checkData.registerAdjoints();

    scalar J = checkControl.calcCost();
    if(Pstream::master())
    {
        AD::derivative(J) = 1.0;
    }
    AD::interpretTapeTo(startPosition);
    checkData.storeAdjoints();
    AD::resetTapeTo(startPosition);

    // adjoin one iteration at a time
    for(label i=0; i< nSteps; i++){
        AD::switchTapeToPassive();
        // restore nearest checkpoint
        label targetTime = endTimeID - i;
        if(targetTime == 1){
            checkData.restoreCheckpoint(0);
        }else{
            for (int j = currCheck - 1; j >= 0; j--)
            {
                if ((checkData.getCheckTimes()[j]).first < targetTime)
                { // restore if time index smaller than target time
                    checkData.restoreCheckpoint(j);
                    Info << "CHECK: restore Checkpoint " << j << " ("
                            << runTime.timeIndex() << ", "
                            << runTime.timeOutputValue() << ")" << endl;
                    break;
                }
            }
        }
        Info << "CHECK: Advance passive from " << runTime.timeIndex() << " to "
             << targetTime-1 << endl;
        while(runTime.timeIndex() < targetTime-1){
            checkControl.runStep();
        }
        // adjoin single (last) iteration step
        Info << "CHECK: Advance active from " << runTime.timeIndex() << " to " << targetTime << endl;
        AD::switchTapeToActive();
        checkData.registerAdjoints(); // register variables
        checkControl.runStep();

        checkData.restoreAdjoints();
        AD::interpretTapeTo(startPosition);
        checkData.storeAdjoints();
        AD::resetTapeTo(startPosition);
        checkControl.postInterpret();
    }
    runTime.setTime(lastTimeIndex.first, lastTimeIndex.second);
    runTime.stopAt(Foam::Time::stopAtControls::saWriteNow);
    checkControl.write();
}
#endif
