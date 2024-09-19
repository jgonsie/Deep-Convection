#if defined(DAOF_AD_MODE_A1S)
#include <stack>

#include "CheckController.H"
#include "CheckDatabase.H"
#include "CheckPiggyback.H"

CheckPiggyback::CheckPiggyback(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& runTime)
    :
    CheckMethod(cd, checkDict, runTime),
    firstRun(true),
    lastInterpretation(false),
    nDoneTapeSteps(0),
    revolve(nCheckpoints), /* online checkpointing */
    checkpointSettings(checkDict.checkpointDict())
{}

void CheckPiggyback::run(CheckController& checkControl)
{
    //AD::switchTapeToActive();
    auto reset_to = AD::getTapePosition();
    AD::switchTapeToActive();
    //checkData.storeAdjoints(); // dummy store adjoints

    int i = 0;
    do
    {
        AD::zeroAdjointVector();
        checkData.registerAdjoints();

        bool endReached = checkControl.runStep();

        // end was already reached in previous iteration step, so just break here
        if(endReached){
            break;
        }

        scalar J = checkControl.calcCost();

        if(Pstream::master()){
            AD::derivative(J)=1.0;
        }
        if(i>0){
            checkData.restoreAdjoints();
        }

        AD::interpretTape();

        checkData.storeAdjoints();

        Info << "PIGGY: " << checkData.calcNormOfStoredAdjoints() << endl;
        checkControl.postInterpret();

        AD::resetTapeTo(reset_to);
        i++;
        checkControl.write();
    } while(true);
}
#endif
