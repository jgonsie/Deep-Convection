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
    ADmode::global_tape->switch_to_active();
    auto reset_to = ADmode::global_tape->get_position();

    int i = 0;
    do
    {
        i++;

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

        ADmode::global_tape->interpretTape();
        checkControl.write();

        checkData.storeAdjoints();
        ADmode::global_tape->reset_to(reset_to);
        ADmode::global_tape->zeroAdjointVector();

    } while(true);
}
#endif
