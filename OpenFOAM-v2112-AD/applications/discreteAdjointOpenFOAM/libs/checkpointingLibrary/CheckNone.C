#if defined(DAOF_AD_MODE_A1S)
#include <stack>

#include "CheckController.H"
#include "CheckDatabase.H"
#include "CheckNone.H"

CheckNone::CheckNone(CheckDatabase& cd, CheckDict& checkDict,
                     Foam::Time& runTime)
    : CheckMethod(cd, checkDict, runTime)
{
    startPositionSet = false;
}

void CheckNone::run(CheckController& checkControl)
{
    std::stack<AD::position_t> tapePositions;
    std::stack<std::pair<scalar,label>> timeSteps;

    tapePositions.push(AD::getTapePosition());
    timeSteps.emplace(runTime.timeOutputValue(),runTime.timeIndex());

    std::pair<scalar,label> lastTimeIndex(0,0);

    AD::switchTapeToActive();
    checkData.registerAdjoints();

    bool finished = false;
    while (!finished)
    {
        Info << "runTimeID: " << runTime.timeIndex() << endl;
        tapePositions.push(AD::getTapePosition());
        timeSteps.emplace(runTime.timeOutputValue(),runTime.timeIndex());
        finished = checkControl.runStep();
        runTime.write();
    }
    tapePositions.pop();
    timeSteps.pop();

    lastTimeIndex = std::pair<scalar,label>(runTime.timeOutputValue(), runTime.timeIndex());

    scalar J = checkControl.calcCost();
    if(Pstream::master()){
        AD::derivative(J)=1.0;
    }

    while (!tapePositions.empty())
    {
        auto pos = tapePositions.top();
        tapePositions.pop();
        AD::interpretTapeTo(pos);
        AD::resetTapeTo(pos);
        checkControl.postInterpret();
        runTime.setTime(timeSteps.top().first, timeSteps.top().second);
        timeSteps.pop();

        runTime++;

        if (runTime.writeTime())
        {
        //    Info << "CheckNone: WriteNow" << endl;
            checkControl.write();
        }
    }

    // write final sensitivity result into last timestep
    runTime.setTime(lastTimeIndex.first, lastTimeIndex.second);
    runTime.stopAt(Foam::Time::stopAtControls::saWriteNow);
    checkControl.write();
}
#endif
