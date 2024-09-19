#include "CheckDatabase.H"
#include "CheckDict.H"
#include "CheckMethod.H"

CheckMethod::CheckMethod(CheckDatabase& cd,
                         CheckDict& checkDict, Foam::Time& runTime)
    : checkData(cd), runTime(runTime)
{
    startTimeID = runTime.startTimeIndex();
    deltaT = runTime.deltaT().value();

    startTime = AD::passiveValue(runTime.startTime().value());
    endTime = AD::passiveValue(runTime.endTime().value());

    // Info << "END TIME " << runTime.endTime().value() << " STOP AT " <<
    // runTime.stopAt().value() << endl;
    if (deltaT != 0 && runTime.endTime().value() < 1e10)
    {
        targetTime = static_cast<Foam::label>(
            (AD::passiveValue(runTime.endTime().value()) /
             AD::passiveValue(deltaT)));
        endTimeID = runTime.startTimeIndex() +
                    label(AD::passiveValue((endTime - startTime) / deltaT));
        nCalcSteps = label(AD::passiveValue((endTime - startTime) / deltaT));
    }
    else
    {
        targetTime = 0;
        endTimeID = startTimeID;
        nCalcSteps = 0;
    }

    nCheckpoints = checkDict.subDict("checkpointSettings").lookupOrDefault<label>("nCheckpoints",1);
}

int CheckMethod::getTargetTime() { return targetTime; }
