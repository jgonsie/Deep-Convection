//#define DAOF_AD_MODE_A1S
#if defined(DAOF_AD_MODE_A1S)
#include "CheckController.H"
#include "CheckDatabase.H"
#include "CheckEquidistant.H"

CheckEquidistant::CheckEquidistant(CheckDatabase& cd, CheckDict& checkDict,
                                   Foam::Time& runTime)
    : CheckMethod(cd, checkDict, runTime)
{
    nAdjoinedSteps = 0;
    targetTime = endTimeID;
    firstRun = true;
    lastInterpretation = false;
    startPositionSet = false;
    // don't create more checkpoints than needed
    checkDistance =
        max(nCalcSteps / nCheckpoints + (nCalcSteps % nCheckpoints != 0),
            nTapeSteps);
    currCheck = 0;
}

void CheckEquidistant::run(CheckController& checkControl)
{
    auto startPosition = ADmode::global_tape->get_position();
    ADmode::global_tape->switch_to_passive();
    while (targetTime > startTime)
    {
        // find and restore next matching checkpoint
        if (!firstRun)
        {
            label restoreTime = max((targetTime - nTapeSteps),
                                    startTimeID); // don't allow negative times
            for (int j = currCheck - 1; j >= 0; j--)
            {
                if ((checkData.getCheckTimes()[j]).first <= restoreTime)
                { // time index smaller than target time?
                    checkData.restoreCheckpoint(j);
                    Info << "CHECK: restore Checkpoint " << j << " ("
                         << runTime.timeIndex() << ", "
                         << runTime.timeOutputValue() << ")" << endl;
                    break;
                }
            }
        }
        // perform passive steps
        if (runTime.timeIndex() < targetTime - nTapeSteps)
        {
            Info << "CHECK: Advance passive from " << runTime.timeIndex()
                 << " to " << targetTime - nTapeSteps << endl;
        }
        while (runTime.timeIndex() < targetTime - nTapeSteps)
        {
            // try to align positions of checkpoints so that the least amout of
            // recomputation needs to be done
            if (firstRun &&
                (runTime.timeIndex() - startTimeID) % checkDistance == 0)
            {
                Info << "CHECK: store Checkpoint " << currCheck << " ("
                     << runTime.timeIndex() << ", " << runTime.timeOutputValue()
                     << ")" << endl;
                checkData.replaceCheckpoint(currCheck++);
            }
            checkControl.runStep();
            if (firstRun)
            {
                checkControl.write(firstRun);
            }
        }

        std::stack<ADmode::tape_t::position_t> tapePositions;
        std::stack<scalar> timeSteps;

        // perform active steps
        ADmode::global_tape->switch_to_active(); // switch on tape
        ADmode::global_tape->reset_to(startPosition);
        checkData.registerAdjoints(); // register variables

        int nActiveSteps = min(nTapeSteps, targetTime - runTime.timeIndex());
        Info << "CHECK: Advance active from " << runTime.timeIndex() << " to "
             << runTime.timeIndex() + nActiveSteps << "( " << nActiveSteps
             << " )" << endl;
        for (int i = 0; i < nActiveSteps; i++)
        {
            tapePositions.push(ADmode::global_tape->get_position());
            timeSteps.push(runTime.timeOutputValue());
            checkControl.runStep();
            if (firstRun)
                checkControl.write(firstRun);
        }
        // calc cost function
        if (firstRun)
        {
            checkControl.calcCost();
            firstRun = false;
        }
        else
        {
            checkData.restoreAdjoints(); // restore adjoints if resume of
                                         // previous interpretation
        }

        // interpret tape one step at a time
        while (!tapePositions.empty())
        {
            auto position = tapePositions.top();
            tapePositions.pop();
            ADmode::global_tape->interpretTape_and_reset_to(position);

            runTime.setTime(timeSteps.top(), runTime.timeIndex() - 1);
            timeSteps.pop();

            scalar currentTime = runTime.timeOutputValue();
            scalar nearestTime = runTime.findClosestTime(currentTime).value();
            checkControl.postInterpret();
            if (abs(nearestTime - currentTime) < doubleScalarSMALL)
            {
                checkControl.write(false);
            }
        }
        checkData.storeAdjoints(); // store adjoints
        ADmode::global_tape->zeroAdjointVectorTo(startPosition);

        ADmode::global_tape->switch_to_passive(); // switch off tape
        targetTime -= nActiveSteps;
        nAdjoinedSteps += nActiveSteps;
    }
}
#endif
