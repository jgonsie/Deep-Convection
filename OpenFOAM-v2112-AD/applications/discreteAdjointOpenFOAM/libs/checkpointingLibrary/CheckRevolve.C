//#define DAOF_AD_MODE_A1S
#if defined(DAOF_AD_MODE_A1S)
#include <stack>

#include "CheckController.H"
#include "CheckDatabase.H"
#include "CheckRevolve.H"

CheckRevolve::CheckRevolve(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& runTime)
    : 
    CheckMethod(cd, checkDict, runTime), 
    firstRun(true),
    revolve(nCheckpoints) /* online checkpointing */
{
}

void CheckRevolve::run(CheckController& checkControl)
{
    auto startPosition = AD::getTapePosition();
    auto positionTapeSwitchOn =
        AD::getTapePosition(); // save position in order to interpret until here

    std::pair<scalar,label> lastTimeIndex(0,0);
    AD::switchTapeToPassive();

    enum ACTION::action whatodo;
    label nSteps = 0;
    do
    {
        whatodo = revolve.revolve();
        // store checkpoint and return
        if (whatodo == ACTION::takeshot)
        {
            Info << "CHECK: store Checkpoint " << revolve.getcheck() << " ("
                 << runTime.timeIndex() << ", " << runTime.timeOutputValue()
                 << ")" << endl;
            checkData.replaceCheckpoint(revolve.getcheck());
        }
        // restore checkpoint and return
        else if (whatodo == ACTION::restore)
        {
            checkData.restoreCheckpoint(revolve.getcheck());
            Info << "CHECK: restore Checkpoint " << revolve.getcheck() << " ("
                 << runTime.timeIndex() << ", " << runTime.timeOutputValue()
                 << ")" << endl;
        }
        // run passively until it is time to switch on tape
        else if (whatodo == ACTION::advance)
        {
            int nAdvanceSteps = revolve.getcapo() - revolve.getoldcapo();
            Info << "CHECK: Advance passive from " << revolve.getoldcapo()
                 << " to " << revolve.getcapo() << endl;
            for (int i = 0; i < nAdvanceSteps; i++)
            {
                bool endReached = checkControl.runStep();
                nSteps++;
                if(endReached){ // lastTimeIndex can be different to what we expect due to residualControl
                    lastTimeIndex = std::pair<scalar,label>(runTime.timeOutputValue(), runTime.timeIndex());
                    revolve.turn(nSteps);
                    firstRun = false;
                    break;
                }
            }
        }
        else if (whatodo == ACTION::firsturn) // adjoin cost functional back to state after last iteration
        {
            // calc cost and interpret
            AD::switchTapeToActive();
            auto p = AD::getTapePosition();
            checkData.registerAdjoints();
            scalar J = checkControl.calcCost();
            if(Pstream::master())
            {
                AD::derivative(J) = 1.0;
            }
            AD::switchTapeToPassive();
            AD::interpretTapeTo(p);
            checkData.storeAdjoints();
            AD::resetTapeTo(p);
        }
        // active section
        else if (whatodo == ACTION::youturn)
        {
            if (whatodo == ACTION::firsturn)
            {
                startPosition = AD::getTapePosition();
            }

            AD::switchTapeToActive(); // switch on tape
            AD::resetTapeTo(startPosition);
            checkData.registerAdjoints();
            positionTapeSwitchOn = AD::getTapePosition();

            Info << "CHECK: Advance active from " << revolve.getoldcapo()
                 << " to " << revolve.getoldcapo() + 1 << endl;

            checkControl.runStep();
            checkData.restoreAdjoints();

            //Info << "Tape size: "  << (dco::size_of(ADmode::global_tape) / 1024.0 / 1024.0) << " MB" << endl;
            AD::interpretTapeTo(positionTapeSwitchOn);
            AD::resetTapeTo(positionTapeSwitchOn);
            checkControl.postInterpret();
            checkControl.write();

            checkData.storeAdjoints(); // store adjoints
            AD::switchTapeToPassive(); // switch off tape
        }
        if (whatodo == ACTION::error)
        {
            Info << " irregular termination of revolve " << endl;
        }
    } while((whatodo != ACTION::terminate) && (whatodo != ACTION::error));

    // make sure sensitivity result is written to last iteration step
    runTime.setTime(lastTimeIndex.first, lastTimeIndex.second);
    runTime.stopAt(Foam::Time::stopAtControls::saWriteNow);
    checkControl.write();
}
#endif
