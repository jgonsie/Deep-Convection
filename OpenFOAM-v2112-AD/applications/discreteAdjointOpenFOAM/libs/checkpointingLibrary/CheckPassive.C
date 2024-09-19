#include "CheckDatabase.H"
#include "CheckNone.H"

CheckNone::CheckNone(CheckDatabase* cd, CheckDict* checkDict,
                     Foam::Time* timeReg)
    : CheckMethod(cd, checkDict, timeReg) // call constructor of superclass
{
    startPositionSet = false;
}

CHECK::CheckFlags CheckNone::preStep()
{
    CHECK::CheckFlags returnFlags;
    returnFlags.set(CHECK::DO_STEP);

    (*timeReg)++; // advance timestep
    label runTime = timeReg->timeIndex();

    if (!startPositionSet)
    {
        startPosition = ADmode::global_tape->get_position();
        ADmode::global_tape->switch_to_active();
        startPositionSet = true;
    }

    if (runTime == targetTime)
    { // target time reached
        returnFlags.set(
            CHECK::DO_CALC_COST); // Cost Function has to be calculated and set!
                                  // Tell that to the main loop!
    }

    return returnFlags;
}

CHECK::CheckFlags CheckNone::postStep()
{
    CHECK::CheckFlags returnFlags;
    label runTime = timeReg->timeIndex();
    std::cout << "Checkpoint:: Post Step: current timestep: " << runTime
              << std::endl;

    if (!ADmode::global_tape->is_active())
    {
        timeReg->write();
    }
    else
    {
        std::cout << "Tape Switch 1 " << timeReg->elapsedCpuTime() << std::endl;
        std::cout << "Tape Switch 0 " << timeReg->elapsedCpuTime() << std::endl;
        ADmode::global_tape->switch_to_passive();
        timeReg->write();
        ADmode::global_tape->switch_to_active();
        std::cout << "Tape Switch 0 " << timeReg->elapsedCpuTime() << std::endl;
        std::cout << "Tape Switch 1 " << timeReg->elapsedCpuTime() << std::endl;
    }

    if (runTime == targetTime)
    { // target time reached
        ADmode::global_tape->interpretTape_and_reset_to(startPosition);
        returnFlags.set(CHECK::TERMINATE); // ACTION::TERMINATE
    }
    return returnFlags;
}
