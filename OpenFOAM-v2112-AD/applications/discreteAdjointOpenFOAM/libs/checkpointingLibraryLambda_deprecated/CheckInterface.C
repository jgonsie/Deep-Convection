#include "CheckInterface.H"
#include "CheckMethod.H"

#if defined(DAOF_AD_MODE_A1S)
#include "CheckEquidistant.H"
#include "CheckNone.H"
#include "CheckRevolve.H"
#include "CheckReverseAcc.H"
#include "CheckPiggyback.H"
#include "CheckDebug.H"
#endif
#include "CheckDummy.H"

#include "CheckController.H"
#include "CheckDatabase.H"
#include "CheckDict.H"

CheckInterface::CheckInterface(Time& runTime)
    : runTime(runTime), checkDict(runTime), checkData(runTime, checkDict),
      checkMethod(nullptr)
{
    Foam::word checkMethodChoice(
        checkDict.checkpointSettings("checkpointingMethod"));
#if defined(DAOF_AD_MODE_A1S)
    if (checkMethodChoice == "revolve")
    {
        checkMethod = make_unique<CheckRevolve>(checkData, checkDict, runTime);
    }
    else if (checkMethodChoice == "equidistant")
    {
        checkMethod = make_unique<CheckEquidistant>(checkData, checkDict, runTime);
    }
    else if (checkMethodChoice == "none")
    {
        checkMethod = make_unique<CheckNone>(checkData, checkDict, runTime);
    }
    else if (checkMethodChoice == "debug")
    {
        checkMethod = make_unique<CheckDebug>(checkData, checkDict, runTime);
    }
    else if (checkMethodChoice == "reverseAccumulation")
    {
        checkMethod = make_unique<CheckReverseAcc>(checkData, checkDict, runTime);
    }
    else if (checkMethodChoice == "piggyback")
    {
        checkMethod = make_unique<CheckPiggyback>(checkData, checkDict, runTime);
    }
    else
#endif
    if (checkMethodChoice == "dummy")
    {
        checkMethod = make_unique<CheckDummy>(checkData, checkDict, runTime);
    }
    else
    {
        std::cout << "Checkpointing Method " << checkMethodChoice
                  << " unknown! \nvalid choices are: revolve, equidistant, none"
                  << std::endl;
    }

    assert(checkMethod != NULL);
}

CheckInterface::~CheckInterface() {}

void CheckInterface::run(CheckController& checkController)
{
    checkMethod->run(checkController); 
}

int CheckInterface::getTargetTime() { return checkMethod->getTargetTime(); }

double CheckInterface::getCheckpointSize()
{
    return checkData.getCheckDatabaseSize();
}

CheckDatabase& CheckInterface::checkDatabase() { return checkData; }
