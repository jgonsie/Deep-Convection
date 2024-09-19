#ifndef CHECK_PIGGY_H
#define CHECK_PIGGY_H

#include "CheckMethod.H"
#include "ADdefines.H"
#include "revolve.h"

class CheckController;

class CheckPiggyback : public CheckMethod
{
  public:
    CheckPiggyback(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& runTime);
    void run(CheckController& checkControl);

  private:
    bool firstRun;
    bool lastInterpretation;
    int nDoneTapeSteps;
    Revolve revolve;
    const dictionary& checkpointSettings;
};

#endif // CHECK_PIGGY_H
