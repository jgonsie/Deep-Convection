#ifndef CHECKREVERSE_ACC_H
#define CHECKREVERSE_ACC_H

#include "CheckMethod.H"
#include "ADdefines.H"
#include "revolve.h"

class CheckController;

class CheckReverseAcc : public CheckMethod
{
  public:
    CheckReverseAcc(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& runTime);
    void run(CheckController& checkControl);

  private:
    bool firstRun;
    bool lastInterpretation;
    int nDoneTapeSteps;
    Revolve revolve;
    const dictionary& checkpointSettings;
};

#endif // CHECKREVERSE_ACC_H
