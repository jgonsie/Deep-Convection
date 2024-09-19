#ifndef CHECKREVOLVE_H
#define CHECKREVOLVE_H

#include "CheckMethod.H"
#include "ADdefines.H"
#include "revolve.h"

class CheckController;

class CheckRevolve : public CheckMethod
{
  public:
    CheckRevolve(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& runTime);
    void run(CheckController& checkControl);

  private:
    bool firstRun;
    Revolve revolve;
};

#endif // CHECKREVOLVE_H
