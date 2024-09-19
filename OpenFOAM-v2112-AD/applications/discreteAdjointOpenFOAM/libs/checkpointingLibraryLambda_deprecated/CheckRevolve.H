#ifndef CHECKREVOLVE_H
#define CHECKREVOLVE_H

#include "CheckMethod.H"
#include "dco.hpp"
#include "revolve.h"

class CheckController;

class CheckRevolve : public CheckMethod
{
  public:
    CheckRevolve(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& runTime);
    void run(CheckController& checkControl);

  private:
    bool firstRun;
    bool lastInterpretation;
    int nDoneTapeSteps;
    Revolve revolve;
};

#endif // CHECKREVOLVE_H
