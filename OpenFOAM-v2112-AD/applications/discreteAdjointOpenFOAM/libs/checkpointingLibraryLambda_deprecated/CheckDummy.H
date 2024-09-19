#ifndef CHECKDummy_H
#define CHECKDummy_H

#include "CheckMethod.H"

class CheckDummy : public CheckMethod
{
  public:
    CheckDummy(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& runTime);
    void run(CheckController& checkControl);
};

#endif // CHECKEQUI_H
