#ifndef CHECKNONE_H
#define CHECKNONE_H

#include "CheckMethod.H"

class CheckController;

class CheckNone : public CheckMethod
{
  public:
    CheckNone(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& timeReg);
    void run(CheckController& checkControl);

  private:
    bool startPositionSet;
};

#endif // CHECKEQUI_H
