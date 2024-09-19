#ifndef CHECKDEBUG_H
#define CHECKDEBUG_H

#include "CheckMethod.H"

class CheckController;

class CheckDebug : public CheckMethod
{
  public:
    CheckDebug(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& timeReg);
    void run(CheckController& checkControl);

  private:
    bool startPositionSet;
};

#endif // CHECKEQUI_H
