#ifndef CHECKEQUI_H
#define CHECKEQUI_H

#include "CheckMethod.H"

class CheckController;

class CheckEquidistant : public CheckMethod
{
  public:
    CheckEquidistant(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& timeReg);
    void run(CheckController& checkControl);
};

#endif // CHECKEQUI_H
