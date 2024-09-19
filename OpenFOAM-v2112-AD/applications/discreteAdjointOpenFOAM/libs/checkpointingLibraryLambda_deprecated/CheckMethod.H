#ifndef CHECKMETHOD_H
#define CHECKMETHOD_H
#include "Time.H"
#include <iostream>

#include "CheckActions.H"
#include "dco.hpp"

using namespace std;

class CheckController;
class CheckDatabase;
class CheckDict;

class CheckMethod
{
  public:
    CheckMethod(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& runTime);
    virtual ~CheckMethod() {}
    virtual void run(CheckController& checkControl) = 0;
    int getTargetTime();

  protected:
    CheckController* checkControl;
    CheckDatabase& checkData;
    Foam::Time& runTime;

    Foam::label targetTime;

    Foam::scalar startTime;
    Foam::scalar endTime;
    Foam::label startTimeID;
    Foam::label endTimeID;

    Foam::scalar deltaT;

    Foam::label nCalcSteps;
    Foam::label nTapeSteps;
    Foam::label nCheckpoints;
};

#endif // CHECKMETHOD_H
