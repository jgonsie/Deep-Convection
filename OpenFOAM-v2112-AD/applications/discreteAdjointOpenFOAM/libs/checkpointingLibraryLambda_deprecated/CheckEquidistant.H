#ifndef CHECKEQUI_H
#define CHECKEQUI_H

#include "CheckMethod.H"

class CheckController;

class CheckEquidistant : public CheckMethod
{
  public:
    CheckEquidistant(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& timeReg);
    void run(CheckController& checkControl);

  private:
    // label currentTime;
    bool startPositionSet;
    // ADmode::tape_t::position_t startPosition;
    // ADmode::tape_t::position_t positionTapeSwitchOn;

    int nAdjoinedSteps;
    // int nTargetAdvanceSteps;

    int checkDistance;
    int currCheck;

    bool firstRun;
    bool lastInterpretation;
};

#endif // CHECKEQUI_H
