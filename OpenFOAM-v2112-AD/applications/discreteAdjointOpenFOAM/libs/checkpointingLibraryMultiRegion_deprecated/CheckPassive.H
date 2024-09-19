#ifndef CHECKNONE_H
#define CHECKNONE_H

#include "CheckMethod.H"

class CheckNone : public CheckMethod
{
  public:
    CheckNone(CheckDatabase* cd, CheckDict* checkDict, Foam::Time* timeReg);
    int preStep();
    int postStep();

  private:
    // label currentTime;
    bool startPositionSet;
    // ADmode::tape_t::position_t startPosition;
    // ADmode::tape_t::position_t positionTapeSwitchOn;
};

#endif // CHECKNONE_H
