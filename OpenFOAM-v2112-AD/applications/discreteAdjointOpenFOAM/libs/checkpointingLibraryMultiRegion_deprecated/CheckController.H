#ifndef CheckController_H
#define CheckController_H
#include "CheckInterface.H"

namespace Foam
{
class Time;
class fvMesh;
} // namespace Foam

class CheckController
{
  protected:
    Foam::Time& runTime;
    //Foam::fvMesh& mesh;
    CheckInterface interface;

  public:
    CheckController(Foam::Time& runTime/*, Foam::fvMesh& mesh*/);
    CheckInterface& checkInterface();

    virtual bool runStep() = 0;
    virtual scalar calcCost() = 0;

    virtual void postInterpret();
    virtual void write(bool firstRun = false) = 0;
    virtual void run();

    virtual ~CheckController(){};
};

#endif
