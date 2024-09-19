#ifndef CheckInterface_H
#define CheckInterface_H
#include "CheckDatabase.H"
#include "CheckDict.H"
#include <memory>

class CheckMethod;
class CheckController;

#include "CheckActions.H"

#include "Time.H"

using namespace Foam;

class CheckInterface
{
  public:
    CheckInterface(Time& runTime);
    ~CheckInterface();
    void run(CheckController& checkController);
    int getTargetTime();
    double getCheckpointSize();
    CheckDatabase& checkDatabase();

  private:
    //Time& runTime;
    CheckDict checkDict;
    CheckDatabase checkData;
    std::unique_ptr<CheckMethod> checkMethod;
};

#endif // CheckInterface_H
