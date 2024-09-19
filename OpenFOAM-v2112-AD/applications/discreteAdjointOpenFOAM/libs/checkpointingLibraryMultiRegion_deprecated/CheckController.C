#include "CheckActions.H"
#include "CheckController.H"
#include <bitset>

CheckController::CheckController(Foam::Time& runTime /*, Foam::fvMesh& mesh*/)
    : runTime(runTime), /*mesh(mesh),*/ interface(runTime)
{
}

void CheckController::postInterpret()
{
    // default empty
}

void CheckController::run() { interface.run(*this); }

CheckInterface& CheckController::checkInterface() { return interface; }
