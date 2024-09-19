#include "CheckDatabase.H"
#include "CheckDummy.H"

CheckDummy::CheckDummy(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& timeReg)
    : CheckMethod(cd, checkDict, timeReg) // call constructor of superclass
{
}

void CheckDummy::run(CheckController& checkControl)
{
    Perr << "Dummy Checkpoint Library called! You should not use this..." << endl;
}
