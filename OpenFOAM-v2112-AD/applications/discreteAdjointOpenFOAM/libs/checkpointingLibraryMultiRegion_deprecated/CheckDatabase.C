#include "CheckDatabase.H"
#include "CheckDict.H"

#include "CheckObjectScalar.H"
#include "CheckObjectVector.H"

CheckDatabase::CheckDatabase(Time& runTime, CheckDict& checkDict)
    : runTime(runTime), checkObjects(checkDict.getCheckObjects())
{
}

CheckDatabase::~CheckDatabase()
{
    for (auto* o : checkObjects)
    {
        delete o;
    }
}

label CheckDatabase::nCheckpoints() const { return checkTimes.size(); }

#if defined(DAOF_AD_MODE_A1S)
void CheckDatabase::registerAdjoints(const bool alwaysRegister)
{
    for (auto* o : checkObjects)
    {
        o->registerAdjoints(alwaysRegister);
    }
}

void CheckDatabase::registerAsOutput()
{
    for (auto* o : checkObjects)
    {
        o->registerAsOutput();
    }
}

void CheckDatabase::storeAdjoints()
{
    for (auto* o : checkObjects)
    {
        o->storeAdjoints();
    }
}

void CheckDatabase::restoreAdjoints()
{
    for (auto* o : checkObjects)
    {
        o->restoreAdjoints();
    }
}

double CheckDatabase::calcNormOfStoredAdjoints()
{
    scalar norm = 0;
    for (auto* o : checkObjects)
    {
        norm += o->calcNormOfStoredAdjoints();
    }
    Foam::reduce(norm, sumOp<scalar>());
    return AD::passiveValue(norm);
}
#endif

void CheckDatabase::addCheckpoint()
{
    for (auto* o : checkObjects)
    {
        o->addCheckpoint();
    }
    std::pair<int, double> tmp(runTime.timeIndex(),
                               AD::passiveValue(runTime.timeOutputValue()));
    checkTimes.push_back(tmp);
}

void CheckDatabase::replaceCheckpoint(int id)
{
    // create missing checkpoints on the fly
    for (int i = checkTimes.size(); i <= id; i++)
    {
        addCheckpoint();
    }

    for (auto* o : checkObjects)
    {
        o->replaceCheckpoint(id);
    }
    std::pair<int, double> tmp(runTime.timeIndex(),
                               AD::passiveValue(runTime.timeOutputValue()));
    checkTimes[id] = tmp;
}

void CheckDatabase::restoreCheckpoint(int id)
{
    for (auto* o : checkObjects)
    {
        o->restoreCheckpoint(id);
    }
    runTime.setTime(checkTimes[id].second, checkTimes[id].first);
}

void CheckDatabase::listCheckTimes()
{
    Info << "checkpoints at:" << endl;
    for (unsigned int i = 0; i < checkTimes.size(); i++)
    {
        if (i == 0 || checkTimes[i].first > 0)
        {
            Info << checkTimes[i].first << ", ";
        }
    }
    Info << endl;
}

double CheckDatabase::getCheckDatabaseSize()
{
    double memMB = 0;
    for (auto* o : checkObjects)
    {
        Info << "Name: " << o->name() << " Size: " << o->getObjectSize()
             << endl;
        memMB += o->getObjectSize();
    }
    return memMB;
}

std::vector<std::pair<int, double>>& CheckDatabase::getCheckTimes()
{
    return checkTimes;
}

void CheckDatabase::addCheckObject(CheckBaseObject* obj)
{
    // append required numbers of checkpoints to newly generated checkpoint
    // Object
    for (unsigned int i = 0; i < checkTimes.size(); i++)
    {
        obj->addCheckpoint();
    }
    checkObjects.push_back(obj);
}

void CheckDatabase::addScalarCheckpoint(scalar& s){
    addCheckObject(new CheckObjectScalar(s));
}

void CheckDatabase::addVectorCheckpoint(Foam::vector& v){
    addCheckObject(new CheckObjectVector(v));
}

