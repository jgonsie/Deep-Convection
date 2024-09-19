#ifndef CHECKDATABASE_H
#define CHECKDATABASE_H

#include "CheckBaseObject.H"
#include <Time.H>

using namespace Foam;

class CheckDict;

class CheckDatabase
{
  public:
    CheckDatabase(Foam::Time& runTime, CheckDict& checkDict);
    ~CheckDatabase();

#if defined(DAOF_AD_MODE_A1S)
    void registerAdjoints(const bool = false);
    void registerAsOutput();
    // void storeTapeIndex();
    void storeAdjoints();
    void restoreAdjoints();
    double calcNormOfStoredAdjoints();
#endif
    void addCheckpoint();
    void replaceCheckpoint(int i);
    void restoreCheckpoint(int i);
    void addCheckObject(CheckBaseObject* obj);

    void addScalarCheckpoint(scalar& s);
    void addVectorCheckpoint(Foam::vector& v);

    void listCheckTimes();
    double getCheckDatabaseSize();
    label nCheckpoints() const;

    std::vector<std::pair<int, double>>& getCheckTimes();

  private:
    Foam::Time& runTime;
    std::vector<std::pair<int, double>>
        checkTimes; // store time index and time, potentially also have to store
                    // dt at some point
    std::vector<CheckBaseObject*> checkObjects;
};

#endif // CHECKDATABASE_H
