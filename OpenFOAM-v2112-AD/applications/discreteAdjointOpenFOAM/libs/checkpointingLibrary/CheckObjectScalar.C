//#include "CheckObjectScalar.H"

CheckObjectScalar::CheckObjectScalar(scalar& object, bool reverseAccumulation)
    : objRef(object)
{
}

void CheckObjectScalar::addCheckpoint()
{
    checkpoints.push_back(AD::value(objRef));
}

void CheckObjectScalar::replaceCheckpoint(int i)
{
    checkpoints[i] = AD::value(objRef);
}

void CheckObjectScalar::restoreCheckpoint(int i) { objRef = checkpoints[i]; }

#if defined(DAOF_AD_MODE_A1S)
void CheckObjectScalar::registerAdjoints()
{
    AD::registerInputVariable(objRef);
    tapeIndexStore = AD::tapeIndex(objRef);
}

void CheckObjectScalar::registerAsOutput()
{
    AD::registerOutputVariable(objRef);
}

void CheckObjectScalar::storeAdjoints()
{
    adjointStore = AD::adjointFromIndex(tapeIndexStore);
}

void CheckObjectScalar::restoreAdjoints()
{
    AD::derivative(objRef) = adjointStore;
}
#endif

double CheckObjectScalar::getObjectSize()
{
    return (sizeof(long int) + (checkpoints.size() + 1) * sizeof(double)) /
           1024 / 1024; // return MB
}

double CheckObjectScalar::calcNormOfStoredAdjoints()
{
    return fabs(AD::passiveValue(adjointStore));
}
