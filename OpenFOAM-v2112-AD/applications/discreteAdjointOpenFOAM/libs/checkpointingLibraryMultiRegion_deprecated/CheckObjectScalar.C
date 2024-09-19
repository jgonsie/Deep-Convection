//#include "CheckObjectScalar.H"

CheckObjectScalar::CheckObjectScalar(scalar& object, bool reverseAccumulation)
    : objRef(object)
{
}

void CheckObjectScalar::addCheckpoint()
{
    checkpoints.push_back(dco::value(objRef));
}

void CheckObjectScalar::replaceCheckpoint(int i)
{
    checkpoints[i] = dco::value(objRef);
}

void CheckObjectScalar::restoreCheckpoint(int i) { objRef = checkpoints[i]; }

#if defined(DAOF_AD_MODE_A1S)
void CheckObjectScalar::registerAdjoints(const bool alwaysRegister)
{
    ADmode::global_tape->register_variable(objRef);
    tapeIndexStore = dco::tapeIndex(objRef);
}

void CheckObjectScalar::registerAsOutput()
{
    ADmode::global_tape->registerOutputVariable(objRef);
}

void CheckObjectScalar::storeAdjoints()
{
    adjointStore = ADmode::global_tape->_adjoint(tapeIndexStore);
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
