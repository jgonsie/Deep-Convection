//#include "CheckObjectDictionary.H"

CheckObjectDictionary::CheckObjectDictionary(Foam::dictionary& object, bool reverseAccumulation)
    : objRef(object)
{
}

void CheckObjectDictionary::addCheckpoint()
{
    checkpoints.push_back(objRef);
}

void CheckObjectDictionary::replaceCheckpoint(int i)
{
    checkpoints[i] = objRef;
}

void CheckObjectDictionary::restoreCheckpoint(int i) { objRef = checkpoints[i]; }

#if defined(DAOF_AD_MODE_A1S)
void CheckObjectDictionary::registerAdjoints()
{}

void CheckObjectDictionary::registerAsOutput()
{}

void CheckObjectDictionary::storeAdjoints()
{}

void CheckObjectDictionary::restoreAdjoints()
{}
#endif

double CheckObjectDictionary::getObjectSize()
{
    return 0;
}

double CheckObjectDictionary::calcNormOfStoredAdjoints()
{
    return 0;
}
