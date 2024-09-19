#include "CheckDict.H"
#include "CheckObjectGeometricField.H"
#include "IOmanip.H"
#include "Time.H"

int CheckDict::debug(Foam::debug::debugSwitch("fvSchemes", false));

void CheckDict::clear()
{
    checkpointSettings_.clear();
    checkpointRequired_.clear();
}

CheckDict::CheckDict(Time& runTime)
    : IOdictionary(IOobject("checkpointingDict", runTime.time().system(),
                            runTime, IOobject::MUST_READ, IOobject::NO_WRITE)),
      checkpointSettings_(
          ITstream(objectPath() + "::checkpointSettings", tokenList())()),
      checkpointRequired_(
          ITstream(objectPath() + "::checkpointRequired", tokenList())()),
      runTime(runTime)
{
    read();
    reverseAccumulation = checkpointSettings_.lookupOrDefault<bool>("reverseAccumulation", false) 
        || checkpointSettings_.lookupOrDefault<Foam::word>("checkpointingMethod","") == "reverseAccumulation";
}

const dictionary& CheckDict::checkpointDict() const{
    return checkpointSettings_;
}

bool CheckDict::read()
{
    if (regIOobject::read())
    {
        const dictionary& dict = *this; // checkpointDict();

        // persistent settings across reads is incorrect
        clear();

        if (dict.found("checkpointSettings"))
        {
            checkpointSettings_ = dict.subDict("checkpointSettings");
        }

        if (dict.found("checkpointRequired"))
        {
            checkpointRequired_ = dict.subDict("checkpointRequired");
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool CheckDict::checkpointRequired(const word& name) const
{
    if (debug)
    {
        Info << "Lookup checkpointRequired for " << name << endl;
    }

    if (checkpointRequired_.found(name))
    {
        return true;
    }
    else
    {
        return false;
    }
}

Foam::ITstream& CheckDict::checkpointSettings(const word& name) const
{
    if (debug)
    {
        Info << "Lookup checkpointSettings for " << name << endl;
    }

    if (checkpointSettings_.found(name))
    {
        return checkpointSettings_.lookup(name);
    }
    else
    {
        return checkpointSettings_.lookup(name);
    }
}

// allocate checkpoint objects for all requested fields of type
template <class Type, template <class> class PatchField, class GeoMesh>
void CheckDict::getObjectsOfType(const Foam::fvMesh& mesh,
                                 std::vector<CheckBaseObject*>& objVec,std::string s)
{
    HashTable<const GeometricField<Type, PatchField, GeoMesh>*> ht =
        mesh.lookupClass<GeometricField<Type, PatchField, GeoMesh>>();
    typename HashTable<
        const GeometricField<Type, PatchField, GeoMesh>*>::iterator it =
        ht.begin();
    for (it = ht.begin(); it != ht.end(); it++)
    {
        if (checkpointRequired((*it)->name()))
        {
            GeometricField<Type, PatchField, GeoMesh>* tmp =
                (const_cast<GeometricField<Type, PatchField, GeoMesh>*>(*it));
            CheckObjectGeometricField<Type, PatchField, GeoMesh>* co =
                new CheckObjectGeometricField<Type, PatchField, GeoMesh>(
                    tmp, this->reverseAccumulation);
            objVec.push_back(co);
            Info << setw(10) << (*it)->name() << ":\t"
                 << "yes" << endl;
        }
        else
        {
            Info << setw(10) << (*it)->name() << ":\t"
                 << "no" << endl;
        }
    }
}

// allocate checkpoint objects for all requested fields
std::vector<CheckBaseObject*> CheckDict::getCheckObjects()
{
    std::vector<CheckBaseObject*> vec;
    // Iterate over all mesh regions
    for(const auto s : runTime.names<Foam::fvMesh>()){
        const Foam::fvMesh& mesh = runTime.lookupObject<Foam::fvMesh>(s);
        Info << "Creating Checkpoints for fields in " << s << ":"  << endl;
        getObjectsOfType<scalar,fvPatchField,volMesh>(mesh,vec,"volScalarField");
        getObjectsOfType<vector,fvPatchField,volMesh>(mesh,vec,"volVectorField");
        getObjectsOfType<scalar,fvsPatchField,surfaceMesh>(mesh,vec,"surfaceScalarField");
        getObjectsOfType<vector,fvsPatchField,surfaceMesh>(mesh,vec,"surfaceVectorField");
        Info << "--------------------------------" << endl;
    }
    return vec;
}

bool& CheckDict::reverseAccumulationEnabled() { return reverseAccumulation; }
