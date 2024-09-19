#ifndef checkpointing_H
#define checkpointing_H

#include "IOdictionary.H"

#include "CheckDatabase.H"
#include "fvCFD.H"
#include <vector>

using namespace Foam;

class CheckDict : public IOdictionary
{
  private:
    dictionary checkpointSettings_;
    dictionary checkpointRequired_;

    // Private Member Functions
    //- Clear the dictionaries and streams before reading
    void clear();

    //- Disallow default bitwise copy construct
    CheckDict(const CheckDict&);

    //- Disallow default bitwise assignment
    void operator=(const CheckDict&);

    template <class Type, template <class> class PatchField, class GeoMesh>
    void getObjectsOfType(const Foam::fvMesh& mesh,
                          std::vector<CheckBaseObject*>&,std::string);

    bool reverseAccumulation;
    label targetTime;
    static int debug;
    Time& runTime;

  public:
    CheckDict(Time& runTime);
    const dictionary& checkpointDict() const;
    ITstream& checkpointSettings(const word& name) const;
    bool checkpointRequired(const word& name) const;
    bool read();
    std::vector<CheckBaseObject*> getCheckObjects();
    bool& reverseAccumulationEnabled();
};

//#include "checkpointingI.H"

#endif
