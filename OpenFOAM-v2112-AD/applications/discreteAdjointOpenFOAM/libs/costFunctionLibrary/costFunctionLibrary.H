#include "fvCFD.H"
// #include "fvMeshFunctionObject.H"

class CostFunction{
public:
    CostFunction(Foam::fvMesh& mesh);
    Foam::scalar eval();
private:
    Foam::fvMesh& mesh;
    Foam::scalar eval_pressure(const Foam::wordList& patches);
    Foam::scalar eval_drag(const Foam::wordList& patches);
    Foam::scalar eval_drag_legacy(const Foam::wordList& patches);
    Foam::scalar eval_liftdragratio(const Foam::wordList& patches);
    //- Return the effective viscous stress (laminar + turbulent).
    Foam::tmp<Foam::volSymmTensorField> devRhoReff() const;
    Foam::scalar rho() const;
};
