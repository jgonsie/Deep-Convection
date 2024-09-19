#include <map>
namespace symbolic{
    using namespace Foam;
#if defined(ENABLE_SDLS)
    template<class Type>
    inline void fillSolverGap
    (
        const Foam::direction cmpt,
        const Foam::fvMatrix<Type>& A,
        const Foam::volScalarField& x,
        const Foam::scalarField& b
    ){
        static std::map<Foam::word,Foam::scalarField> old_a1b_map;

        bool tapeWasActive = AD::switchTapeToPassive();

        Foam::word fieldNameCmpt = x.name().lessExt();
        if((cmpt < 3) || strcmp(pTraits<Type>::typeName,"scalar")){
            fieldNameCmpt = fieldNameCmpt + Foam::pTraits<Type>::componentNames[cmpt];
        }

        // incoming adjoints of solution x
        Foam::scalarField a1_x(x.size());
        for(int i = 0; i < a1_x.size(); i++){
            a1_x[i] = AD::derivative(x[i]);
        }
        if(gSumMag(a1_x)>0){
            Foam::fvMatrix<Type> A_T(A); // will hold transpose of A
            int nu = 0; if(A.hasUpper()) nu = A.upper().size();
            int nl = 0; if(A.hasLower()) nl = A.lower().size();
            int nd = A.diag().size();

            // get interfaces for dimension cmpt, component is ignored for scalar field
            Foam::FieldField<Foam::Field,Foam::scalar> bcmpts = A_T.boundaryCoeffs().component(cmpt)();
            Foam::FieldField<Foam::Field,Foam::scalar> icmpts = A_T.internalCoeffs().component(cmpt)();

            // transpose matrix if necessary
            bool sym = A.symmetric();
            bool diagonal = A.diagonal();
            if(!sym && !diagonal){
                A_T.lower() = A.upper();
                A_T.upper() = A.lower();
                // switch boundary and internal coeffs for transposed
                icmpts = A_T.boundaryCoeffs().component(cmpt)();
                bcmpts = A_T.internalCoeffs().component(cmpt)();
            }

            // Adressing for upper and lower half of matrix
            const Foam::labelUList& uAddr = A_T.lduAddr().upperAddr();
            const Foam::labelUList& lAddr = A_T.lduAddr().lowerAddr();

            Foam::word reverseFieldName = x.name().lessExt() + Foam::word("Reverse");
            // construct fields a1_x, a1_b from x (for reference to mesh etc)
            Foam::volScalarField a1_b(reverseFieldName,x);
            for(int i = 0; i < a1_b.size(); i++){
                a1_b[i] = 0; // initialize b1
            }

            const dictionary& fvSolutionDict = a1_b.mesh().solutionDict();
            dictionary reverseSolverControls;
            if(fvSolutionDict.subDict("solvers").found(reverseFieldName)){
                // use settings of reverse entry
                //Info << "Using solver dict " << reverseFieldName << endl;
                reverseSolverControls = a1_b.mesh().solverDict(reverseFieldName);
            }else{
                // use same settings as primal solver
                //Info << "Using solver dict " << x.name().lessExt() << endl;
                reverseSolverControls = a1_b.mesh().solverDict(x.name().lessExt());
            }

            //const dictionary& reverseSolverControls = a1_b.mesh().solverDict(reverseFieldName);
                    /*(
                        a1_b.select
                        (
                            a1_b.mesh().data::template lookupOrDefault<bool>
                            ("finalIteration", false)
                        )
                    );*/

            bool storeAdjointSolution = reverseSolverControls.lookupOrDefault<Foam::Switch>("storeAdjointSolution",true);
            if(storeAdjointSolution && old_a1b_map.count(fieldNameCmpt)>0){
                a1_b.primitiveFieldRef() = old_a1b_map[fieldNameCmpt];
            }

            Foam::autoPtr<Foam::lduMatrix::solver> solver = Foam::lduMatrix::solver::New
                    (
                        fieldNameCmpt + Foam::word("Reverse"),
                        A_T,
                        bcmpts,
                        icmpts,
                        a1_b.boundaryField().scalarInterfaces(),
                        reverseSolverControls
                    );

            // solve for b_1
            Foam::solverPerformance solverPerf = solver->solve(a1_b.primitiveFieldRef(), a1_x);
            solverPerf.print(Foam::Info.masterStream(A.mesh().comm()));

            // store a1_b to speed up solution next time
            if(storeAdjointSolution){
                old_a1b_map[fieldNameCmpt] = a1_b.primitiveField();
            }

            // increment input adjoint for b
            for(int i = 0; i < a1_b.size(); i++){
                AD::derivative(b[i]) += AD::passiveValue(a1_b[i]);
            }

            // increment input adjoint for A (upper part)
            if(A.hasUpper()){
                if(sym){
                    for(int i = 0; i < nu; i++){
                        const double tmp = AD::passiveValue(-a1_b[lAddr[i]]*x[uAddr[i]] - a1_b[uAddr[i]]*x[lAddr[i]]);
                        AD::derivative(A.upper()[i]) += tmp;
                    }
                }else{
                    for(int i = 0; i < nu; i++){
                        const double tmp = AD::passiveValue(-a1_b[lAddr[i]]*x[uAddr[i]]);
                        AD::derivative(A.upper()[i]) += tmp;
                    }
                }
            }

            // increment input adjoint for A (lower part)
            if(A.hasLower()){
                if(sym){
                    for(int i = 0; i < nl; i++){
                        const double tmp = AD::passiveValue(-a1_b[uAddr[i]]*x[lAddr[i]] - a1_b[lAddr[i]]*x[uAddr[i]]);
                        AD::derivative(A.lower()[i]) += tmp;
                    }
                }else{
                    for(int i = 0; i < nl; i++){
                        const double tmp = AD::passiveValue(-a1_b[uAddr[i]]*x[lAddr[i]]);
                        AD::derivative(A.lower()[i]) += tmp;
                    }
                }
            }

            // increment input adjoint for A (diag part)
            for(int i = 0; i < nd; i++){
                AD::derivative(A.diag()[i]) += AD::passiveValue(-a1_b[i]*x[i]);
            }

            Foam::volScalarField xSF(x); // copy x because modified by MPI comm
            forAll(x.boundaryField(),i){
                if(x.boundaryField().types()[i] == "processor"){
                    // swap boundaries around
                    a1_b.boundaryFieldRef()[i].initEvaluate();
                    a1_b.boundaryFieldRef()[i].evaluate();
                    xSF.boundaryFieldRef()[i].initEvaluate();
                    xSF.boundaryFieldRef()[i].evaluate();

                    const scalarField a1_b_this = a1_b.boundaryField()[i].patchInternalField()();
                    const scalarField x_other = xSF.boundaryField()[i].patchNeighbourField()();
                    const scalarField procBoundary = A.boundaryCoeffs().component(cmpt)()[i]; // why not reference?

                    forAll(procBoundary,j){
                        const double t1 = AD::passiveValue( x_other[j] );
                        const double t2 = AD::passiveValue( a1_b_this[j] );
                        //Pout << "Increment " << AD::tapeIndex(procBoundary[j]) << " with " << AD::derivative(procBoundary[j]) << " + " << t1*t2 << endl;
                        AD::derivative(procBoundary[j]) += AD::passiveValue(t1*t2);
                    }
                }
            }
        }else{
            Info << "skipping " << fieldNameCmpt << " because zero incoming adjoints" << endl;
        }

        // if tape was active before entering this switch back on
        AD::switchTapeTo(tapeWasActive);
    }
#endif
}; // end namespace

