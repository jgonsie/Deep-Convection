    // Solve the Momentum equation

    MRF.correctBoundaryVelocity(U);

    scalar logistic_k = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<scalar>("logistic_k",2.0);
    scalar logistic_shift = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<scalar>("logistic_shift",5.0);
    scalar alphaMax = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<scalar>("alphaMax",1000);

    penalty = 1.0 / (scalar(1.0)+exp(-logistic_k*(alpha-logistic_shift)));

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(rho, U)
      + turbulence->divDevRhoReff(U)
      + fvm::Sp(penalty*dimensionedScalar("1",dimMass/dimTime/dimLength/dimLength/dimLength,alphaMax), U)
     ==
        fvOptions(rho, U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (simple.momentumPredictor())
    {
        solve
        (
            UEqn
         ==
            fvc::reconstruct
            (
                (
                  - ghf*fvc::snGrad(rho)
                  - fvc::snGrad(p_rgh)
                )*mesh.magSf()
            )
        );

        fvOptions.correct(U);
    }
