    // Solve the Momentum equation

    MRF.correctBoundaryVelocity(U);

    penalty = 1.0 / (scalar(1.0)+exp(-logistic_k*(alpha-logistic_shift)));

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(rho, U)
      + turb.divDevRhoReff(U)
      + fvm::Sp(penalty*dimensionedScalar("1",dimMass/dimTime/dimLength/dimLength/dimLength,alphaMax), U)
     ==
        fvOptions(rho, U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (momentumPredictor)
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
