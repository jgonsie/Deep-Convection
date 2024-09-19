{
    volScalarField& thermo_he = thermo.he();

    auto& turb = *turbulence;
    Info << "thermo_he name: " << thermo_he.name() << endl;
    /*volScalarField*/ alphaEff = turb.alphaEff(); // Cp*alpha
    
    dimensionedScalar fluidRho("fluidRho", rho.dimensions(), 1027);
    dimensionedScalar solidRho("solidRho", rho.dimensions(), 8930);

    volScalarField rho_thermo = penalty*solidRho + (1-penalty)*fluidRho;

    fvScalarMatrix EEqn
    (
        fvm::div(phi, thermo_he)
      + (
            thermo_he.name() == "e"
          ? fvc::div(phi, volScalarField("Ekp", 0.5*magSqr(U) + p/rho_thermo))
          : fvc::div(phi, volScalarField("K", 0.5*magSqr(U)))
        )
      - fvm::laplacian(alphaEff, thermo_he)
     /*==
        rho_thermo*(U&g)
      + radiation->Sh(thermo, thermo_he)
      + fvOptions(rho_thermo, thermo_he)*/
    );

    EEqn.relax();

    fvOptions.constrain(EEqn);

    EEqn.solve();

    fvOptions.correct(thermo_he);

    thermo.correct();
    radiation->correct();

    he == thermo_he;

    scalar fluidCp = 4195; //this->mixtureDict_.subDict("mixture").subDict("thermodynamics").getScalar("Cp");
    const scalar Tref = 298.15;
    forAll(heRef,celli){
      heRef[celli] = fluidCp*(thermo.T()[celli] - Tref); //mixture_.HE(pp[facei], pT[facei]);
    }
}
