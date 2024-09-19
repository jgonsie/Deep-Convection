Info<< "These are the arguments: \narg0: " << argv[0]<<"\narg1: "<< argv[1]<<"\narg2: "<< argv[2]<<endl;

// Get the shared memory identifier from the command line argument
int SM_input_id = atoi(argv[1]);

// Attach the shared memory segment to pointer
char *input_dict = static_cast<char *>(shmat(SM_input_id, NULL, 0));
if (input_dict == (char *)-1) {
	perror("shmat");
	exit(1);
}
// use the array
dictionary dictArg;
Foam::IStringStream ss(input_dict);
Foam::ISstream is(ss);
dictArg.read(is);
//Info<< "Mi diccionario es:\n"<<dictArg << endl;


if (dictArg.found("p_dict"))
{
	Info<< "Initializing field p from RAM reading\n" << endl;
}
else
{
	Info<<"FATAL ERROR: It was not possible to read p field"<< endl;
	exit(-1);
}

dictionary p_dict
(
dictArg.subDict("p_dict")
);

volScalarField p
(
	IOobject
	(
		"p",
		runTime.timeName(),
		mesh,
		IOobject::NO_READ,
		IOobject::NO_WRITE
	),
	mesh,
	p_dict
);    

//-------------------------------------------------------------------------

if (dictArg.found("U_dict"))
{
	Info<< "Initializing field U from RAM reading\n" << endl;
}
else
{
	Info<<"FATAL ERROR: It was not possible to read U field"<< endl;
	exit(-1);
}

dictionary U_dict
(
dictArg.subDict("U_dict")
);

volVectorField U
(
	IOobject
	(
		"U",
		runTime.timeName(),
		mesh,
		IOobject::NO_READ,
		IOobject::NO_WRITE
	),
	mesh,
	U_dict
);

//-------------------------------------------------------------------------

if (dictArg.found("Ufaces_dict"))
{
	Info<< "Initializing field Ufaces from RAM reading\n" << endl;
}
else
{
	Info<<"FATAL ERROR: It was not possible to read Ufaces field"<< endl;
	exit(-1);
}

dictionary Ufaces_dict
(
	dictArg.subDict("Ufaces_dict")
);

surfaceVectorField Ufaces
(
	IOobject
	(
		"Ufaces",
		runTime.timeName(),
		mesh,
		IOobject::NO_READ,
		IOobject::NO_WRITE
	),
	mesh,
	Ufaces_dict
);

//-------------------------------------------------------------------------

if (dictArg.found("k_dict"))
{
	Info<< "Initializing field k from RAM reading\n" << endl;
}
else
{
	Info<<"FATAL ERROR: It was not possible to read k field"<< endl;
	exit(-1);
}

dictionary k_dict
(
	dictArg.subDict("k_dict")
);

//-------------------------------------------------------------------------

if (dictArg.found("omega_dict"))
{
	Info<< "Initializing field Omega from RAM reading\n" << endl;
}
else
{
	Info<<"FATAL ERROR: It was not possible to read Omega field"<< endl;
	exit(-1);
}

dictionary Omega_dict
(
	dictArg.subDict("omega_dict")
);

//-------------------------------------------------------------------------

if (dictArg.found("nut_dict"))
{
	Info<< "Initializing field nut from RAM reading\n" << endl;
}
else
{
	Info<<"FATAL ERROR: It was not possible to read nut field"<< endl;
	exit(-1);
}

dictionary nut_dict
(
	dictArg.subDict("nut_dict")
);     

//-------------------------------------------------------------------------

if (dictArg.found("startTime"))
{
	Info<< "StartTime from RAM reading\n" << endl;
}
else
{
	Info<<"FATAL ERROR: It was not possible to read startTime"<< endl;
	exit(-1);
}
scalar startTime (readScalar(dictArg.lookup("startTime")));
//-------------------------------------------------------------------------

if (dictArg.found("endTime"))
{
	Info<< "EndTime from RAM reading\n" << endl;
}
else
{
	Info<<"FATAL ERROR: It was not possible to read endTime"<< endl;
	exit(-1);
}
scalar endTime (readScalar(dictArg.lookup("endTime")));
//-------------------------------------------------------------------------
surfaceScalarField phi(
		IOobject
		(
			"phi",
			runTime.timeName(),
			mesh,
			IOobject::NO_READ,
			IOobject::NO_WRITE
		),
		mesh,
		dimensionedScalar("phi", dimensionSet(0,3,-1,0,0,0,0), 0)
	);

if (dictArg.found("phi_dict"))
{
	Info<< "Initializing field phi from RAM reading\n" << endl;
	dictionary phi_dict(
		dictArg.subDict("phi_dict")
	);
	phi.~GeometricField();
    new (&phi) surfaceScalarField(
		IOobject
		(
			"phi",
			runTime.timeName(),
			mesh,
			IOobject::NO_READ,
			IOobject::NO_WRITE
		),
		mesh,
		phi_dict
	);
}
else
{
	#include "createPhiRAM.H"
}
//-------------------------------------------------------------------------

if (dictArg.found("dU_dict"))
{
	Info<< "Initializing field dU from RAM reading\n" << endl;
}
else
{
	Info<<"FATAL ERROR: It was not possible to read U field"<< endl;
	exit(-1);
}

dictionary dU_dict
(
dictArg.subDict("dU_dict")
);

volVectorField dU
(
	IOobject
	(
		"dU",
		runTime.timeName(),
		mesh,
		IOobject::NO_READ,
		IOobject::NO_WRITE
	),
	mesh,
	dU_dict
);

label nCells = U.size();
std::vector<double> U_bar(2*nCells);
for(int i=0; i<nCells; i++){
		U_bar[i*2+0] = static_cast <double> (AD::value(dU[i][0]));
		U_bar[i*2+1] = static_cast <double> (AD::value(dU[i][1]));
}
	
	


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, pimple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());


singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

#include "createMRF.H"
#include "createFvOptions.H"
