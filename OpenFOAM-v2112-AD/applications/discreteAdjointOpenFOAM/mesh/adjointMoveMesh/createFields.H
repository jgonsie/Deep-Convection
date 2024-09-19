// Info<< "sensSurfaceVec p\n" << endl;
// surfaceVectorField sensSurfaceVec
// (
//     IOobject
//     (
//         "sensSurfaceVec",
//         runTime.timeName(),
//         mesh,
//         IOobject::MUST_READ,
//         IOobject::AUTO_WRITE
//     ),
//     mesh
// );

Info<< "Reading field surfSens\n" << endl;
surfaceScalarField surfSens
(
    IOobject
    (
        "surfSens",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    mesh
);

// pointVectorField sensPointVec
// (
//     IOobject
//     (
//         "sensPointVec",
//         runTime.timeName(),
//         mesh,
//         IOobject::READ_IF_PRESENT,
//         IOobject::AUTO_WRITE
//     ),
//     pointMesh::New(mesh),
//     // mesh,
//     dimensionedVector("(0,0,0)", dimless, vector::zero),
//     "fixedValue"
// );
