    dictionary simpleDict = mesh.solutionDict().subDict("SIMPLE");

    const wordList costFunctionPatches(
        mesh.solutionDict().subDict("SIMPLE").lookup("costFunctionPatches")
    );

    const Foam::wordList designSpaceZones(
            mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<Foam::wordList>("designSpace",Foam::wordList())
    );

    List<label> designSpaceCells;
    if(designSpaceZones.size()>0){
        forAll(designSpaceZones, i){
            const label cellZoneID = mesh.cellZones().findZoneID(designSpaceZones[i]);
            designSpaceCells.append( mesh.cellZones()[cellZoneID] );
        }
    }else{ // add all cellZones
        if(mesh.cellZones().size() > 0){
            forAll(mesh.cellZones(), i){
                designSpaceCells.append( mesh.cellZones()[i] );
            }
        }else{
            forAll(alpha,i){
                designSpaceCells.append(i);
            }
        }
    }
    //Info << designSpaceCells << endl;
