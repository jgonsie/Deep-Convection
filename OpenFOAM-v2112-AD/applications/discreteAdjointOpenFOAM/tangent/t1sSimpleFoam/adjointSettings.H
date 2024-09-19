    dictionary simpleDict = mesh.solutionDict().subDict("SIMPLE");

    autoPtr<wordList> costFunctionPatches;
    costFunctionPatches.reset(new wordList(0));

    if (simpleDict.found("costFunctionPatches"))
    {
        costFunctionPatches.reset
        (
            new wordList(simpleDict.lookup("costFunctionPatches"))
        );
    }else{
        Info << "Warning: Keyword costFunctionPachtes not found in fvSolution/SIMPLE" << endl;
        Info << "Example:\nSIMPLE{\n  costFunctionPatches (inlet outlet);\n}" << endl;
    }

    point searchPoint = simpleDict.lookupOrDefault<point>("derivPoint",point(0,0,0));
    bool seedAll = simpleDict.lookupOrDefault<bool>("seedAllCells",false);

    if(!seedAll){
        derivCellID = mesh.findCell(searchPoint);
        Pout << "Cell nearest to " << searchPoint << ": " << derivCellID << endl;
    }
