graph:
	adjointSimpleCountFoam -case pitzDaily | grep "m4_" > graph_data
	m4 -P graph_data graph.dot.m4 > graph.dot
	dot graph.dot -T pdf > graph.pdf
	dot graph.dot -T png > graph.png

clean:
	rm -f graph_data graph.dot *.png *.pdf
