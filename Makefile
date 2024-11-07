 build_mnist:
	gcc -o mnist_mlp mnist_mlp.c -lm

build_simple:
	gcc -o mlp_simple mlp_simple.c -lm

run_mnist: 
	./mnist_mlp

run_simple: 
	./mlp_simple

build_run_mnist: build_mnist run_mnist 

build_run_simple: build_simple run_simple 

data_download:
	bash data_download.sh