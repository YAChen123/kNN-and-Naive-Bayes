PYTHON = python3

# Output directory
OUTPUT_DIR = outputs

# Targets
all: $(OUTPUT_DIR) run

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

run:
	$(PYTHON) main.py -train inputs/kNN1_train.csv -test inputs/kNN1_test.csv -K 3 -v > $(OUTPUT_DIR)/knn1_K3.out
	$(PYTHON) main.py -train inputs/kNN2_train.csv -test inputs/kNN2_test.csv -K 3 -v > $(OUTPUT_DIR)/knn2_K3.out
	$(PYTHON) main.py -train inputs/kNN3_train.csv -test inputs/kNN3_test.csv -K 3 -v > $(OUTPUT_DIR)/knn3_K3.out
	$(PYTHON) main.py -train inputs/kNN3_train.csv -test inputs/kNN3_test.csv -K 5 -v > $(OUTPUT_DIR)/knn3_K5.out
	$(PYTHON) main.py -train inputs/kNN3_train.csv -test inputs/kNN3_test.csv -K 7 -v > $(OUTPUT_DIR)/knn3_K7.out

	$(PYTHON) main.py -train inputs/ex1_train.csv -test inputs/ex1_test.csv -K 0 > $(OUTPUT_DIR)/nb1.0.out
	$(PYTHON) main.py -train inputs/ex1_train.csv -test inputs/ex1_test.csv -K 0 -v > $(OUTPUT_DIR)/nb1.0.v.out

	$(PYTHON) main.py -train inputs/ex1_train.csv -test inputs/ex1_test.csv -K 0 -C 1 > $(OUTPUT_DIR)/nb1.1.out
	$(PYTHON) main.py -train inputs/ex1_train.csv -test inputs/ex1_test.csv -K 0 -v -C 1 > $(OUTPUT_DIR)/nb1.1.v.out
	
	$(PYTHON) main.py -train inputs/ex2_train.csv -test inputs/ex2_test.csv -K 0 > $(OUTPUT_DIR)/nb2.0.out
	$(PYTHON) main.py -train inputs/ex2_train.csv -test inputs/ex2_test.csv -K 0 -v > $(OUTPUT_DIR)/nb2.0.v.out
	
	$(PYTHON) main.py -train inputs/ex2_train.csv -test inputs/ex2_test.csv -K 0 -C 1 > $(OUTPUT_DIR)/nb2.1.out
	$(PYTHON) main.py -train inputs/ex2_train.csv -test inputs/ex2_test.csv -K 0 -v -C 1 > $(OUTPUT_DIR)/nb2.1.v.out
	
	$(PYTHON) main.py -train inputs/ex2_train.csv -test inputs/ex2_test.csv -K 0 -C 2 > $(OUTPUT_DIR)/nb2.2.out
	$(PYTHON) main.py -train inputs/ex2_train.csv -test inputs/ex2_test.csv -K 0 -v -C 2 > $(OUTPUT_DIR)/nb2.2.v.out
	
	$(PYTHON) main.py -train inputs/Kmeans1.csv -d e2 "0,0" "200,200" "500,500" > $(OUTPUT_DIR)/kmeans1.e2.out
	$(PYTHON) main.py -train inputs/Kmeans1.csv -d manh "0,0" "200,200" "500,500" > $(OUTPUT_DIR)/kmeans1.manh.out

	$(PYTHON) main.py -train inputs/Kmeans2.csv -d e2 "0,0,0" "200,200,200" "500,500,500" > $(OUTPUT_DIR)/kmeans2.e2.out
	$(PYTHON) main.py -train inputs/Kmeans2.csv -d manh "0,0,0" "200,200,200" "500,500,500" > $(OUTPUT_DIR)/kmeans2.manh.out


clean:
	rm -f *.pyc
	rm -rf outputs
