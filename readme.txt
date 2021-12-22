# Neural Network, SVM and K-NN implementation using MNIST dataset

# If you are getting errors or not getting the output in PART 1 then try PART 2

## IMP: Please give the path till the dataset folder. The code takes the data from the train and test folder inside the path

> pip install -r requirements.txt

# -------------PART 1 a-------------
## Run the files with filepath as arguments
## for KNN run the following with 3 nearest neighbors algorithm = brute and weights = distance
> python KNN_multi.py 3 brute 2 distance

## for SVC run the following with C=1 gamma = 0.001 and kernel = rbf
> python SVM_multi.py 1 0.001 rbf

## for SVC run the following with C=1 gamma = 0.001 and kernel = rbf
> python SVM_multi.py 1 0.001 rbf

## for MLP run the following with report_mode = 0 (parameter to get non report results), solver = adam , activation = relu iterations = 100 alpha = 0.001
python NN_multi.py 0 adam relu 100 0.001

## OUTPUTS of all the experiments outcome is provided in output_combinations_files folder

# -------------PART 2-------------
# Steps to run the code... commands are tested in linux.. you can apply alternative commands for windows/MacOS
## Step 1 creating a virtual environment to run the code so that it does not conflicts with other instaled packages on the machine
> python3 -m venv my_env
## Step 2 if the above gives error then make sure your python version is 3.6 or above and install the venv package. If no error move to Step 3
	### for linux and MacOS
	> python3 -m pip install --user virtualenv
	### for windows
	> py -m pip install --user virtualenv

## Step 3 activate the environment
> source my_env/bin/activate
> pip install --upgrade pip

## Step 2 use requirements.txt file to install required packages
> pip install -r requirements.txt

After this you are good to use the python files and can run using the above commands specified

### once done with grading of the code you can deactivate the environment and delete it
> deactivate
> rm -r my_env
