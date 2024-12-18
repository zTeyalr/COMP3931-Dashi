INDIVIDUAL PROJECT FOR COMP3931
Program developed in VSCode (Windows)

There is a link to Google Drive for downloading Sample Data and Preprocessed_data.

Sample data zip folder contains the ECG recordings from PhysioNet DB.
Preprocessed_data, is the already preprocessed data using the "preprocess" step.
Please unzip them both before running the program.
If needed to try preprocess without the use of the existing one, please delete the existing preprocess_data folder as they would overlap.

ecg_12lead_model.pth is the trained model before the new changes in the preprocess phase that gave the results stated in the paper (probably data leaking)

Install the requirements by running:
pip install -r requirements.txt

To run the program using the CMD:
Preprocessing: python main.py preprocess
Training: python main.py train
-- num_worker X is allowed to fasten the training process only (adviced for 32gb+ RAM)

NOTE: After some changes on the pre-processing phase, the training phase had some problems, hence it is not possible at the moment to run the train bit.