This folder contains all files necessary to run the spike transformer. 

To install Python dependencies:
pip install -r requirements.txt

The model folder contains subfolders where all information regarding a model is stored.
This can include weights, configuration files, evaluations predictions, everything. 

If you already have a model's weights file(xxx.ckpt), you can skip directly to the 1_calculate_predictions.ipynb.
To see how to train a model look into 0_train_model.ipynb

To caluculate predictions look into 1_calculate_predictions.ipynb
To caluculate ROC curve look into 2_calculate_AUC.ipynb