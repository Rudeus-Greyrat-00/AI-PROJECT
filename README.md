# AI-PROJECT

/////TODO here: write a brief description about this project

# HOW TO RUN THE PROJECT:

1 - Download the dataset (https://1drv.ms/u/s!AgIq60Fi24cTuK5itVdJabtgFFGcdw?e=DO9oUL), there are three folders (TC, ASD, TOTAL), you need all of them.

2 - Create a file "conf.py" inside the project folder (between all the other .py files) and add theese lines:
``` 
PATH_ASD = "[PATH TO "ASD" FOLDER]"

PATH_TC = "[PATH TO "TC" FOLDER]"

PATH_TOTAL = "[PATH TO "TOTAL" FOLDER]"

ANNOTATION_FILE = "[PATH TO "TOTAL" FOLDER]\\label.csv"
```

3 - Run the "main.py" file to test if everything work, hopefully it will do.

# DOCUMENTATION:
- The file models.py contains neural net models
- The file mri_datautils.py contains the custom dataset that work with .nii files (as well as all the function needed to read .nii files)
- The file training.py contains the function to train the net and check the accuracy
- The file dataset_utils.py contains some function to generate test/train datasets (using the dataset defined in mri_datautils.py)

Further documentation is available inside the code!
