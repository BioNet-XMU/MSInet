## MSInet: A framework designed for unsupervised segmentation in Mass Spectrometry Imaging (MSI) data.

<img width="6805" height="5495" alt="New_Main_Figure drawio" src="https://github.com/user-attachments/assets/96c15b39-89c2-4cfd-bdc1-db3851fcb377" />

## Installation

## (A) Running on Local Machine (Command Line Interface)

Run MSInet locally on your computer:

1. Clone this repository and install all dependencies:

   git clone https://github.com/BioNet-XMU/MSInet.git

2. Navigate to the folder

   cd MSInet
   
3. Install all dependencies

   pip install -r requirements.txt

4. After installation, run:

   python main.py

5. You’ll be interactively asked to provide: 
   Enter path to your .npy data file:
   
   Enter image shape (height width channels):

   The path to the default data is "datasets/simulated.npy" and image shape is "70 70 3"

## (B) Running on Google Colab

1. !git clone https://github.com/BioNet-XMU/MSInet.git

2. %cd MSInet

3. !pip install -r requirements.txt

4. %run main.py

5. You’ll be interactively asked to provide: 
   Enter path to your .npy data file:
   
   Enter image shape (height width channels):

   The path to the default sample data is "/content/MSInet/datasets/simulated.npy" and image shape is "70 70 3"


