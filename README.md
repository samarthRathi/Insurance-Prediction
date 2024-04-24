
## **Project Title : Predicting Health Insurance Premium using Regression Model with Age and Gender Detection from Image using CNN**

## Group Members

1. Aman Valera @ amanvalera7
2. Sai Ruthik Katta @ SaiRuthik
3. Samarth Rathi @ SamarthRathiH00422752
4. Suhashini Selvaraj @ Suhashini-Selvaraj

## Initial Project Proposal

* The Aim of this project is to develop a user friendly system which predicts how much Health Insurance a person might have to pay. It comprises of two parts:

* **Estimating Health Insurance:** This is the initial objective on which we will be focusing. The input data is a person's basic information such as Age, Gender, Residing location etc.,. Using a suitable Algorithm, our Machine Learning model will predict the estimate a person has to pay as an Health Insurance using the required Features. We aim to predict this a bit more accurately which helps the people to get right insurance deals.

* **Predicting who you are:** The second segment of this project focuses on predicting the basic data of a person based on his/her photo. We use Convolution Neural Networks to determine the Age and Gender based on the provided image of the person.

* Combining these two aspects would result in an easy-to-use platform that tells how much insurance might cost and help insurance companies understand you in a better way. We try to simplify the health insurance process and make it more personalized.

### Research objectives
How accurately can we predict health insurance charges based on attributes using traditional regression modeling techniques?

What is the effectiveness of an Convolution Neural Network (CNN) model in predicting demographic attributes such as age and gender?

How can image processing techniques be integrated into the system to extract attributes from images for accurate premium estimation?

### Milestones

* **Data collection and preprocessing:** Fix missing values and organise the format of the data to create the two datasets prepared for analysis.
* **Ethical Consideration of Dataset:** Data used for this project is public and ethically cleared to use. The license of the datasets has been updated in the [dataset section](README.md#datasets).
* **Clustering Analysis:** Using clustering algorithms, divide policyholders into groups according to common features and demographic characteristics.
* **Data Visualization:** Utilize visualizations to gain insights into the dataset's patterns and relationships among variables.
* **Development of Regression Models:** Create a prediction model to calculate insurance costs according to policyholder data.
* **Fine-Tuning and Optimisation:** Enhance the system's overall performance, resilience, and accuracy.
* **CNN Model Training:** Using the demographic dataset, train an CNN model to predict demographic features.
* **System Integration:** To build an integrated health insurance prediction system, integrate the image processing skills, and predictive models.
* **Testing and Validation:** In order to ensure precise premium calculation, test the system's functionality using a variety of picture inputs and real-world circumstances.

# Prerequisites
Note: These following pre-requisites is based on this respository and project.
  * For Regression (Health Insurance Prediction) any general setup like [google colab](https://colab.research.google.com/) can be used, but for CNN to avoid epoch processing time a local setup of anaconda might be required. The following steps will guide for the same.

* **Python Version :**
1. For Insurance Premium Prediction: 3.10.12
2. For CNN: 3.7.16 --->To avoid any complications during the installation of TensorFlow with local GPU support.
* **Anaconda Version: 22.9.0**
 ---> To install Ananconda, clcik on this link: [Download Ananconda](https://www.anaconda.com/download)
* **Note** : Create a local environmnet in anaconda and then install the pre-requisite libraries below.
* Packages and versions: [Python Installed Packages](./documentation/requirements.txt)
* To use local GPU for CNN follow this link: [Using Local GPU with Tensorflow Guide](https://www.tensorflow.org/install/pip)

**Important Point**
* To run the CNN model directly inside the [regression code](./notebook/code.ipynb), a model file is upload on [OneDrive](https://heriotwatt-my.sharepoint.com/:u:/g/personal/sr2077_hw_ac_uk/EW4-ohf0bHpBqu9zU7JSGaABOZK_r5ZXZnNFPrBUMGfhbg?e=hpCT6Y) and a zip file is also uploaded inside the project directory : [Cnn_Model.zip](./notebook/Cnn_Model.zip). You can download the file from OneDrive (if incase the zip file causes some issue) and then save it in the same directory as the 'code.ipynb' file.
Anyone with the Heriot-Watt Email ID can access the CNN Model.

### DISCLAIMER
**NOTE : [ChatGpt-3.5](https://chat.openai.com/) was used to develop undersatnding for the project and few algorithms but 'NO' code or data was copied from ChatGpt.**