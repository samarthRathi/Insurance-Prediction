
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

## Findings Report

<!-- Below you should report all of your findings in each section. You can fill this out as the project progresses. -->


### Research objectives
<!-- What questions you are trying to answer? -->
1. **Assess Predictive Accuracy:**
* Using standard regression models, assess the precision of health insurance charge estimates.
* Examine how  attributes affect premium estimation accuracy.
2. **Analyse the effectiveness of CNNs:**
* Analyse the predictive power of an Convolution Neural Network (CNN) model in predicting demographic variables (such as age and gender).
* To improve effectiveness, investigate fine-tuning CNN model.
3. **Image Processing Integration:**
* Look into incorporating image processing techniques into the system design.
* Investigate advanced image processing techniques for obtaining user attributes from images.

4. **Integrate Regression and CNN Models:**
* Investigate the relationship between classical regression models and Convolutional Neural Networks (CNNs) for predicting health insurance rates.
* Investigate the idea of using CNNs to extract demographic variables (such as age and gender) from image.
* Examine the effect of image processing on premium estimating accuracy.
* Combine demographic data from CNNs with user-provided information to improve the overall automation and precision of premium forecasts.



## Datasets
Dataset 1 :  [Health Insurance Premium dataset](https://www.kaggle.com/datasets/simranjain17/insurance/data)
* License : [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

#### Dataset description
<!-- Briefly describe your task and dataset -->
In regards tot his dataset we will predict the insurance premium i.e., charges based on other independent features age, sex, bmi, children, smoker, and region.
* We will clean and visualise the dataset.
* Next, will proceed towards Clustering to see what features we can use to make our labels and then visualise that as well.
* After that, we will proceed towards the machine learning - modelling part.

The aim is to train and test few models like Linear Regression, Descision tree, Random Forest etc., by creating pipeline to maintain ease of code making it modular and then
compare the performance of each thorugh Mean Squared Error and R2-Score, and select the best one in the end. Also, we plan on using hyperpaprameter tuning through GridSearchCV so that we get best parameters for out models.

* **The data card for this dataset is**:
1. **age**: It has age ranging from 19 to 82. It is Numerical data type denoting the age of the individuals.
2. **sex**: A categorical variable with two unique values - Male and Female defining the gender of the individuals.
3. **bmi**: The Body Mass Index being numerical continuous variable. Based on the height and weight of an individual the body fat is calculated giving the BMI.
4. **children**: A numerical variable with unique values from 0 to 5. This shows how many childrens does an individual have.
5. **smoker**: A categorical variable with 2 unique values telling if a person is a smoker or not through 'yes' or 'no'.
6. **region**: Categorical variable with 4 unique values as 'southwest', 'southeast', 'northwest', 'northeast'.
7. **charges**: A numerical variable showing the Health Insurance Premium amount charged as premium coverage.

**Independent Features**: Age, Sex(Gender), BMI, Children, Smoker and Region.

**Target Variable** : Charges

### Dataset example
<!-- Add a couple of example instances and the dataset format -->
| index | age  | sex    | bmi   | children | smoker | region     | charges    |
|-------|------|--------|-------|----------|--------|------------|------------|
| 0     | 19.0 | female | 27.9  | 0.0      | yes    | southwest  | 16884.92400|
| 1     | 18.0 | male   | 33.77 | 1.0      | no     | southeast  | 1725.55230 |
| 2     | 28.0 | male   | 33.0  | 3.0      | no     | southeast  | 4449.46200 |
| 3     | 33.0 | male   | 22.705| 0.0      | no     | northwest  | 21984.47061|
| 4     | 32.0 | male   | 28.88 | 0.0      | no     | northwest  | 3866.85520 |

Dataset 2 (zip file):  [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new)
**Note**: anyone with the Heriot-Watt Email Id can access this extracted dataset folder: [UTKFace](https://heriotwatt-my.sharepoint.com/:f:/g/personal/sr2077_hw_ac_uk/El3TPkVlWohIhPV6F0jp08sBq65XPyJvcRl0jwMcKFngCQ)
* License : [Data files © Original Authors](https://susanqq.github.io/UTKFace/)

#### Dataset description
Using this dataset, we'll apply CNN to predict age and gender from a person's facial image. We incorporate these demographics from a new image into our regression model after getting them from a new image. Details such as BMI, number of children, smoking status, and region are gathered from user input. Based on these combined variables, the regression model estimates the insurance premium.

* **The data card for this dataset is**:
1. **age**: It has age ranging from 19 to 82. It is Numerical data type denoting the age of the individuals.
2. **gender**: A categorical variable with two unique values - Male and Female defining the gender of the individuals.
3. **ImageName**: image file name.

### Dataset example
| Age | Gender | ImageName                                   |
|-----|--------|--------------------------------------------|
| 100 | 0      | 100_0_0_20170112213500903.jpg.chip.jpg     |
| 100 | 0      | 100_0_0_20170112215240346.jpg.chip.jpg     |
| 100 | 1      | 100_1_0_20170110183726390.jpg.chip.jpg     |
| 100 | 1      | 100_1_0_20170112213001988.jpg.chip.jpg     |
| 100 | 1      | 100_1_0_20170112213303693.jpg.chip.jpg     |

Here, the file name format consists of age, gender, ethnicity (not used for our project).
For eg., 100_0_0_20170112213500903.jpg ---> Age:100, Gender: Male, image Name: 20170112213500903.jpg.chip.jpg


## Dataset exploration
<!-- What is the size of the dataset? -->
* **Dataset 1** has 2038 Rows and 7 Columns (as mentioned in [data card](README.md#dataset-description) above)
* **Dataset 2** has 23708 rows in total with 3 columns ([as seen above](README.md#dataset-example-1)).

<!-- Train,validation,splits? -->
<!-- Summary statistics of your dataset -->
## [Data Exploration](./notebooks/code.ipynb#Data-Exploration)

Here, we tried to understand the data. Hence we used some inbuilt functions to explore it. Such as:
* **.shape**, which shows the dimensions of the dataset as a tuple. In our case, output is (2038, 7) as mentioned above .
* **.columns**, which displays the names of the features in an oredrly fashion. These have been mentioned above in the [data card](README.md#dataset-description).
* **.dtypes**, which outputs the data types of each feature adjacent to the feature name.
* **.unique()**, this function displays the unique values in each feature. We displayed them in a readable way.
* **value_counts()**, this function displays the count of each unique values in each feature. We displayed them in a readable format.
* **describe()**, this function displays the statistical characterstics. It is shown below:

| Statistic | Age | BMI | Children | Charges |
|-----------|-----|-----|----------|---------|
| Count     | 2038| 2038| 2038     | 2038    |
| Mean      | 43.81| 32.05| 1.55    | 22870.50|
| Std       | 17.72| 8.18 | 1.53    | 20415.32|
| Min       | 18   | 15.96| 0       | 1121.87 |
| 25%       | 29   | 26.22| 0       | 6552.58 |
| 50%       | 43   | 31.13| 1       | 13399.57|
| 75%       | 56   | 36.96| 3       | 38448.10|
| Max       | 88   | 53.13| 5       | 63770.43|
(Stratos Idreos, Papaemmanouil, & Chaudhuri, 2015)

The above is the statistics table of **Health Insurance Premium Dataset**. Here, we can see that various statistics have been analysed, such as:
1. Count: Showing the number of data points in ecah feature, which is same for all as we have same number of rows in all features.
2. Mean: It indicates the average of the values in that respective feature. Might be useful for the BMI feature.
3. Std: It is the standard Deviation of the values. Might be useful for the Age feature.
4. Min and Max: Shows the minimum and maximum values in each feature. Helps in normalization of the features.
5. 25%, 50%, 75%: These are 25, 50 and 75 percentile values in each feature. Helps in outlier detecton. (Zuur, Ieno, & Elphick, 2010)

## [Data Wrangling](./notebooks/code.ipynb#Data-Wrangling)
1. For dataset 1, we have chnaged the intial sex column to gender and also mapped Male as 0 and female as 1, to match with the CNN part of the code. This would help in smooth integration of both the parts. Furthermore, we have mapped smoker column Yes to 1 and No to 0 and similarly for region --->'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4.
This overall mapping of gender, smoker and region is done to convert the categorical variables to numerical variables for easy processing and model training such that we get the correct prediction of premium charges.

2. Next, we check for null values in the dataset, but upon exploration it is found that the dataset has no null values for any of the 7 columns. To keep the uniformity of data we convert age, children, smoker to integer values, and bmi and charges to two decimal values.

3. After this we create a new column called adult-group, where we categorise age based on three age bins, that are:
* Age 18 to 32 : young
* Age 32 to 48 : middle-aged
* Age 48 to 88 : old

4. Added another column titled 'bmi_weight_label'. This column tells us that if a person is Underweight, Normal Weight, Overweight or Obese. The calculation used for the creation of this column, as mentioned in NHS inform (n.d.), is:
* Underweight: BMI is less than 18.5
* Normal weight: BMI is 18.5 to 24.9
* Overweight: BMI is 25 to 29.9
* Obese: BMI is 30 or more

5. Creating a column charge_average, which tells is created by finding the average value of the "charges" where the "bmi_weight_label" is "Obese". After this it labels them as Above Avg or below Avg based on where the result of the mean for the charges of that particluar row falls.

6. One more new column : Stress Level, is added using the calculation ---> BMI + Children / Age. This is applies to the dataset and in the end, the stress level for each individual is calculated.


**After Data Wrangling the dataset looks like this:**

| age | gender | bmi   | children | smoker | region | charges  | adult_group  | bmi_weight_label | charge_average | stress      |
|-----|--------|-------|----------|--------|--------|----------|--------------|------------------|----------------|-------------|
| 19  | 1      | 27.90 | 0        | 1      | 1      | 16884.92 | young        | Overweight       | Below Avg       | 0.000000    |
| 18  | 0      | 33.77 | 1        | 0      | 2      | 1725.55  | young        | Obese            | Below Avg       | 0.533017    |
| 28  | 0      | 33.00 | 3        | 0      | 2      | 4449.46  | young        | Obese            | Below Avg       | 2.545455    |
| 33  | 0      | 22.70 | 0        | 0      | 3      | 21984.47 | middle-aged  | Normal Weight    | Below Avg       | 0.000000    |
| 32  | 0      | 28.88 | 0        | 0      | 3      | 3866.86  | middle-aged  | Overweight       | Below Avg       | 0.000000    |

<!-- Visualisations of your dataset -->
## Data Visualisation
### Health Insurance Dataset
* **Scatter Plots** : Higher charges are associated with Higher BMI levels, according to both scatter plots. One [scatter plot](./Results/Visualization/ScatterPlot_BMIvsCharges_AdultGroup.png) displays the distribution of adult groups, while the [other](./Results/Visualization/ScatterPlot_BMIvsCharges_Stress.png) depicts the distribution of stress levels.

* **[The Count Plot](./Results/Visualization/CountPlot_BMIGroup_Smoker.png)** indicates that the majority of individuals fall into the ‘Overweight’ category, with non-smokers being the most prevalent in each BMI weight group.

* The **Regression line** indicates that insurance charges tend to increase with age, showing a positive correlation.

![Regression Plot](./Results/Visualization/RegressionPlot.png)

* The **[Bar Plot](./Results/Visualization/Barplot%20smoker%20vs%20non-smokers.png)** illustrates health insurance costs for smokers and non-smokers, considering the number of children. **Main insights:** Smokers face higher charges, and having more children is associated with increased costs, indicating a significant difference.

* The box plots show the median, quartiles, and outliers for the [independent features](README.md#dataset-description). Some factors have more variation and influence than others. For example, smoker and charges have a lot of outliers, indicating that smoking and high charges are not common among the customers.
![Box Plot](./Results/Visualization/Box%20plot%20for%20age%2Cgender%2Cbmi%2Csmoker%2Cchildren%20and%20charges.png)

* **More analysis using Box Plot :** It can be seen that the median insurance charge is highest for the obese group and lowest for the underweight group. Furthermore, **[Box Plot](./Results/Visualization/BoxPlot_BMIGroup_Charges_Stress.png)** indicates that the obese group has the highest stress metric, while the underweight group has the lowest.

 **![Box Plot](./Results/Visualization/BoxPlot_BMIGroup_Charges.png)**


* These **[Distribution plots](./Results/Visualization/Distribution%20Plots)** reveals how Age, Gender, BMI, Children, Smoker, Region, Charges, and Stress are spread out, helping identify their shape, center, spread, and any outliers or unusual values.

* The **stacked bar** plot visually reveals the link between smoking habits and insurance costs. The **[Count plot](./Results/Visualization/Count%20plot%20for%20age%20groups(smokers%20vs%20non-smokers).png)** illustrates age-wise smoker distribution, emphasizing more young and old smokers. Additionally, the associated **[violin plot](./Results/Visualization/ViolinPlot_Charges_Children_for_Smoker.png)** highlights higher charges for smokers across all categories of the number of children.

![SChart](./Results/Visualization/Stacked%20bar%20smokers%20vs%20non%20smokers%20charges.png)

### **CNN UTK Face Data Visualisation**
While Visualising the data and images for UTKFace dataset, we have used few plots to make the understanding of data more refined. Here below are the few pointers and observations done:

* **[Histogram Plot and Bar Plot](./Results/Visualization/Histogram%20and%20Bar%20plot%20for%20age%20and%20gender.png)** shows a peak in the late 20s and early 30s, suggesting that the population's major age range is approximately 35 years old, whilst the bar graph highlights the gender distribution, providing insight into the male-female ratio within that age category.

* **[Violin Plot](./Results/Visualization/Violin%20plot%20for%20age%20vs%20gender.png)** shows the age distribution of males and females. The plot shows that the males have a wider range of ages than the females, and that the median age for males is higher than the median age for females.

* **Image Clustering** is used to transform a gradient of colors into a discrete partition of colors. This helps to reduce the complexity and size of the image, as well as to highlight the different regions of colors.
![Kmeans](./Results/Visualization/Clustered%20image.png)

* **[Intensity Heatmap](./Results/Visualization/Heatmap%20image.png)** shows the effect of applying different heatmaps to a grayscale image of a person’s face. It is used here to highlight patterns, trends, and outliers in the data. The “hot” heatmap emphasizes the high-intensity regions of the image, such as the eyes, nose, and mouth. The “cool” heatmap emphasizes the low-intensity regions of the image, such as the hair, forehead, and cheeks.
* We have splitted the image intensity into four parts red, green, blue, and grey and visualised them. Here the Grey is they average of all the other intensities.

![this output](./Results/Visualization/Intensity_split.png)

<!-- Analysis of your dataset -->

### Clustering
The **[Elbow Method](./Results/Visualization/Distortion_Score_Elbow.png)** helped us overcome one of the biggest obstacles in clustering: deciding how many clusters to create.

#### Experimental design
<!-- Describe your experimental design and choices for the week. -->
**K-Means Clustering:**
Initially, the dataset was preprocessed to extract variables such as "age," "bmi," "children," and "smoker" that were pertinent for clustering. The KElbowVisualizer was used to calculate the ideal number of clusters, and K-Means clustering was then used using the selected cluster count. To see the clusters and their centers, scatter plots were created. (Na, Xumin, & Yong, 2010)

**Hierarchical Clustering:**
Then Hierarchical clustering was conducted using Ward's method with different linkage strategies, including single, complete, average, and ward. Dendrograms were generated to illustrate the hierarchical relationships between data points.

* **Agglomerative Clustering:**
Agglomerative clustering was performed on selected features, namely 'age', 'bmi', 'children', and 'smoker'. The resulting clusters were visualized with a scatter plot, and a dendrogram was generated to display the hierarchical relationships.

* **DBSCAN Clustering:**
DBSCAN clustering was applied to the scaled data with specified parameters for epsilon (eps) and minimum samples (min_samples).

* **KModes Clustering** was also performed using Huang initialization method. As suggested above in Elbow Method, we decided to divided the data into 5 Clusters. Then, the Algorithm was run 10 times to find the most stable solution.

#### Results
<!-- Tables showing the results of your experiments -->
* This is the graph showing the spliting of data into 5 clusters based on the KMeans Clustering Algorithm.
  **![KMeans Clustering](./Results/Clustering/K_Means/kme_clu.png)**

*  **Agglomerative Clustering** has grouped the data into 5 clusters. The dendrogram visualizes the hierarchical nature of these clusters.
  **![Agglomerative Clustering](./Results/Clustering/Agglomerative/agg_clu_and_ddg.png)**

* **[Hierarchical Clustering](./Results/Clustering/Heirarchial/hhr_clu_ddg.png)** algorithm has grouped the data into two clusters, as visualized in the dendrogram.

* **[DBSCAN Clustering](./Results/Clustering/DBSCAN/dbs_clu.png)** algorithm has grouped the data into clusters based on BMI and charges, as visualized in the scatter plot.

* **[KModes Clustering](./Results/Clustering/K_Modes/kmo_clu.png)** : This image shows the Data splitting into 5 Clusters based on KModes Clustering Algorithm.

#### Discussion
<!-- A brief discussion on the results of your experiment -->

* KMeans Clustering identifies groups based on 'Age' and 'BMI'. Each cluster represents population with similar characteristics. This enables us to find patterns and perform segmentation within the dataset. (Jin & Han, 2011)

* Based on the above results between KMeans and KModes, we can consider that KMeans is much better than KModes. KModes generates mixed clusters, this is because KModes doesn't perform good for Numerical Data (which is our case here). And hence, KMeans provides much more promising results than KModes.

* Hirarchichal Clustering only creates two clusters, K-Means and Agglomerative are able to get 5 clusters, which is more suitable for our dataset.