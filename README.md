# Latana 


## 1. Build your own model 

We ran a 3 wave, 3 country brand tracking study for a B2C client in the technology sector. You can find the codebook including the survey questionnaire here. Using ‘brand_awareness_client’ as your dependent variable, please build a model that predicts the probability that a person with certain demographics e.g. female, 16-25 years, low education is aware of the client’s brand. You can download the relevant data set here.  As your deliverables for this section, please provide:


1. Your commented source code.  
    It is on the notebook "Kaue_Latana_nb.ipynb" inside this repository, in case you wish to run in your enviroment, there is also a requirements_latananb.txt 

2.A summary explaining your model to internal (non-engineering/scientific) stakeholders. Make sure your explanation includes:
- How you selected the predictors.
- Why you chose the specific class of model and how it’s appropriate for this specific product. 
- An evaluation of your model’s performance.


The problem stated to use certain demographics predict the probability of a person with some characteristics has knowledge about the brand, gender was not in the dataset and as gender it was cited on the problem and it is one major feature in any questionnaire, therefore my first understanding of the problem was:

Create a model with only the demographic features and use gender. 

Then my first goals were: 

- identify hidden/latent gender behaviour in order to create this feature. 
        how I would do this: 
        Using Latent Dirichlet Allocation [1], I could find a hidden binary feature that makes the other features have different distribution,  
        and I would use income ( as we live in a sexist world we could check the gender paying gap) to identify which is male or female in the  
        latent feature created.
      
- Create a simple model using only dem. features .

      How I would do this: 
      This is more straight forward, however for this step I would use a simpler model to have more interpretability, 
      therefore I would use logistic regression or a decision tree. 
      
- Use the other features to explain the difference between the demographic groups. 

      How I would do this: 
      I create a profile based on the other features to justify the different behaviour towards the brand,  
      First I would select only features which have some correlation with the dependant variable, and from that on  
      take those features and try to see how much they fluctuate from each demographic group from the previous point. 
      
      
 
 ### However 
 
 The problem was much simpler, and I just need to create a classification model for an imbalanced class with mostly categorical features. 
 
 So the steps were: 
 
 - General EDA 
    Simple data exploration of the demographic data. 
    
 - Missing data handling  
     Some data was merged making the missing disappear other data I used MICE [2], 
     MICE is a missing data imputer which fills those points comparing similar rows regarding the other features and then imputing the missing one  
     
 - Data transformation 
    I transformed all the categorical features in dummies (One hot encoded) and into numeric, to be adjusted to be use in any model 
   
     
 - Select features and create a baseline model 
     I ran a Random forest to select the most important features, and I selected all that had over 0.1% of importance for the dependant variable. 
     With that also I had a first glance 
     
 - Still too many features, Dimensional Reduction  
    I split the features I selected in the previous step in two, above 1% importance  and below, only 17 features were above and over 100 were below. 
    The first group I will use as it is, and the bigger group I created a component using Linear Discriminant Analysis [3], this means that I tried to transform      all the important information of the 100 features into a single one, and as all the features from  
    the bigger group were categorical LDA is preferred  instead of PCA.
    
 - Failed attempt 
   In order to correct the imbalance I try to produce synthetic  data to oversample using the SMOTE [4],  
   however the prediction got worse, so I used the regular data
     
 - Models 
   So the final data used was the best 17 features which had feature importance above 1% and the component create with the Linear Discriminant analysis. 
   I tried Logistic Regression, Random Forest and LightGBM.  As the data was imbalanced I had to pay attention to the Macro F1-score to see which one was the      better model. 
   
   And after a few runs the best model was the Random Forest, initially was overfitting due not having a sizable minimum size leaf, after this, still have some     overfit but not as in the beginning. As lightGBM is fast to run and the results were close to Random Forest, I decided to tune the hyperparameters, however 
   even tunning still lost for Random Forest which the performance can be seen below: 
   
   -*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-
RF train F1 Score ->  0.7268322450187327
-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-
RF test F1 Score ->  61.69
              precision    recall  f1-score   support

           0       0.84      0.93      0.88      3576
           1       0.49      0.27      0.35       866

    accuracy                           0.80      4442
   macro avg       0.67      0.60      0.62      4442
weighted avg       0.77      0.80      0.78      4442

     
     
     
     In all models the precision of the category 1 (if the brand is known) was low, and this model was the most balanced.
     
     
     
     
     

       
       
       
       
       
       
 ### 2. MRP
Please read through section 3.1 (Multilevel regression and poststratification) of this paper on forecasting with non representative polls. 
At Dalia we use MRP to build models that enable our clients to get highly accurate estimates of their brand KPIs on a nationally representative level while at the same time allowing them to drill down into niche target groups that are important for them. 

- What kind of challenges can you anticipate for building such a model?  
- 
  As we have a combination of 176,256 cells. We might have sparsity  in our model as we will not have enough members for each cell,  
  furthermore averaging the few members of a cell might build a sampling bias, that means that few respondents might "answer" for a whole group.

 - What limitations and capabilities would you think such a model would have? 
  
  Poststratifying showed inaccurate  in the election Biden x Trump 2020, as some republicans or groups of people were avoiding answering polls [5] , 
  which almost made a "quantum effect" those that answered/were seen had a different behaviour from those that did not.  
  There must first prove that there is no bias on the respondents of certain group first before extrapolating.  
  
  Each new feature/demographic selected grows the number of groups exponentially.  
  
  This model is great, but it would be even more useful using the odds ratio of each feature to see the impact in the final vote or the binary studied choice 
  Also for the politics and the model itself it would be good to study cells which have high swings/volatility, which this model does not capture well.
  
 
 
 
    [1] https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?TB_iframe=true&width=370.8&height=658.8
    [2] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/ 
    [3] https://link.springer.com/chapter/10.1007/978-0-387-78189-1_8
    [4] https://arxiv.org/pdf/1106.1813.pdf
    [5] https://www.nytimes.com/2020/07/29/upshot/polls-political-party-republicans.html



      
      
      
      
      
      
      
      
      
      
      
      
     
