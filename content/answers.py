"""Verbatim student writing from Exam1.ipynb. Do not rewrite."""

# Empty exam cells 6b and 7a are omitted on purpose.

DATA_SIZE = """1. There are 2000 patients in the dataset.

2. There are 16 values in the dataset.

3. The column cholesterol has 170 missing values, vessels_colored has 79 values missing, and thal has 60 values missing (309 missing values total)."""

LAZY_BASELINE = """1. This model would get an accuracy of 0.728 because if the share of patients with no heart disease is 0.728, than the model would predict that share correctly since it is simply saying "no heart disease" for every patient.

2. That model would be useless for the clinic as the goal of implementing our machine learning model is to accurately predict what patients should be sent to the hospital for an angiogram. Since the model is saying "no heart disease" for every patient, it directly fails at the previously defined goal for the clinic."""

DROPPED_COLUMNS = """1. The two columns that should not be present in our model's features are "followup_booked" and "patient_id".

2. The variable "followup_booked" needs to be dropped becasue the presence of this column would cause a data leakage within our model, as the director mentioned in the story that "When the hospital sends back a result that shows heart disease, our system books a follow up with a cardiologist automatically. That is in the file too." To ensure our model does not overfit, "followup_booked" should not be a feature. The "patient_id" feature should be dropped as well since it is not relevant in providing our model with any data regarding their heart. It is just a filler column that would be useless for our model in terms of predicting heart diseases, also if they are a brand new patient they may not even be assigned a patient id on the day they walk in, making it beneficial for us to drop before moving forward."""

FOLLOWUP_LEAKAGE = """1. The share of patients with a follow up that have heart diseases is about 0.83, while the share without a follow up is only about 0.03.

2. A brand new patient who walks into the clinic would have a "followup_booked" value of 0 (representing that they do not have a follow up booked). In this case, our model will falsely misrepresent them as they will presume that the new patient does not have heart disease when they may infact have it. This is crucial as the tradeoff in this case is putting a human life at risk, meaning we have no margin for error whatsoever when it comes to our model as it is costly when it is wrong."""

CHEST_PAIN_BY_SEX = """1. For men, the kind of chest pain with the highest share of heart disease is asymptomatic.

2. For women, the kind of chest pain with the highest share of heart disease is typical angina, which is not the same as men.

3. This would be a problem for a model trained on this data because it would underrepresent the amount of women who go to the director's clinic. If trained on this data, the model will recognize the patterns found within male patients and apply those to female patients. However, we can see that what is applicable to men is not always relevant to women."""

LOGISTIC_CV = """The mean ROC AUC is 0.922, the standard deviation (or spread) is 0.006."""

PIPELINE_IMPUTATION = """1. I filled the missing values in step 4a. (cell 7), specifically in the lines ("fill", SimpleImputer(strategy="most_frequent")) and
("fill", SimpleImputer(strategy="median")) outlined in our pipeline.

2. If a patient is missing their cholesterol than the median of the patients in the 1st, 2nd, 3rd, and 4th (which is the training fold in this case since the 5th fold is the testing fold) is used to fill it in.

3. That is the right choice because it prevents data leakage by using only values from the training fold to compute the median. This gives us an idea of how our model will perform under new and unseen data, which is the goal so that we can apply our model on new patient data after its finetuned."""

FOREST_OVERFIT = """1. The training score is 1.000, this means that the tree correcly identified all the patients that had heart disease. However it is also a warning sign that overfitting was present when we use 1 as our min_samples_leaf value and no param_model__max_depth.

2. The cross validated score is lower because of the overfitting that occured when the model was trained since it was scored on the same values which it was trained on.

3. The gap between the training score and the cross validated score for the logistic regression model is smaller than the one for the forest because the forest model overfit as a result of the min_samples_leaf value being 1 and param_model__max_depth being none.

4. The model chose {'model__max_depth': None, 'model__min_samples_leaf': 5} for the settings of the forest, min_samples_leaf determines the amount of training samples used for each leaf. In this case, the best forest model had a min_samples_leaf of 5."""

MODEL_CHOICE = """1. The best ROC AUC for the logistic regression model is 0.923 and the best ROC AUC for the random forest is 0.940.

2. The difference between the two rocs is 0.017, this is bigger than the difference between the respective spreads which is 0.001.

3. I choose to use the random forest model because it had the higher ROC AUC score compared to the logistic regression model. I am using the ROC AUC as the metric to judge which model to use moving forward because the difference in the spread across both models is visibly less than the difference of the averages, thus meaning that the difference between them is not due to chance but determined by who has the higher ROC AUC score. This is the random forest model in this scenario."""

AUC_PLAIN = """If you pick one patient with heart disease and one without, at random the model gives the patient with heart disease the higher probability of heart disease 94% of the time."""

CONFUSION_AT_HALF = """1023 patients that did not have heart disease were sent home.

70 patients that did not have heart disease were sent for an angiogram.

105 patients that did have heart diseases were sent home.

302 patients that did have heart diseases were sent for an angiogram."""

THRESHOLD_CHOICE = """1. I chose the threshold 0.2, since we care more about having a high recall given the fact that a false negative could cost someone their life if misdiagnosed and 0.2 meets that requirement while having the best balance of precision and recall.

2. We can not miss more than more than 1 out of every 10 patients with heart disease.

3. This costs more patients who do not have heart disease to go in for an unecessary follow up appoitment. Compared to a threshold of 0.5, 243 more patients were sent for an angiogram but 84 fewer patients with heart disease got sent home."""

RECALL_BY_SEX = """1. The recall for the women is 0.885 and the recall for men is 0.96.

2. In plain words, this means that women have less of a chance than men to be correctly identified for heart disease.

3. The model will do worse for women because the data it is trained on is inherently biased, as outlined in the story the reasearch used came from clinical trials dominated by men. This leads to the underepresentation of women in our model, which is something to highlight as one third of the clinic patients are female."""

TEST_VS_CV = """1. Yes the test numbers are close to the cross validated numbers, with most of them being slightly lower than the original cross validated numbers. For example, the recall for men originally was 0.96 on the training set while it was 0.949 on the testing set. Another instance includes, the total recall of both the training and testing set at a threshold of 0.2, with the testing set having a value of 0.926 while the training set had a recall of 0.948.

2. There are 19 women with heart disease in the test set, if the model caught one more of them the recall would go up.

3. No, I do not trust the test recall for women on its own because the data in which the model was trained on was biased as it underepresented women in comparison to men."""

DIRECTOR_LETTER = """The model takes the patient data that your clinic is already collecting and turns that into validated predictions on whether or not someone may have a heart disease thus, streamlining the angiogram process by weeding out patients who are less likely to be diagnosed. The model would send 196 patients in for an angiogram, and miss 10 patients out of a group of 500. It is also important to be more cautious if the patient is a woman when using this model because it is bias which leds it to underperform in comparison to male patients. For example, the model misses 2 out of 10 women with heart disease, while it misses only 1 out of 10 men with heart disease. """
