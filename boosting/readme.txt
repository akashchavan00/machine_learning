*What is Boosting?
->Boosting is an ensemble technique where:
Multiple weak learners are trained one after another, and each new learner 
focuses on correcting the mistakes made by the previous learners.
The final prediction is a combination of all these learners.

###########################

*What is a Weak Learner?
->A weak learner is a model that performs only slightly better than random guessing.
In boosting, the weak learner is usually a small decision tree (often called a decision 
stump if it has only one split).
For example:
Imagine a classifier that predicts correctly 60% of the time.
That's not great, but it's better than random (50% for binary classification), so it 
qualifies as a weak learner.
Boosting combines many such weak learners to create a strong learner.

############################

Real-Life Example
Imagine you are learning mathematics.
Test 1
You score:
60/100
Your teacher checks your paper and highlights your mistakes.
You study only those weak areas.

#############################

Why Is It Called "Boosting"?
Each new model boosts the performance of the overall system by fixing errors 
made by earlier models.
Instead of replacing previous models, it builds on them.

#############################

*Key Characteristics of Boosting
1. Models are trained sequentially, not independently.
2. Each new model learns from the mistakes of the previous ones.
3. Weak learners combine to form a strong learner.
4. Final prediction is based on the combined output of all learners.

###################################

Q: What is Boosting?

Answer:
Boosting is an ensemble learning technique in which multiple weak learners are trained 
sequentially. Each new learner focuses on correcting the mistakes made by the previous 
learners. By combining the predictions of all the learners, boosting creates a strong 
model that generally achieves higher accuracy than any individual weak learner.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Understanding the Parameters:
    DecisionTreeClassifier(max_depth=1)
Creates a Decision Stump (a tree with only one split).
AdaBoost typically uses these simple trees as weak learners.

    n_estimators=50
Builds 50 weak learners sequentially.
Each new learner focuses more on the samples that previous learners misclassified.
    learning_rate=1.0
Controls how much each weak learner contributes to the final prediction.
Smaller values (e.g., 0.1) make learning slower but can improve generalization.
    random_state=42
Ensures reproducible results.
Running the code multiple times with the same seed produces the same train/test split and model initialization.

What Happens Internally?
Train the first decision stump.
Find which training samples it misclassified.
Increase the weights of those misclassified samples.
Train the second stump, giving more importance to those harder examples.
Repeat this process for all 50 stumps.