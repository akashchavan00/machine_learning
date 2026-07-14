Logistic regression is a supervised machine learning algorithm used for classification 
problems (typically binary classification). Despite having 'regression' in its name, it's 
actually a classification algorithm — it predicts the probability that a given input 
belongs to a particular class.

It works by taking a linear combination of input features (like linear regression does), 
and then passing that result through a sigmoid (logistic) function, which squashes the 
output into a range between 0 and 1. This output represents the probability of the positive 
class. We then apply a threshold (commonly 0.5) to convert that probability into a class 
label.

The math :
Linear part: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
Sigmoid function: σ(z) = 1 / (1 + e^(-z))
Output is a probability between 0 and 1

