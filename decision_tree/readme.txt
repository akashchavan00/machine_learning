A **decision tree** is a machine learning model that makes 
predictions by asking a series of yes/no (or true/false) 
questions, like a flowchart.

**How it works:**
1. Start at the top with all your data.
2. Ask a question about a feature (e.g., "Is age > 30?").
3. Split the data into two branches based on the answer.
4. Keep asking questions on each branch, splitting further.
5. Eventually reach a "leaf" — the final prediction/answer.

**Simple example:** Predicting if someone will buy a product:
```
Is income > $50k?
├── No  → Won't buy
└── Yes → Is age > 25?
          ├── No  → Won't buy
          └── Yes → Will buy
```

**Key ideas:**
- Each split tries to separate the data as cleanly as possible 
(using measures like *Gini impurity* or *entropy/information gain*).
- The tree keeps splitting until it reaches a stopping point 
(max depth, minimum samples, or perfectly pure groups).
- **Root node** = top of the tree (first question). **Leaf node** 
= final prediction.

**Pros:** Easy to understand and visualize, works for both 
classification and regression, no need to scale data.

**Cons:** Can easily overfit (memorize training data) if too deep —
 this is why methods like **Random Forest** (many trees combined)
  are often used instead.

Want me to show a visual diagram of how a decision tree splits data, 
or explain how it's actually built (Gini/entropy math)?