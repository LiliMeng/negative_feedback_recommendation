# negative_feedback_recommendation

Paper: Learning from Negative User Feedback and Measuring Responsiveness for Sequential Recommenders https://arxiv.org/pdf/2308.12256
<img width="711" alt="Screenshot 2024-12-04 at 3 47 42 PM" src="https://github.com/user-attachments/assets/238ae1d6-2d50-43ca-8969-3a53a98fd170">


### Cross-Entropy Loss and Negative-Valued Label Weights

The **cross-entropy loss** for a single instance can be expressed as:

$L_i = -\log(p(y_i | s_i))$

Where:
- $p(y_i | s_i)$ is the predicted probability of the correct label $y_i$ given the input $s_i$.
- $L_i$ measures the dissimilarity between the predicted probability distribution and the true distribution.

### Incorporating Negative-Valued Label Weights

If we use **negative-valued label weights** for certain items to penalize recommending "unwanted" items, the loss function becomes:

$L_i = w_i \cdot (-\log(p(y_i | s_i)))$

Where:
- $w_i$ is the label weight, which could be **negative** for unwanted items.
- A negative $w_i$ inverts the penalty: instead of minimizing $L_i$, the system tries to **maximize** $p(y_i | s_i)$, reducing the likelihood of these items being recommended.

---

### Gradient Blow-Up Issue

#### Gradients of Cross-Entropy Loss:
The gradient of $L_i$ with respect to the model's output (logit) $z_i$ is:

$\frac{\partial L_i}{\partial z_i} = w_i \cdot \frac{\partial (-\log(p(y_i | s_i)))}{\partial z_i} = w_i \cdot \left(-\frac{1}{p(y_i | s_i)} \cdot \frac{\partial p(y_i | s_i)}{\partial z_i}\right)$

When $p(y_i | s_i)$ to 0, the term $\frac{1}{p(y_i | s_i)}$ approaches infinity. This leads to:

- **Large gradient values** (gradient blow-up), which destabilize training, especially with **negative weights** where the penalty inversion amplifies this effect.
- The gradient magnitude becomes disproportionately large for low-probability unwanted items, which can overwhelm updates to the model and make optimization difficult.

---

### Intuition Behind the Problem

For **unwanted items**:
- A **negative weight** $w_i$ inverts the goal, making the model try to **maximize** $p(y_i | s_i)$ for those items.
- If $p(y_i | s_i)$ is already close to 0 (very unlikely), the gradient signal becomes excessively large due to $-\log(p(y_i | s_i)) \to \infty$.
- The optimizer takes excessively large steps, causing instability and poor convergence.

---

### Possible Solutions

To avoid gradient blow-up while discouraging unwanted items, consider these alternatives:

#### 1. **Clip Gradients**
   - Apply gradient clipping to cap the maximum gradient magnitude during backpropagation.

#### 2. **Smooth the Logarithm**
   - Use a smoothed version of cross-entropy loss to prevent the log function from exploding:
    $L_i = w_i \cdot (-\log(p(y_i | s_i) + \epsilon))$
    
     Where $\epsilon$ is a small positive constant.

#### 3. **Limit Negative Weights**
   - Instead of making weights strongly negative, limit their range to a less extreme value to avoid excessively penalizing unwanted items.

#### 4. **Regularize Probabilities**
   - Add a regularization term to push $p(y_i | s_i)$ for unwanted items closer to 0 without explicitly penalizing them with extreme loss terms.

#### 5. **Alternative Loss Functions**
   - Consider alternative losses like focal loss, which down-weighs the contribution of low-probability instances:
     $L_i = -w_i \cdot (1 - p(y_i | s_i))^\gamma \cdot \log(p(y_i | s_i))$
     This focuses learning on more confident predictions.

## Evaluation metrics for Negative Feedback 

Evaluating the effectiveness of negative feedback in recommender systems requires metrics that consider both the avoidance of disliked items and the overall user satisfaction. Here are the key evaluation metrics:

---

### 1. **Precision/Recall with Negative Feedback**
- **Modified Precision**: Measures the proportion of recommended items that are relevant (positively preferred) and do not belong to the negatively preferred set.
  
  $\text{Precision} = \frac{\text{True Positives (Relevant and Not Disliked)}}{\text{Total Recommended Items}}$
- **Modified Recall**: Measures how many positively preferred items were correctly recommended while avoiding negatively preferred items.
- 
  $\text{Recall} = \frac{\text{True Positives (Relevant and Not Disliked)}}{\text{Total Relevant Items}}$

---

### 2. **Discounted Cumulative Gain (DCG)**
- Incorporate a penalty for recommending disliked items by assigning them negative or zero relevance scores in the ranking.
- The formula:
  $DCG = \sum_{i=1}^{N} \frac{\text{Relevance}(i)}{\log_2(i+1)}$
  If an item is disliked, set its relevance to a negative or low value.

---

### 3. **Normalized Discounted Cumulative Gain (NDCG)**
- Normalize DCG to account for the ideal ranking, ensuring recommendations respect both positive and negative preferences:
  $NDCG = \frac{DCG}{IDCG}$
  Here, negatively preferred items contribute less or negatively to the numerator.

---

### 4. **Coverage of Negative Feedback**
- Measures how well the system avoids recommending items explicitly marked as disliked:
  $\text{Coverage of Negative Feedback} = 1 - \frac{\text{Number of Disliked Items Recommended}}{\text{Total Number of Disliked Items}}$
- Higher coverage indicates the system successfully avoids negatively marked items.

---

### 5. **Hit Rate (Avoidance Version)**
- Focuses on whether the system successfully avoids items in the negative set:
  $\text{Avoidance Rate} = 1 - \frac{\text{Hits on Negative Items}}{\text{Total Negative Items}}$

---

### 6. **Negative Feedback Precision**
- Measures how well the system correctly identifies items that should not be recommended:
  $\text{Negative Precision} = \frac{\text{Correctly Avoided Negative Items}}{\text{Total Negative Predictions}}$

---

### 7. **User Satisfaction/Engagement Metrics**
- Evaluate the broader impact of negative feedback on user experience:
  - **Click-Through Rate (CTR)**: Higher CTR on positively preferred items indicates effective use of negative feedback.
  - **Session Length**: Longer sessions may indicate improved recommendations after integrating negative feedback.
  - **Dwell Time**: Reflects user engagement with positively recommended items.

---

### 8. **F1-Score for Negative Feedback**
- Combines precision and recall for negative preferences to provide a balanced evaluation:
  $F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

---

### 9. **Custom Weighted Metrics**
- Assign different weights to penalize recommendations of disliked items more heavily:
  $\text{Weighted Loss} = \sum_{i=1}^{N} w_i \cdot \text{Loss}(i)$
  Where $w_i$ is higher for negatively preferred items.

---

### 10. **Overall Utility**
- Evaluate the combined effect of including negative feedback on system utility:
  $\text{Utility} = \sum_{i=1}^{N} (\text{Positive Relevance} - \text{Negative Relevance})$

