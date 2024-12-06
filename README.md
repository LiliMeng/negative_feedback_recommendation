# negative_feedback_recommendation

Paper: Learning from Negative User Feedback and Measuring Responsiveness for Sequential Recommenders https://arxiv.org/pdf/2308.12256

<img width="711" alt="Screenshot 2024-12-04 at 3 47 42 PM" src="https://github.com/user-attachments/assets/238ae1d6-2d50-43ca-8969-3a53a98fd170">

### Result:
The experiment model using dislikes as both input feature and training labels reduces dislike rate by 2.44% compared to a baseline of not using dislike signals. This effect is much larger than only using dislike as input feature but not training labels (-0.34%, not significant), or a heuristic solution that excludes disliked items from the model’s input sequence (-0.84%). Repeated dislike rate on the same creator decreases by 9.60%, suggesting that the model reduces similar recommendations after negative feedback



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

Evaluating negative user feedback in recommendation systems involves identifying, measuring, and analyzing signals that indicate dissatisfaction. Here are common quantitative metrics to evaluate and address negative feedback:

---

### **1. Direct Negative Feedback Metrics**
These metrics measure explicit user actions indicating dissatisfaction.

- **Dislike Rate:**  
  Percentage of items explicitly disliked by users (e.g., thumbs down or "not interested" actions).  
  **Formula:**  
  $\text{Dislike Rate} = \frac{\text{Number of Dislikes}}{\text{Total Recommendations}} \times 100$

- **Negative Feedback Rate:**  
  Includes other explicit feedback like “report” or “block” actions.  
  **Formula:**  
  $\text{Negative Feedback Rate} = \frac{\text{Number of Negative Feedback Actions}}{\text{Total Recommendations}} \times 100$

---

### **2. Indirect Negative Feedback Metrics**
These metrics use implicit signals to infer dissatisfaction.

- **Skip Rate:**  
  Frequency of users skipping recommended content (e.g., skipping videos shortly after they start).  
  
  $\text{Skip Rate} = \frac{\text{Number of Skips}}{\text{Number of Recommendations Viewed}} \times 100$

- **Short Engagement Duration:**  
  Average time spent on content that is abandoned quickly, compared to the expected duration.  
  **Metric:**  
  - Ratio of **actual watch time** to **video length**.
  - Use thresholds to flag dissatisfaction (e.g., <10% of video watched).

- **Bounce Rate:**  
  Fraction of users leaving the platform immediately after interacting with a recommendation.  
  $\text{Bounce Rate} = \frac{\text{Sessions Ending Shortly After Content Click}}{\text{Total Sessions}} \times 100$

---

### **3. Comparative Metrics**
Evaluate how negative feedback correlates with system-generated metrics.

- **CTR-to-Engagement Mismatch:**  
  If click-through rates (CTR) are high but engagement rates (e.g., watch time) are low, it might indicate clickbait recommendations.  
  **Formula:**  
  \[
  \text{CTR-to-Engagement Mismatch} = \text{CTR} - \text{Average Engagement Rate}
  \]

- **Content Quality Drop-Off:**  
  Identify categories, creators, or content types that disproportionately trigger negative feedback.

---

### **4. User Retention and Satisfaction Metrics**
Measure the long-term effects of negative feedback.

- **Churn Rate:**  
  Fraction of users reducing their engagement or leaving the platform after receiving recommendations they didn’t like.  
  $\text{Churn Rate} = \frac{\text{Users Leaving After Negative Feedback}}{\text{Total Users}}$

---

### **5. Context-Aware Analysis**
Include contextual metrics to understand dissatisfaction.

- **Recommendation Diversity Impact:**  
  Negative feedback often results from repetitive or irrelevant recommendations. Measure diversity changes in recommendations over time.  
  **Metric:**  
 $\text{Diversity Score} = \text{Entropy or Coverage of Categories in Recommended Items}$

- **Relevance Score Changes:**  
  Compare personalized relevance scores for negatively received recommendations with those receiving positive feedback.

---

### **6. Root Cause Analysis**
Segment negative feedback by:
- **Demographics:** Age, location, or other attributes.
- **Context:** Time of day, platform (mobile vs. desktop), or session type.
- **Algorithm Variant:** Compare performance across different recommendation algorithms.

---
## Relate Business metrics with negative feedback metrics
### **1. Map Negative Feedback to Business Metrics**
Identify how each negative feedback metric influences business outcomes:

#### **Revenue Impact**
- **Dislike Rate / Negative Feedback Rate → Ad Revenue or Subscription Loss**  
  Negative feedback can reduce user engagement, leading to fewer impressions for advertisements or reduced subscription renewals.  
  **Action:** Quantify the decrease in ad revenue per percentage point increase in dislike rate.

- **Skip Rate → Ad Completion Rate**  
  High skip rates for videos or content also reduce the likelihood of users watching in-stream ads, directly impacting ad completion metrics and advertiser satisfaction.

---

#### **User Retention**
- **Bounce Rate → User Churn Rate**  
  A high bounce rate often correlates with users abandoning the platform, leading to increased churn.  
  **Action:** Measure the relationship between bounce rate increases and the likelihood of users not returning within a defined period (e.g., 7-day or 30-day churn).

- **Short Engagement Duration → Lifetime Value (LTV)**  
  Reduced engagement may signal declining user interest, impacting long-term customer lifetime value.  
  **Action:** Use cohort analysis to track LTV differences for users with high vs. low engagement durations.

---

#### **User Satisfaction**
- **Net Promoter Score (NPS) Impact → Brand Loyalty**  
  Dissatisfied users reflected in NPS surveys are less likely to recommend the platform, affecting user acquisition and overall market share.

- **CTR-to-Engagement Mismatch → Trust in Recommendations**  
  If recommendations appear clickbaity (high CTR but low engagement), it can erode user trust and satisfaction. This undermines user confidence in the platform and reduces long-term engagement.

---

### **2. Quantify the Impact**
Establish quantitative relationships between negative feedback and business metrics using historical data:

#### **Correlation Analysis**
- Identify correlations between negative feedback metrics (e.g., dislike rate, bounce rate) and key business metrics (e.g., revenue, retention).
- Example: Analyze how a 1% increase in the dislike rate correlates with decreases in user retention or ad revenue.

#### **Attribution Modeling**
- Use multi-touch attribution models to measure how negative feedback events contribute to user churn, reduced engagement, or lower ad revenue.

#### **A/B Testing**
- Test interventions that address negative feedback (e.g., improving recommendation diversity) and measure the direct impact on business metrics like user retention or LTV.

---

### **3. Align Metrics with Specific Business KPIs**
Translate negative feedback into actionable insights for specific business goals:

#### **Engagement KPIs**
- **Metric Connection:**  
  Skip Rate → Average Session Duration, Number of Content Pieces Viewed
- **Goal:**  
  Increase session duration by reducing content skips, leading to higher engagement metrics.

#### **Revenue KPIs**
- **Metric Connection:**  
  Ad Completion Rate → Ad Revenue
- **Goal:**  
  Reduce negative feedback to ensure users stay on the platform longer and see more ads.

#### **Retention KPIs**
- **Metric Connection:**  
  Bounce Rate → Daily Active Users (DAU), Monthly Active Users (MAU)
- **Goal:**  
  Improve retention by reducing bounce rates, stabilizing DAU/MAU metrics.

---

### **4. Integrate Metrics into a Business Dashboard**
Use a combined dashboard to monitor both user-centric and business-centric metrics side by side. Key metrics to track:
- **Negative Feedback Metrics:** Dislike Rate, Skip Rate, Bounce Rate.
- **Business Metrics:** Revenue (ad or subscription), User Retention, LTV, NPS.

---

### **5. Prioritize Based on Business Impact**
Not all negative feedback has equal business implications. Prioritize addressing feedback metrics with the most significant impact:
- Conduct sensitivity analysis to identify which feedback metrics (e.g., skip rate vs. dislike rate) have the highest influence on business goals.
- Focus resources on improving those metrics with measurable ROI.

---

### **6. Present Findings to Stakeholders**
Clearly communicate how reducing negative feedback aligns with business goals:
- Use case studies or experiments to show concrete improvements (e.g., "Reducing skip rate by 5% increased ad revenue by 10%").
- Quantify the potential upside of addressing negative feedback.

