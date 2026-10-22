
# Midterm Exam Notes: Data Wrangling & Feature Engineering in Machine Learning

## Module 1: ML Foundations and End-to-End Pipeline

# Lecture 1: Introduction to Machine Learning

## 1.1 What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence that enables computers to learn patterns from data and use those patterns to make predictions or decisions without requiring every rule to be explicitly programmed.

In traditional programming, a programmer writes the rules that a computer follows. In machine learning, the programmer provides data and a learning algorithm that discovers useful patterns from examples.

### Traditional programming vs. machine learning

Traditional programming

Rules + Input Data → Computer → Output

Example: A programmer defines a rule that a student passes if their marks are at least 50. The program applies this rule to calculate the result.

Machine learning

Training Data + Expected Outputs → Learning Algorithm → Trained Model

Example: A model learns from previous students' marks, attendance, and pass/fail outcomes to predict whether a new student is likely to pass.

The main difference is that traditional programming explicitly specifies the decision rules, whereas machine learning estimates patterns or rules from examples.

## 1.2 Three core requirements of machine learning

According to your notes, three essential elements are needed for the learning process.

### 1. Input data points

Input data points are the observations or measurements supplied to the learning algorithm. They describe the examples from which the model learns.

For example, a dataset used to predict student performance might contain:

- Study hours
- Attendance percentage
- Assignment marks
- Previous examination marks

Each row represents an observation, and each input column represents a feature.

### 2. Expected outputs or labels

Labels are the known target values associated with training examples in supervised learning. They provide the expected answers against which model predictions can be compared.

For example:

| Study hours | Attendance | Result |
| ----------- | ---------- | ------ |
| 2           | 55%        | Fail   |
| 5           | 80%        | Pass   |
| 7           | 95%        | Pass   |

Here, study hours and attendance are input features, while `Result` is the target label.

Labels are necessary for supervised classification and regression. Unsupervised learning, by contrast, can learn from data without predefined target labels.

### 3. Objective function or measure of goodness

An objective function defines what the learning algorithm tries to optimize. It helps determine how well a model performs and guides the adjustment of model parameters.

Different tasks use different evaluation measures.

- Classification: Accuracy, precision, recall, F1-score, or log loss.
- Regression: Mean Absolute Error (MAE), Mean Squared Error (MSE), or $R^2$.

For example, in a regression task, Mean Squared Error measures the average squared difference between actual and predicted values:

$$
 MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

Here, $y_i$ is the actual value, $\hat y_i$ is the predicted value, and $n$ is the number of observations. Lower MSE generally indicates better predictions on the evaluated dataset.

## 1.3 Reasoning modes: deduction and induction

### Deduction

Deduction moves from a general rule to a specific conclusion. If the premises are true and the reasoning is valid, the conclusion follows logically.

Example:

1. All registered students have student IDs.
2. Ali is a registered student.
3. Therefore, Ali has a student ID.

Deduction applies an existing rule to a particular case.

### Induction

Induction moves from particular observations toward a general pattern or rule.

Example:

1. Several previous students who studied consistently passed their exams.
2. A learner observes this pattern across many examples.
3. The learner infers that consistent study may increase the likelihood of passing.

Inductive conclusions are not guaranteed to be true in every case. New observations can reveal exceptions.

Connection to machine learning: Machine learning automates aspects of inductive inference. It analyzes examples to learn patterns that can generalize to previously unseen data. The quality of generalization depends on the training data, the algorithm, and the evaluation process.

## 1.4 Mitchell's formal definition of machine learning

Tom Mitchell's definition describes learning through three components:

- $T$: Task — what the program is required to do.
- $E$: Experience — the data or experience from which it learns.
- $P$: Performance measure — how success is evaluated.

A program learns from experience $E$ with respect to task $T$ and performance measure $P$ if its performance at task $T$, measured by $P$, improves with experience $E$.

### Example: Email spam detection

- Task (T): Classify emails as spam or not spam.
- Experience (E): A collection of previously labeled emails.
- Performance measure (P): A suitable evaluation measure, such as precision, recall, or F1-score.

As the algorithm learns from examples, it can improve its ability to classify new emails.

This definition is important because it explains machine learning in terms of a task, experience, and measurable improvement rather than merely describing a particular algorithm.

## 1.5 Main machine learning task types

### Classification

Classification predicts a discrete category or class.

Examples:

- Spam or not spam
- Pass or fail
- Grant recommended or not recommended
- Cat, dog, or zebra

Classification can be binary, involving two classes, or multiclass, involving more than two classes.

### Regression

Regression predicts a numerical quantity that can take continuous values.

Examples:

- Predicting house prices
- Estimating a student's final marks
- Forecasting monthly sales
- Predicting temperature

A simple distinction is that classification predicts which category, while regression predicts how much.

# Lecture 2: Machine Learning Landscape and Practical Reality

## 2.1 Artificial Intelligence, Machine Learning, and Deep Learning

These terms are related but do not mean exactly the same thing.

### Artificial Intelligence (AI)

Artificial Intelligence is the broad field concerned with creating systems that perform tasks associated with intelligent behavior, such as reasoning, planning, perception, language understanding, and decision-making.

AI includes approaches that use explicit rules, search algorithms, logic, and learning from data.

### Machine Learning (ML)

Machine Learning is a subfield of AI in which algorithms learn patterns from data to perform tasks such as classification, regression, clustering, and prediction.

Instead of manually defining every decision rule, a developer trains a model on examples.

### Deep Learning (DL)

Deep Learning is a subfield of machine learning that uses neural networks with multiple layers to learn increasingly complex representations of data.

It is particularly useful for complex data such as images, audio, natural language, and video.

The relationship is:

$$
 \text{Deep Learning}\subseteq\text{Machine Learning}\subseteq\text{Artificial Intelligence}
$$

In other words, deep learning is part of machine learning, and machine learning is part of AI.

## 2.2 Real-world applications

### Virtual assistants

Systems such as Siri use speech recognition, language processing, and related AI techniques to interpret commands and respond to users.

### Autonomous driving

Autonomous driving systems use sensor data, computer vision, prediction, and planning to identify objects, estimate their movement, and support driving decisions.

### IBM Watson and Jeopardy!

IBM Watson demonstrated the use of AI techniques to interpret natural-language clues and select answers in the quiz show Jeopardy!

### Deep Blue and chess

IBM's Deep Blue demonstrated powerful computer-based chess play using search, evaluation functions, and specialized computing techniques. It is an example of AI, but not a typical modern example of a system learning from training data.

## 2.3 Why data preparation matters in machine learning

Your notes emphasize that data scientists may spend approximately 70–80% of project time on data cleaning, wrangling, and feature engineering. This is an estimate rather than a universal rule, but it highlights an important practical reality.

Real-world datasets frequently contain:

- Missing values
- Duplicate records
- Incorrect data types
- Inconsistent spellings
- Categorical variables represented as text
- Outliers and measurement errors
- Features with very different numerical scales
- Irrelevant or redundant columns

A machine learning algorithm cannot automatically resolve every data quality issue correctly. Poor input data can lead to unreliable predictions even when an advanced algorithm is used.

Example: If a dataset stores prices as text strings such as `"1000"` and `"2500"`, numerical calculations may fail until the values are converted into numeric types.

```
import pandas as pd

df = pd.DataFrame({    "price": ["1000", "2500", "1800"]})

df["price"] = pd.to_numeric(df["price"])

print(df["price"].mean())
```

Output:

```
1766.6666666666667
```

The important lesson is that preparing meaningful, consistent features is a central part of building a reliable machine learning model.

# Lecture 3: End-to-End Supervised Classification — Student Grant Recommendation

## 3.1 Problem definition

The objective is to predict whether a student should be recommended for a grant.

The target variable is `Recommend`, with two possible values:

- `Yes`
- `No`

The input features are:

- `ResearchScore`
- `ProjectScore`
- `OverAllGrade`
- `Obedient`

Because the target contains two categories, this is a binary classification problem.

The overall process includes data preparation, feature transformation, model training, prediction on new records, and saving the trained objects for later use.

## 3.2 Feature scaling

Feature scaling transforms numerical features so that their magnitudes or distributions are placed on a comparable scale.

Suppose a dataset contains:

- `ResearchScore`, ranging from 0 to 10.
- `ProjectScore`, ranging from 0 to 1000.

A feature with a much larger numerical range can dominate certain distance calculations or affect optimization in scale-sensitive algorithms.

### Standardization using StandardScaler

Standardization transforms a feature using its mean and standard deviation:

$$
z = \frac{x - \mu}{\sigma}
$$

Where:

- $x$ is the original value.
- $\mu$ is the training feature's mean.
- $\sigma$ is the training feature's standard deviation.
- $z$ is the standardized value.


The transformed feature generally has a mean near zero and a standard deviation near one on the training data, assuming nonzero variance.

### Python example

```python
import pandas as pdfrom sklearn.preprocessing
import StandardScalerdf = pd.DataFrame({    "ResearchScore": [2, 4, 6, 8, 10],    "ProjectScore": [100, 300, 500, 700, 900]})

scaler = StandardScaler()scaled_values = scaler.fit_transform(    df[["ResearchScore", "ProjectScore"]])scaled_df = pd.DataFrame(    scaled_values,    columns=["ResearchScore", "ProjectScore"])

print(scaled_df)

```

`fit_transform()` performs two operations:

1. `fit()` learns the mean and standard deviation from the supplied training data.
2. `transform()` uses those learned statistics to scale the values.

Important: Fit the scaler on the training data only. Use that same fitted scaler to transform validation, test, and future data. Fitting it independently on new data can produce inconsistent transformations and data leakage.

## 3.3 Categorical encoding using one-hot encoding

Machine learning algorithms generally require numerical feature representations. However, the student dataset contains categorical variables such as grades and obedience status.

For example:

- `OverAllGrade`: A, B, C, E, F
- `Obedient`: Y, N

One-hot encoding converts each category into a separate binary indicator column.

Example:

| OverAllGrade | OverAllGrade_A | OverAllGrade_B | OverAllGrade_C |
| ------------ | -------------- | -------------- | -------------- |
| A            | 1              | 0              | 0              |
| B            | 0              | 1              | 0              |
| C            | 0              | 0              | 1              |

A value of 1 indicates the category present in the observation; 0 indicates its absence.

### Python example

```python
import pandas as pd

df = pd.DataFrame({    "OverAllGrade": ["A", "B", "C", "A"],    "Obedient": ["Y", "N", "Y", "N"]})encoded_df = pd.get_dummies(    df,    columns=["OverAllGrade", "Obedient"],    dtype=int)

print(encoded_df)
```

The result contains binary columns such as `OverAllGrade_A`, `OverAllGrade_B`, `Obedient_Y`, and `Obedient_N`.

`pd.get_dummies()` is convenient for converting categories into indicator columns. For production machine learning systems, a fitted encoder is often preferable because it can preserve the expected feature schema.

## 3.4 Training logistic regression

Logistic regression is a supervised learning algorithm commonly used for classification.

Despite its name, it estimates class probabilities using a logistic function. In binary classification, it models the probability of a positive class from a weighted combination of the input features.

The logistic function is:

$$
 P(y=1\mid x)=\frac{1}{1+e^{-z}}
$$

where $z$ is the weighted sum of the input features, including the model's intercept.

The resulting probability lies between 0 and 1. A decision threshold, often 0.5 by default, can be used to convert that probability into a predicted class.

### Python example

```python
import pandas as pdfrom sklearn.linear_model
import LogisticRegressionX = pd.DataFrame({    "ResearchScore": [0.2, 0.4, 0.6, 0.8, 1.0, 0.3],    "ProjectScore": [0.1, 0.3, 0.5, 0.7, 0.9, 0.2]})y = ["No", "No", "No", "Yes", "Yes", "No"]

model = LogisticRegression()

model.fit(X, y)

print(model.predict(X))
```

Here:

- `X` contains the input features.
- `y` contains the known target labels.
- `fit()` trains the model.
- `predict()` produces class predictions.

The example illustrates the training interface; a real project should use a proper training/test split and evaluate predictions on unseen records.

## 3.5 The prediction alignment problem

A major practical issue occurs when a model is trained with one set of encoded columns but a new record produces a different set.

Suppose the training data has the following columns:

```
OverAllGrade_A
OverAllGrade_B
OverAllGrade_C
Obedient_N
Obedient_Y
```

Now consider a new student whose grade is A and whose obedience status is Y.

Encoding only that single record might create:

```
OverAllGrade_A
Obedient_Y
```

The model expects five columns, but the new record contains only two. This can cause a feature-name or shape error, or lead to incorrect inputs if columns are misaligned.

### Solution using reindex

```python
import pandas as pdtraining_columns = [    "OverAllGrade_A",    "OverAllGrade_B",    "OverAllGrade_C",    "Obedient_N",    "Obedient_Y"]new_student = pd.DataFrame({    "OverAllGrade": ["A"],    "Obedient": ["Y"]})

new_encoded = pd.get_dummies(    new_student,    columns=["OverAllGrade", "Obedient"],    dtype=int)

new_encoded = new_encoded.reindex(    columns=training_columns,    fill_value=0)

print(new_encoded)
```

Expected output:

```
   OverAllGrade_A  OverAllGrade_B  OverAllGrade_C  Obedient_N  Obedient_Y
0               1               0               0           0           1
```

The `reindex()` operation ensures that the new input has the same columns in the same order as the training data. Missing columns receive zero.

For a full workflow, the same fitted scaler must also be applied to numerical features, and the final feature matrix must use the exact feature order expected by the model.

Best practice: Use a scikit-learn `Pipeline` with a `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder(handle_unknown="ignore")`. This keeps preprocessing and model prediction together and reduces the risk of inconsistent transformations.

## 3.6 Model persistence using joblib

Model persistence means saving a trained model to disk so it can be loaded and reused without training it from scratch.

The `joblib` library is commonly used to save scikit-learn models and preprocessing objects.

```python
import joblibjoblib.dump(model, "model.pickle")joblib.dump(scaler, "scaler.pickle")
```

To load the saved objects:

```python
loaded_model = joblib.load("model.pickle")

loaded_scaler = joblib.load("scaler.pickle")
```

The loaded model can then be used for prediction, provided the new data undergoes the same preprocessing and feature alignment as the training data.

Important: Only load pickle/joblib files from trusted sources. Such files can execute malicious code during deserialization. Keep the model, scaler, encoder, and feature schema consistent as part of the saved workflow.

# Lecture 4: Machine Learning Taxonomy and Algorithms

## 4.1 Supervised learning

Supervised learning uses labeled training data. Each training observation has input features and a known target.

The model learns a relationship between inputs and targets so it can predict the target for new observations.

### Main tasks

Classification: Predicts a category.

Example: Predicting whether a customer will leave a service.

Regression: Predicts a numerical quantity.

Example: Predicting a house's selling price.

Common supervised algorithms include logistic regression, linear regression, decision trees, random forests, support vector machines, and neural networks.

## 4.2 Unsupervised learning

Unsupervised learning works with data that does not have predefined target labels. The objective is to discover structure, relationships, or patterns in the observations.

### Clustering

Clustering groups similar observations together.

Example: A retailer groups customers according to purchasing patterns to identify customer segments.

### Association rules

Association rule mining identifies items or events that frequently occur together.

Example: A store may discover that customers who buy one product often buy another product in the same transaction.

Unlike supervised learning, the algorithm does not learn from a predefined correct class label for every example.

## 4.3 Self-supervised learning

Self-supervised learning generates training signals from the data itself rather than requiring every target to be manually labeled.

For example, a language model may learn by predicting a missing or next token from surrounding text. The original text supplies the training signal.

Self-supervised learning is useful when large amounts of raw data are available but human labeling would be expensive or time-consuming.

## 4.4 Reinforcement learning

Reinforcement learning trains an agent to make decisions through interaction with an environment.

Its main components are:

- Agent: The decision-making system.
- Environment: The world or simulation in which the agent acts.
- State: The current situation.
- Action: A decision made by the agent.
- Reward: Feedback indicating the outcome or value of an action.
- Policy: The strategy used to choose actions.

The agent aims to learn a policy that maximizes cumulative reward over time.

Example: A game-playing agent tries different actions and learns strategies from rewards or penalties.

## 4.5 Clustering algorithm taxonomy

### K-Means: Centroid-based clustering

K-Means divides observations into a specified number $K$ of clusters. It assigns each observation to a cluster centroid and repeatedly updates the centroids to reduce within-cluster squared distances.

It is relatively efficient for many numerical datasets but requires choosing $K$, is sensitive to scaling and initialization, and can struggle with irregular cluster shapes.

### Hierarchical clustering

Hierarchical clustering creates a hierarchy of groups.

- Agglomerative: Starts with individual observations and repeatedly merges clusters.
- Divisive: Starts with a large group and repeatedly splits it into smaller groups.

The result can be represented as a tree-like structure called a dendrogram.

### DBSCAN: Density-based clustering

DBSCAN groups observations in dense regions and marks observations in sufficiently sparse regions as noise or outliers.

It can discover irregularly shaped clusters and does not require specifying the number of clusters in advance. However, its results depend on parameters such as neighborhood radius and minimum points, and it can struggle when cluster densities vary considerably.

### Gaussian Mixture Models (GMM): Distribution-based clustering

A Gaussian Mixture Model assumes that the data is generated by a mixture of Gaussian probability distributions.

It estimates the distributions' parameters and can assign each observation probabilities of belonging to different components.

Unlike basic K-Means, which makes a single hard cluster assignment, a GMM can provide soft assignments, representing uncertainty about cluster membership.

## Module 2: Data Acquisition and Preprocessing

# Lecture 5 & 6: Data Collection and Structural Inspection

## 5.1 What is data acquisition?

Data acquisition is the process of collecting data from different sources so it can be analyzed or used to train a machine learning model.

Common sources include:

- CSV and TSV files
- Relational databases
- APIs and web services
- Sensors and IoT devices
- Application logs
- Surveys and experiments
- Images, audio recordings, and text documents

The collected data may be structured, semi-structured, or unstructured.

Structured data has a well-defined organization, such as rows and columns in a spreadsheet.

Semi-structured data has some organizational structure but may not follow a fixed tabular schema, such as JSON documents.

Unstructured data includes information such as free-form text, images, audio, and video that does not naturally fit into a conventional table.

## 5.2 CRISP-DM workflow

CRISP-DM stands for Cross-Industry Standard Process for Data Mining. It is a framework for organizing data mining and machine learning projects into six major phases.

### 1. Business Understanding

Define the business problem, objectives, constraints, and success criteria.

Example: Determine whether the goal is to reduce customer churn or predict sales.

### 2. Data Understanding

Collect initial data, inspect its structure, calculate summaries, and identify data quality problems.

### 3. Data Preparation

Clean records, handle missing values, remove duplicates, transform data types, and create suitable features.

### 4. Modeling

Select algorithms, prepare training data, train models, and tune parameters.

### 5. Evaluation

Assess whether the model meets the project's objectives and performs adequately on unseen data.

### 6. Deployment

Put the model or analytical results into practical use, then monitor performance and maintain the system.

These phases are not always strictly linear. For example, evaluation may reveal data quality problems that require returning to data preparation.

## 5.3 CSV and TSV files

Delimited text files store tabular data as lines of text. Each line typically represents a record, while delimiters separate individual fields.

### CSV: Comma-Separated Values

A CSV file typically uses commas to separate fields.

Example:

```
Name,Age,Marks
Ali,20,85
Sara,21,92
Ahmed,19,78
```

### TSV: Tab-Separated Values

A TSV file typically uses tab characters to separate fields.

Example:

```
Name    Age    Marks
Ali     20     85
Sara    21     92
Ahmed   19     78
```

The displayed spacing in the TSV example represents tab-separated fields.

### Reading CSV and TSV files with pandas

```python
import pandas as pd
# Read a CSV
filedf_csv = pd.read_csv("students.csv")
# Read a TSV
filedf_tsv = pd.read_csv("students.tsv", sep="\t")

print(df_csv.head())
```

The `sep` parameter specifies the field delimiter. CSV files commonly use the default comma separator, while TSV files use `"\t"`.

## 5.4 Inspecting a dataset

Before cleaning or modeling data, it is important to understand the dataset's dimensions, column types, and sample records.

### `df.shape`

Returns the number of rows and columns.

```
print(df.shape)
```

Example output:

```
(1000, 8)
```

This means the dataset contains 1,000 rows and 8 columns.

### `df.dtypes`

Displays the data type of each column.

```
print(df.dtypes)
```

Common pandas data types include:

- `int64`: Integer values
- `float64`: Decimal or floating-point values
- `object`: Often strings or mixed Python objects
- `bool`: Boolean values
- `datetime64[ns]`: Date and time values

The data type matters because it determines which operations are appropriate. For example, a price column stored as text cannot reliably be used in arithmetic until it is converted into a numeric type.

### `df.head(n)`

Displays the first $n$ rows.

```
print(df.head(5))
```

This is useful for checking column names, formatting, unexpected values, and whether the file has been loaded correctly.

### Additional useful inspection functions

```
print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df.nunique())
```

- `info()` summarizes column types and non-null counts.
- `describe()` calculates descriptive statistics for suitable columns.
- `isnull().sum()` counts missing values in each column.
- `nunique()` counts distinct values in each column.

Together, these operations help identify potential preprocessing requirements before modeling begins.

## Module 3: Data Wrangling and Summarization

# Lecture 7: Core Data Wrangling Operations

## 7.1 Definition of data wrangling

Data wrangling, also called data munging, is the process of converting raw, inconsistent, or poorly structured data into a clean and organized form suitable for analysis and machine learning.

It involves inspecting, filtering, correcting, transforming, and summarizing information.

For example, a customer dataset might contain:

- Prices stored as strings
- Missing customer ages
- Duplicate customer IDs
- Inconsistent city names
- Dates stored in different formats

Data wrangling addresses these issues before the data is supplied to an analytical method.

## 7.2 Understanding data

The first step is to inspect the dataset and identify its structure.

```python
import pandas as pd

df = pd.read_csv("sales.csv")

print(df.shape)

print(df.dtypes)

print(df.head())

print(df.isnull().sum())
```

These statements reveal the dataset's dimensions, column types, sample records, and missing-value counts.

Understanding the data helps determine which cleaning operations are necessary.

## 7.3 Filtering data

Filtering selects rows that satisfy specified conditions. It is useful when the analysis concerns only a subset of observations.

Suppose a dataset contains products and their prices.

```python
import pandas as pd

df = pd.DataFrame({    "Product": ["A", "B", "C", "D"],    "Price": [500, 1500, 2500, 800]})expensive_products = df[df["Price"] > 1000]

print(expensive_products)
```

Output:

```
  Product  Price
1       B   1500
2       C   2500
```

The expression `df["Price"] > 1000` produces a Boolean condition. Pandas selects the rows for which the condition is `True`.

### Filtering with multiple conditions

```python
filtered = df[    (df["Price"] > 500) &    (df["Price"] < 2000)]
```

The `&` operator means AND, so both conditions must be satisfied. Parentheses around each condition are important.

Use `|` for OR when either condition may be satisfied.

## 7.4 Type casting

Type casting means converting a variable from one data type to another.

It is necessary when values are stored in a format that does not match their intended meaning.

### Converting strings to numeric values

```python
df["Price"] = pd.to_numeric(    df["Price"],    errors="coerce")
```

`errors="coerce"` converts values that cannot be parsed into `NaN`, which can then be investigated or handled as missing data.

### Converting strings to dates

```python
df["Date"] = pd.to_datetime(    df["Date"],    errors="coerce")
```

This allows date-based operations, such as extracting the year, month, or weekday.

### Converting a column to integers

```python
df["Age"] = pd.to_numeric(    df["Age"],    errors="raise").astype("int64")
```

This example raises an error if the values cannot be converted to numbers. It also assumes there are no missing or non-integral values that would prevent conversion to an integer type.

Type conversion should be performed carefully because invalid conversions can create missing values or errors, while inappropriate conversions can lose information.

## 7.5 Transformation utilities: `map()`, `applymap()`, and `apply()`

These pandas methods are used to transform data, but they operate at different levels.

### A. `map()`

`map()` is a Series method. It applies a mapping or function to individual values in one column.

Example: Convert customer types into numeric codes.

```python
import pandas as pd

df = pd.DataFrame({    "user_type": ["a", "b", "c", "a"]})

df["user_code"] = df["user_type"].map({    "a": 0,    "b": 1,    "c": 2})

print(df)
```

Output:

```
  user_type  user_code
0         a          0
1         b          1
2         c          2
3         a          0
```

If a value is absent from the mapping dictionary, the mapped result is generally missing. Therefore, the mapping should cover the expected categories or explicitly handle unknown categories.

### B. `applymap()`

The source notes use `applymap()` for element-wise transformations across an entire DataFrame. In newer pandas versions, `DataFrame.map()` is the corresponding element-wise DataFrame method; `applymap()` is deprecated in recent versions.

Example using the modern method:

```python
import pandas as pd

df = pd.DataFrame({    "A": [1, 2, 3],    "B": [4, 5, 6]})

result = df.map(lambda x: x * 2)

print(result)
```

Output:

```
   A   B
0  2   8
1  4  10
2  6  12
```

Every individual cell is multiplied by two.

Element-wise functions should be chosen according to the data types in the DataFrame. Applying a numeric operation to text or date values may produce errors.

### C. `apply()`

`apply()` applies a function along a specified axis of a Series or DataFrame. For a DataFrame, `axis=0` applies a function to each column, while `axis=1` applies it to each row.

Example: Calculate the range of each numerical column.

```python
import pandas as pd

df = pd.DataFrame({    "Math": [70, 80, 90],    "Science": [60, 85, 95]})

ranges = df.apply(    lambda column: column.max() - column.min(),    axis=0)

print(ranges)
```

Output:

```
Math       20
Science    35
dtype: int64
```

The range is the maximum value minus the minimum value.

### Quick comparison

| Method            | Main purpose                   | Typical usage                         |
| ----------------- | ------------------------------ | ------------------------------------- |
| `map()`           | Transform values in a Series   | Replace categories using a dictionary |
| `DataFrame.map()` | Transform every DataFrame cell | Apply an element-wise function        |
| `apply()`         | Apply a function along an axis | Calculate column statistics           |

## 7.6 Imputing missing values

Missing values occur when information is absent, unknown, unrecorded, or invalid.

In pandas, missing data is commonly represented by `NaN`, `None`, or `NaT`, depending on the column's type.

Missing values can cause problems because some calculations and machine learning algorithms do not accept them directly.

Common strategies include:

- Removing rows or columns with excessive or essential missing data.
- Replacing missing numerical values with the mean or median.
- Filling missing values using neighboring observations when the data structure justifies it.
- Using more advanced statistical or model-based imputation.

The correct strategy depends on why the data is missing, how much is missing, and what the feature represents.

## 7.7 Handling duplicates

A duplicate record is a repeated observation that may incorrectly increase the apparent frequency of a particular event or entity.

Duplicates can arise from repeated data collection, import errors, or merging datasets.

```
# Identify duplicate rowsprint(df.duplicated().sum())# Remove exact duplicate rowsdf = df.drop_duplicates()
```

Sometimes only a specific identifier should be considered:

```
df = df.drop_duplicates(    subset=["serial_no"],    keep="first")
```

This keeps the first row for each `serial_no` and removes later rows sharing that identifier.

Important: Repeated identifiers do not always mean records are duplicates. Before removing records, confirm that the chosen columns genuinely identify a unique entity or event.

## 7.8 Handling categorical data

Categorical variables represent groups or labels rather than continuous numerical measurements.

Examples include:

- Department
- City
- Product category
- Education level
- Customer type

Many machine learning algorithms require these variables to be encoded numerically.

For nominal variables, one-hot encoding is usually a suitable approach. For ordinal variables, an ordered encoding may preserve meaningful rank information.

```python
import pandas as pd

df = pd.DataFrame({    "Department": ["IT", "HR", "Finance", "IT"]})

encoded = pd.get_dummies(    df,    columns=["Department"],    dtype=int)

print(encoded)
```

The resulting indicator columns represent the departments without incorrectly implying that one department has a greater numerical rank than another.

## 7.9 Normalization and feature scaling

Normalization is often used to describe rescaling values to a common range. More generally, feature scaling makes numerical features comparable in magnitude.

### Min-max normalization

The min-max formula is:

$$
 x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

It usually maps a feature to the interval $[0,1]$ when the minimum and maximum are calculated from the data being transformed.

```python
from sklearn.preprocessing import MinMaxScalerscaler = MinMaxScaler()

df[["Price"]] = scaler.fit_transform(    df[["Price"]])
```

### Standardization

Standardization subtracts the mean and divides by the standard deviation. It is useful for algorithms sensitive to feature scale.

```python
from sklearn.preprocessing import StandardScalerscaler = StandardScaler()

df[["Price"]] = scaler.fit_transform(    df[["Price"]])
```

The two methods are not identical: min-max scaling sets a target range, whereas standardization centers the data and scales it according to its variability.

For predictive modeling, learn scaling parameters from the training set and reuse them on later data.

## 7.10 String manipulation

String manipulation cleans or transforms text values so they follow a consistent format.

Common operations include:

- Lowercasing or uppercasing
- Removing leading and trailing spaces
- Replacing inconsistent spellings
- Extracting information with regular expressions
- Splitting strings into components

Example:

```python
import pandas as pd

df = pd.DataFrame({    "City": [" Lahore ", "KARACHI", " lahore", "Islamabad "]})

df["City"] = df["City"].str.strip().str.lower()

print(df)
```

Output:

```
        City
0     lahore
1    karachi
2     lahore
3  islamabad
```

Standardizing strings helps prevent the same category from being treated as multiple categories because of capitalization or whitespace.

## 7.11 Data summarization

Data summarization compresses detailed records into compact statistics or aggregate values. It helps analysts identify patterns, compare groups, and understand the dataset before modeling.

Typical techniques include:

- Mean, median, minimum, and maximum
- Frequency counts
- Group-level averages
- Totals and counts by category
- Summary statistics for numerical variables

For example, a sales dataset can be summarized by calculating total sales for each product category.

```python
import pandas as pd

df = pd.DataFrame({    "Category": ["A", "B", "A", "B", "A"],    "Sales": [100, 200, 150, 300, 250]})

summary = df.groupby("Category")["Sales"].sum()

print(summary)
```

Output:

```
Category
A    500
B    500
Name: Sales, dtype: int64
```

This shows the total sales contributed by each category.

# Lecture 8: Advanced Missing-Value Imputation and Data Summarization

## 8.1 Missing-value imputation strategies

Imputation means replacing missing values with reasonable estimates or substitute values. It allows more observations to be retained than would be possible if every incomplete row were removed.

However, imputation can introduce bias if the replacement method is inappropriate.

### A. Removal of missing records

Rows with missing values can be removed when the missing information is essential and enough usable observations remain.

```python
df = df.dropna(subset=["date"])
```

This removes rows in which `date` is missing. Missing values in other columns do not automatically cause a row to be dropped by this command.

Advantages:

- Simple to implement
- Avoids inventing replacement values
- Appropriate when a missing field makes an observation unusable

Disadvantages:

- Reduces the sample size
- May introduce bias if missingness is systematic
- Can discard useful information from otherwise complete columns

Removal is not automatically the best choice for large datasets; the reason for missingness still matters.

### B. Mean imputation

Mean imputation replaces missing numerical values with the arithmetic mean of the observed values.

$$
 \bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

Example:

```python
import pandas as pd

df = pd.DataFrame({    "price": [100, 200, None, 300, 400]})mean_price = df["price"].mean()

df["price"] = df["price"].fillna(mean_price)

print(df)
```

The mean is 250, so the missing price is replaced with 250.

Mean imputation is simple, but it can reduce observed variability and is sensitive to extreme outliers.

### C. Median imputation

The median is the middle value after the observed values are sorted. With an even number of values, it is generally the average of the two middle values.

Median imputation is often more robust than mean imputation when numerical data is heavily skewed or contains extreme values.

```python
df["price"] = df["price"].fillna(    df["price"].median())
```

The mean and median are both central-tendency measures, but they respond differently to outliers.

### D. Forward fill

Forward fill replaces a missing value with the most recent preceding non-missing value.

```python
df["user_type"] = df["user_type"].ffill()
```

Example:

| Row | Original value | After forward fill |
| --- | -------------- | ------------------ |
| 1   | A              | A                  |
| 2   | Missing        | A                  |
| 3   | B              | B                  |
| 4   | Missing        | B                  |

Forward fill is useful for ordered data when the last observed value reasonably remains valid for subsequent observations.

### E. Backward fill

Backward fill replaces a missing value with the next subsequent non-missing value.

```python
df["user_type"] = df["user_type"].bfill()
```

Example:

| Row | Original value | After backward fill |
| --- | -------------- | ------------------- |
| 1   | Missing        | A                   |
| 2   | A              | A                   |
| 3   | Missing        | B                   |
| 4   | B              | B                   |

Backward fill is appropriate only when using a future observation is justified. In forecasting or real-time prediction, it can introduce information from the future that would not actually have been available at prediction time.

### Comparison of missing-value strategies

| Method        | Replacement rule                | Main consideration           |
| ------------- | ------------------------------- | ---------------------------- |
| Removal       | Delete selected incomplete rows | Can lose information         |
| Mean          | Average of observed values      | Sensitive to outliers        |
| Median        | Middle observed value           | More robust to outliers      |
| Forward fill  | Previous valid value            | Requires meaningful ordering |
| Backward fill | Next valid value                | Can use future information   |

## 8.2 Detecting and removing duplicates

### Detect duplicates

```python
duplicates = df.duplicated(    subset=["serial_no"])

print(duplicates)
```

The returned Boolean Series indicates whether each row repeats an earlier `serial_no` value.

### Count duplicates

```python
print(    df.duplicated(subset=["serial_no"]).sum())
```

This counts repeated identifiers after their first occurrence.

### Remove duplicates

```python
df = df.drop_duplicates(    subset=["serial_no"],    keep="first")
```

The `keep="first"` parameter retains the first occurrence and removes subsequent rows with the same identifier.

Other options include `keep="last"` to retain the last occurrence or `keep=False` to remove every record belonging to a duplicated group.

Before removing duplicates, check whether the repeated records represent genuine repeated events or accidental duplication.

## 8.3 Data summarization and aggregation

### `groupby()`

The `groupby()` operation divides observations into groups based on one or more columns and then calculates statistics for each group.

Example: Calculate average marks by department.

```python
import pandas as pd

df = pd.DataFrame({    "Department": ["IT", "IT", "HR", "HR"],    "Marks": [80, 90, 70, 85]})

result = df.groupby("Department")["Marks"].mean()

print(result)
```

Output:

```
Department
HR    77.5
IT    85.0
Name: Marks, dtype: float64
```

### Multiple aggregation functions

```python
summary = df.groupby("Department")["Marks"].agg(    ["count", "mean", "min", "max"])

print(summary)
```

This provides several statistics for each group.

### `value_counts()`

`value_counts()` calculates the frequency of each distinct value in a Series.

```python
print(df["Department"].value_counts())
```

This is particularly useful for understanding categorical distributions and identifying rare categories.

### `describe()`

```python
print(df["Marks"].describe())
```

For numerical data, `describe()` typically reports the count, mean, standard deviation, minimum, quartiles, and maximum.

For categorical data, it reports information such as the number of non-missing observations, number of unique categories, most frequent value, and its frequency.

Why summarization matters: It makes patterns visible, supports data quality checks, and helps guide feature engineering and model selection.

## Module 4: Feature Engineering Across Different Data Types

# Lecture 11: Feature Engineering — Numeric Data Basics

## 11.1 What is feature engineering?

Feature engineering is the process of creating, transforming, or selecting input variables so that a machine learning algorithm can learn useful patterns more effectively.

A feature is an input variable used by a model. For example, in a house-price prediction system, area, number of bedrooms, age of the property, and location may be features.

Raw data does not always contain information in the most useful form. Feature engineering converts raw measurements into representations that may better capture the underlying relationships.

The three sub-domains covered in your notes are:

1. Feature extraction and engineering
2. Feature scaling
3. Feature selection

## 11.2 Feature extraction and feature engineering

Feature extraction transforms raw data into meaningful numerical representations.

Examples include:

- Extracting the year from a date.
- Calculating the average purchase amount from transaction records.
- Converting text documents into TF-IDF vectors.
- Extracting pixel intensity statistics from images.

Feature engineering can also create new variables by combining existing features.

For example, a dataset might contain total distance and total travel time. A new feature can be created to represent average speed.

$$
 \text{Average Speed}=\frac{\text{Distance}}{\text{Time}}
$$

```python
import pandas as pd

df = pd.DataFrame({    "Distance": [100, 150, 200],    "Time": [2, 3, 4]})

df["AverageSpeed"] = df["Distance"] / df["Time"]

print(df)
```

The new feature may express a useful relationship more directly than the original variables.

## 11.3 Feature scaling

Feature scaling adjusts the numerical scale of input variables.

Suppose a model receives two features:

- Age: values between 18 and 70.
- Annual income: values between 20,000 and 500,000.

Some algorithms are sensitive to these differences in magnitude. Scaling can prevent features with large numerical ranges from disproportionately influencing distance calculations or optimization.

Common methods include min-max normalization and standardization.

```python
from sklearn.preprocessing import MinMaxScalerscaler = MinMaxScaler()scaled_data = scaler.fit_transform(    df[["Distance", "Time"]])
```

Scaling is particularly important for methods such as K-Means, nearest-neighbor algorithms, and many gradient-based models. Tree-based models generally do not require scaling in the same way.

## 11.4 Feature selection

Feature selection means choosing a useful subset of the available input variables.

A dataset may contain many features, including irrelevant, redundant, or noisy variables. Keeping all of them can increase computational costs and sometimes reduce generalization performance.

Common approaches include:

- Filter methods: Select features using statistical measures, correlations, or other criteria independent of a specific predictive model.
- Wrapper methods: Evaluate feature subsets by repeatedly training and assessing a model.
- Embedded methods: Perform feature selection as part of model training, such as through certain regularization methods.

Example: Select numerical columns before analysis.

```python
numeric_features = df.select_dtypes(    include="number")

print(numeric_features.columns)
```

This identifies numerical columns, although it does not automatically determine which ones are useful.

Feature selection should be based on training data and appropriate validation procedures to avoid leakage from the test set.

## 11.5 Binning and quantization

Binning, also called quantization in this context, converts continuous numerical values into discrete intervals or categories.

For example, a customer's age could be divided into:

- 0–17: Minor
- 18–35: Young adult
- 36–60: Middle-aged
- 61 and above: Senior

Instead of representing age only as an exact numerical value, the model can use the age group as a feature.

### Why binning is useful

- It can simplify complex numerical distributions.
- It can make some nonlinear relationships easier to represent.
- It can reduce the influence of small variations within a range.
- It can improve interpretability when intervals have meaningful definitions.

However, binning also loses information. Two people aged 20 and 34 would belong to the same group even though their exact ages differ.

### Fixed-width binning

Fixed-width binning divides a numerical range into intervals of equal width.

```python
import pandas as pd

df = pd.DataFrame({    "Age": [12, 19, 27, 42, 65, 75]})

df["AgeGroup"] = pd.cut(    df["Age"],    bins=[0, 18, 35, 60, 100],    labels=["Minor", "Young Adult", "Middle-aged", "Senior"],    include_lowest=True)

print(df)
```

This creates groups using specified boundaries. The resulting bins are not necessarily equal in the number of observations.

### Quantile-based binning

Quantile-based binning attempts to divide observations into groups containing approximately equal numbers of records.

```python
df["AgeQuantile"] = pd.qcut(    df["Age"],    q=3,    duplicates="drop")
```

This can be useful when data is highly skewed, although tied values may prevent perfectly equal-sized groups.

## 11.6 Interaction features

An interaction feature represents a combined effect of two or more input variables.

Sometimes, the relationship between a target and one feature depends on the value of another feature.

For example, the effect of study hours on examination marks may differ depending on attendance.

An interaction model can be expressed as:

$$
 y=c_0+c_1x_1+c_2x_2+c_{12}(x_1x_2)
$$

Where:

- $x_1$ and $x_2$ are input features.
- $c_0$ is the intercept.
- $c_1$ and $c_2$ represent individual feature effects.
- $c_{12}$ represents the interaction effect.

The product $x_1x_2$ is a new feature.

### Python example

```python
import pandas as pdfrom sklearn.preprocessing
import PolynomialFeaturesX = pd.DataFrame({    "StudyHours": [2, 4, 6],    "Attendance": [60, 80, 95]})poly = PolynomialFeatures(    degree=2,    interaction_only=True,    include_bias=False)X_new = poly.fit_transform(X)

print(poly.get_feature_names_out(X.columns))

print(X_new)
```

The generated features include the original columns and their interaction:

```
StudyHours
Attendance
StudyHours Attendance
```

`interaction_only=True` prevents squared powers of individual features from being generated, while retaining interaction terms between distinct features.

Interaction features can help linear models represent relationships that would otherwise be difficult to capture.

# Lecture 12: Feature Engineering — Statistical Transformations

## 12.1 Understanding skewness

A probability distribution describes how values are distributed across a variable's possible range.

A symmetric distribution has approximately balanced left and right sides. A skewed distribution has a longer or heavier tail on one side.

### Right-skewed distribution

A right-skewed distribution has a long tail toward larger values. Many observations may be relatively small, while a few observations are extremely large.

Examples include:

- Personal income
- Property prices
- Website views
- Transaction amounts

### Left-skewed distribution

A left-skewed distribution has a longer tail toward smaller values. Many observations may be relatively large, with fewer much smaller values.

Skewness matters because extreme values can affect some statistical summaries and modeling methods.

## 12.2 Why transform numerical data?

Statistical transformations change the numerical representation of a variable while attempting to preserve useful information.

Transformations can:

- Compress large values.
- Reduce right skewness.
- Make variance more stable.
- Help some models capture relationships more effectively.
- Make certain statistical assumptions more plausible.

However, a transformation does not guarantee that the resulting distribution will be Gaussian or normally distributed.

## 12.3 Log transformation

A logarithmic transformation compresses large values more strongly than small values.

For a logarithm with base $b$:

$$
 y=\log_b(x)
$$

Common choices are:

- Natural logarithm: $\ln(x)$, base $e$.
- Common logarithm: $\log_{10}(x)$, base 10.

For example, the natural logarithm of 10,000 is much smaller than 10,000 itself. This compression can reduce the impact of very large observations.

### Python example

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({    "Views": [10, 100, 1000, 10000]})

df["LogViews"] = np.log1p(df["Views"])

print(df)
```

`np.log1p(x)` calculates $\ln(1+x)$. It is often convenient for nonnegative data containing zeros because the transformation remains defined at zero.

For strictly positive values, `np.log(x)` can be used directly.

### Advantages

- Compresses large numerical values.
- Can reduce right skewness.
- Can make multiplicative relationships easier to model.

### Limitations

- The ordinary logarithm is undefined for zero and negative values.
- Log transformation may not help every distribution.
- Predictions and coefficients can require careful interpretation when the target has been transformed.

## 12.4 Box-Cox transformation

The Box-Cox transformation is a family of power transformations controlled by a parameter $\lambda$.

It is defined for positive values $x>0$ as:

$$
 y^{(\lambda)}= \begin{cases} \dfrac{x^\lambda-1}{\lambda}, & \lambda\ne0\\[6pt] \ln(x), & \lambda=0 \end{cases}
$$

The parameter $\lambda$ determines the transformation applied to the data.

Different values of $\lambda$ correspond to different transformations. When $\lambda=1$, the result is $x-1$, which is essentially a shift by a constant. When $\lambda=0$, the transformation becomes the natural logarithm.

An optimal value of $\lambda$ can be estimated statistically, commonly using maximum likelihood under a distributional model.

### Python example

```python
import numpy as npfrom scipy.stats
import boxcoxdata = np.array([10, 20, 50, 100, 500])

transformed_data, estimated_lambda = boxcox(data)

print("Estimated lambda:", estimated_lambda)

print("Transformed data:", transformed_data)
```

The function returns the transformed values and the estimated parameter.

### Important conditions

- Standard Box-Cox requires strictly positive input values.
- If the data contains zeros or negative values, an alternative transformation or an appropriate shift may be needed.
- The transformation may reduce skewness, but it does not guarantee a perfectly normal distribution.

For machine learning, transformation parameters should be learned from the training data and then applied consistently to new observations.

# Lecture 13: Feature Engineering — Categorical Data

## 13.1 What is categorical data?

Categorical data represents discrete groups, labels, or classes rather than measurements on a continuous numerical scale.

Examples include:

- Gender categories
- City names
- Product types
- Education levels
- Customer membership types
- Academic grades

Categorical features are commonly divided into nominal and ordinal variables.

## 13.2 Nominal features

Nominal features have categories without an inherent ranking.

Examples:

- City: Lahore, Karachi, Islamabad
- Color: Red, Blue, Green
- Department: IT, HR, Finance
- Animal type: Cat, Dog, Zebra

There is no natural numerical order in which these categories should be ranked.

### One-hot encoding

One-hot encoding creates a separate binary column for each category.

Suppose the original feature is:

| Color |
| ----- |
| Red   |
| Blue  |
| Green |
| Red   |

After one-hot encoding:

| Color_Red | Color_Blue | Color_Green |
| --------- | ---------- | ----------- |
| 1         | 0          | 0           |
| 0         | 1          | 0           |
| 0         | 0          | 1           |
| 1         | 0          | 0           |

Each row receives a 1 for the category it belongs to and 0 for the other categories.

### Python example using `pd.get_dummies()`

```python
import pandas as pd

df = pd.DataFrame({    "Color": ["Red", "Blue", "Green", "Red"]})encoded_df = pd.get_dummies(    df,    columns=["Color"],    dtype=int)

print(encoded_df)
```

### Python example using `OneHotEncoder`

```python
import pandas as pdfrom sklearn.preprocessing
import OneHotEncoderdf = pd.DataFrame({    "Color": ["Red", "Blue", "Green", "Red"]})encoder = OneHotEncoder(    handle_unknown="ignore",    sparse_output=False)

encoded = encoder.fit_transform(df[["Color"]])

print(encoder.get_feature_names_out(["Color"]))

print(encoded)
```

`OneHotEncoder` learns the categories during fitting and can apply the same encoding to later data. `handle_unknown="ignore"` prevents errors when an unseen category occurs, although an unseen category is represented by zeros across the encoded columns.

The `sparse_output` parameter is used in modern scikit-learn versions; older versions use `sparse=False`.

## 13.3 Ordinal features

Ordinal features have categories with a meaningful order.

Examples:

- Small, Medium, Large
- Beginner, Intermediate, Advanced
- Low, Medium, High
- Poor, Fair, Good, Excellent

Unlike nominal categories, ordinal categories have an order, but the distance between consecutive categories is not necessarily equal.

### Ordinal encoding

Ordinal encoding assigns numerical values according to the category order.

Example:

```python
import pandas as pd

df = pd.DataFrame({    "Size": ["Small", "Medium", "Large", "Small"]})size_mapping = {    "Small": 1,    "Medium": 2,    "Large": 3}df["SizeCode"] = df["Size"].map(size_mapping)

print(df)
```

This maps Small to 1, Medium to 2, and Large to 3.

For ordinal data, the mapping should follow the actual order of the categories rather than alphabetical order.

### Nominal vs. ordinal comparison

| Property                   | Nominal             | Ordinal            |
| -------------------------- | ------------------- | ------------------ |
| Natural order              | No                  | Yes                |
| Example                    | City name           | Education level    |
| Typical encoding           | One-hot encoding    | Ordinal encoding   |
| Meaning of numerical codes | Category indicators | Ordered categories |

Important: Encoding ordinal categories as integers preserves order, but it can imply equal spacing between categories. If the distance between levels is not meaningful, choose a method appropriate for the model and the data.

# Lecture 14: Feature Engineering — Text Data

## 14.1 What is text feature engineering?

Text feature engineering converts unstructured language into numerical representations that machine learning algorithms can process.

Text data appears in:

- Emails
- Product reviews
- News articles
- Social media posts
- Customer support messages
- Research papers
- Chat conversations

Raw text cannot usually be passed directly to conventional numerical machine learning algorithms. It must first be cleaned and represented using suitable features.

## 14.2 Text preprocessing

Text preprocessing prepares documents for further analysis. The appropriate steps depend on the task and language.

### A. Lowercasing

Lowercasing converts uppercase characters to lowercase so words differing only in capitalization can be treated consistently.

Example:

```
Machine Learning
machine learning
MACHINE LEARNING
```

After lowercasing:

```
machine learning
machine learning
machine learning
```

Python example:

```python
text = "Machine Learning Is Useful"clean_text = text.lower()

print(clean_text)
```

Output:

```
machine learning is useful
```

Lowercasing is useful for many text classification tasks, although capitalization can be meaningful for some tasks, such as named-entity recognition.

### B. Tokenization

Tokenization divides text into smaller units called tokens. Depending on the tokenizer, these may be words, subwords, or characters.

Example:

```
"Machine learning is useful"
```

A simple word tokenization might produce:

```
["Machine", "learning", "is", "useful"]
```

Tokens are the basic units used by many text processing methods.

### C. Stop-word removal

Stop words are common words that may contribute relatively little information for some tasks, such as "the," "is," and "and."

Removing them can reduce the size of a vocabulary and emphasize more informative terms.

However, stop-word removal is not always appropriate. Words such as "not" can be essential for understanding sentiment or meaning.

## 14.3 Bag of Words (BoW)

The Bag of Words method represents a document by counting the occurrences of words from a vocabulary.

It ignores most grammatical structure and word order, treating a document primarily as a collection of word counts.

Consider these two documents:

- Document 1: `"cats chase mice"`
- Document 2: `"dogs chase cats"`

The combined vocabulary is:

```
cats, chase, dogs, mice
```

The documents can be represented as vectors:

| Document   | cats | chase | dogs | mice |
| ---------- | ---- | ----- | ---- | ---- |
| Document 1 | 1    | 1     | 0    | 1    |
| Document 2 | 1    | 1     | 1    | 0    |

Each value indicates the number of times a vocabulary term appears in the document.

### Python example

```python
from sklearn.feature_extraction.text import CountVectorizerdocuments = [    "cats chase mice",    "dogs chase cats"]

vectorizer = CountVectorizer()X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())

print(X.toarray())
```

`fit_transform()` learns the vocabulary and creates a document-term matrix.

### Advantages

- Simple and easy to understand.
- Efficient for many traditional text classification tasks.
- Captures word frequency.

### Limitations

- Ignores most word-order information.
- Usually produces high-dimensional sparse vectors.
- Does not directly capture semantic similarity between words.

## 14.4 Bag of N-Grams

N-grams are sequences of $N$ consecutive tokens.

Common examples include:

- Unigrams: Single words.
- Bigrams: Pairs of consecutive words.
- Trigrams: Groups of three consecutive words.

For the sentence:

```
machine learning is useful
```

Unigrams include:

```
machine, learning, is, useful
```

Bigrams include:

```
machine learning
learning is
is useful
```

Trigrams include:

```
machine learning is
learning is useful
```

N-grams preserve some local word-order information, unlike a basic unigram Bag of Words representation.

For example, the bigrams in "not good" and "good" differ, which may help a sentiment classifier identify certain expressions.

### Python example

```python
from sklearn.feature_extraction.text import CountVectorizerdocuments = [    "machine learning is useful",    "machine learning is powerful"]

vectorizer = CountVectorizer(    ngram_range=(1, 2))X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())

print(X.toarray())
```

`ngram_range=(1, 2)` generates both unigrams and bigrams.

Using more n-grams can capture richer context, but it also increases vocabulary size, memory usage, and the risk of learning patterns that occur too rarely.

## 14.5 TF-IDF: Term Frequency–Inverse Document Frequency

TF-IDF represents words according to how frequently they appear in a document and how informative they are across a collection of documents.

The general relationship is:

$$
 \operatorname{TFIDF}(t,d,D) = \operatorname{TF}(t,d) \times \operatorname{IDF}(t,D)
$$

Where:

- $t$ is a term.
- $d$ is a document.
- $D$ is the collection of documents.
- TF measures the term's frequency within a document.
- IDF reduces the weight of terms that appear in many documents.

A word that appears repeatedly in one document but rarely across the whole collection may receive a high TF-IDF weight. A common word that appears in nearly every document typically receives a lower IDF weight.

### Why TF-IDF is useful

Suppose a collection contains many product reviews. Words such as "the" and "is" occur frequently but may not distinguish one review from another. Terms such as "battery," "durability," or "overheating" may be more useful for distinguishing topics.

TF-IDF reduces the influence of very common terms and emphasizes terms that are more distinctive within individual documents.

### Python example

```python
from sklearn.feature_extraction.text import TfidfVectorizerdocuments = [    "the phone has a good battery",    "the battery lasts long",    "the phone has a bright screen"]

vectorizer = TfidfVectorizer()X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())

print(X.toarray())
```

The result is a numerical matrix whose entries represent TF-IDF weights rather than raw counts.

The exact values depend on the vectorizer's term-frequency and inverse-document-frequency conventions.

Important: Fit the vectorizer on training documents only, then use the fitted vectorizer to transform validation, test, and future documents. Learning vocabulary or IDF statistics from test documents can cause data leakage.

## 14.6 Word embeddings

Word embeddings represent words as dense numerical vectors in a continuous space.

Unlike traditional Bag of Words and TF-IDF methods, embeddings can encode patterns of semantic or contextual similarity.

For example, a well-trained embedding may represent "king" and "queen" as more related than "king" and "banana," depending on the training data and model.

### Word2Vec

Word2Vec is a family of methods that learns word vectors from the contexts in which words occur.

Two common approaches are:

- CBOW (Continuous Bag of Words): Predicts a target word from its surrounding context.
- Skip-gram: Predicts surrounding context words from a target word.

Word2Vec learns numerical representations that can capture useful relationships between words.

### Comparison of text representations

| Method       | Representation           | Main strength                           |
| ------------ | ------------------------ | --------------------------------------- |
| Bag of Words | Sparse word-count vector | Simple frequency representation         |
| N-grams      | Counts of word sequences | Captures local word order               |
| TF-IDF       | Weighted sparse vector   | Emphasizes distinctive terms            |
| Word2Vec     | Dense word vectors       | Captures learned semantic relationships |

Word2Vec typically creates a vector for each word. More advanced contextual language models can generate different representations for the same word depending on the sentence in which it appears.

# Lecture 15: Feature Engineering — Temporal Data

## 15.1 What is temporal data?

Temporal data is data associated with dates or times. It is common in financial transactions, website traffic, sensor measurements, medical records, and sales forecasting.

A timestamp can contain useful information beyond the original date and time.

For example, sales may vary according to:

- The month of the year
- The day of the week
- The time of day
- Weekends and holidays
- Seasonal periods

Feature engineering extracts these patterns into separate variables that can be used by a machine learning model.

## 15.2 Date-based feature extraction

Date-based extraction creates new features from a date or timestamp.

Common features include:

- Year
- Month
- Day of the month
- Day of the week
- Day of the year
- Quarter
- Weekend indicator

### Python example

```python
import pandas as pd

df = pd.DataFrame({    "Date": [        "2026-01-05",        "2026-02-14",        "2026-07-20"    ]})

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.yeardf["Month"] = df["Date"].dt.monthdf["Day"] = df["Date"].dt.daydf["DayOfWeek"] = df["Date"].dt.dayofweekdf["DayOfYear"] = df["Date"].dt.dayofyeardf["Quarter"] = df["Date"].dt.quarterdf["IsWeekend"] = (    df["Date"].dt.dayofweek >= 5)

print(df)
```

The `.dt` accessor provides date and time properties for pandas datetime Series.

For `dayofweek`, Monday is 0 and Sunday is 6. Therefore, values 5 and 6 correspond to Saturday and Sunday.

## 15.3 Time-based feature extraction

A timestamp may include information about the hour, minute, second, microsecond, and time zone.

These features can be useful when behavior changes throughout the day.

Example:

```python
import pandas as pd

df = pd.DataFrame({    "Timestamp": [        "2026-10-22 08:30:15",        "2026-10-22 14:45:20",        "2026-10-22 21:10:05"    ]})

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["Hour"] = df["Timestamp"].dt.hourdf["Minute"] = df["Timestamp"].dt.minutedf["Second"] = df["Timestamp"].dt.secondprint(df)
```

For timezone-aware data, timestamps should be handled consistently, especially when records come from different geographic locations.

### UTC offset

UTC offset describes the difference between a local time zone and Coordinated Universal Time (UTC).

A timezone-aware timestamp can be converted to UTC or another time zone before features are extracted. This prevents records from different time zones from being compared as though they used the same clock.

## 15.4 Temporal binning

Temporal binning groups specific times into broader periods.

For example:

- Morning: 05:00–11:59
- Afternoon: 12:00–16:59
- Evening: 17:00–20:59
- Night: 21:00–04:59

These boundaries are illustrative and can be changed according to the application.

### Python example

```python
import pandas as pd

df = pd.DataFrame({    "Hour": [7, 10, 13, 18, 22, 2]})def get_period(hour):    if 5 <= hour < 12:        return "Morning"    elif 12 <= hour < 17:        return "Afternoon"    elif 17 <= hour < 21:        return "Evening"    else:        return "Night"df["TimePeriod"] = df["Hour"].apply(get_period)

print(df)
```

The hour 2 belongs to the Night category because it falls outside the other intervals.

Temporal binning can make results easier to interpret and can capture broad behavioral patterns, but it loses detailed time information.

## 15.5 Temporal feature engineering in time-series modeling

Time-series data is ordered in time, so the order of observations matters.

Additional useful features may include:

- Previous-period values, known as lag features
- Rolling averages
- Days since a particular event
- Time since the previous transaction
- Seasonal indicators

For example, a lag feature can represent the previous day's sales.

```python
import pandas as pd

df = pd.DataFrame({    "Date": pd.date_range(        "2026-01-01",        periods=5,        freq="D"    ),    "Sales": [100, 120, 90, 150, 130]})

df["PreviousDaySales"] = df["Sales"].shift(1)

print(df)
```

The first row has no previous-day sales value because no earlier observation exists in the supplied dataset.

Important: In forecasting, data must remain in chronological order. Lag features, rolling calculations, and training/test splits must not use future information to predict the past.

# Lecture 16: Feature Engineering — Image Data

## 16.1 What is image feature engineering?

Image feature engineering converts image information into numerical features that can be analyzed by machine learning algorithms.

A digital image is represented as an array of pixel values. Each pixel encodes brightness or color information.

Image feature engineering can extract properties such as:

- Pixel intensity
- Color distribution
- Texture
- Shape
- Edges
- Statistical summaries

These features can help distinguish objects, identify patterns, or characterize images without necessarily processing every pixel directly in a predictive model.

## 16.2 Common tools for image processing

Your notes identify three image-processing libraries.

### A. scikit-image

`scikit-image`, imported as `skimage`, provides tools for image transformations, filtering, segmentation, measurement, and feature extraction.

It integrates well with NumPy and scientific Python workflows.

### B. Mahotas

Mahotas is a computer vision and image-processing library that provides functions for tasks such as thresholding, morphology, and texture-related analysis.

### C. OpenCV

OpenCV is a widely used computer vision library. It supports image loading, resizing, color-space conversion, filtering, feature detection, and many other image-processing tasks.

Example of loading an image with OpenCV:

```python
import cv2image = cv2.imread("image.jpg")if image is None:    raise FileNotFoundError("Could not load image.jpg")

print(image.shape)
```

For a typical color image, the shape is:

```
(height, width, channels)
```

OpenCV loads color images in BGR channel order by default rather than RGB.

## 16.3 Grayscale conversion

A color image commonly has three channels: red, green, and blue. A grayscale image represents brightness using a single intensity value per pixel.

Converting a color image to grayscale reduces the channel dimension from three values per pixel to one.

A commonly used luminance approximation is:

$$
 Y=0.299R+0.587G+0.114B
$$

Where:

- $R$ is the red intensity.
- $G$ is the green intensity.
- $B$ is the blue intensity.
- $Y$ is the grayscale intensity estimate.

The exact conversion can depend on the color space and library used.

### Python example using OpenCV

```python
import cv2image = cv2.imread("image.jpg")if image is None:    raise FileNotFoundError("Could not load image.jpg")

gray = cv2.cvtColor(    image,    cv2.COLOR_BGR2GRAY)

print("Original shape:", image.shape)

print("Grayscale shape:", gray.shape)
```

The original image has three color channels, while the grayscale result has one intensity value per pixel.

### Why grayscale conversion is useful

- Reduces the number of channels.
- Simplifies brightness-based analysis.
- Can reduce computational requirements.
- Is useful for many edge-detection and texture-analysis tasks.

However, grayscale conversion removes color information. It may be inappropriate when color itself is an important feature, such as distinguishing ripe and unripe fruit by color.

## 16.4 Intensity histograms

An intensity histogram summarizes how frequently different pixel brightness values occur in an image.

For an 8-bit grayscale image, pixel intensities typically range from 0 to 255:

- 0 represents black.
- 255 represents white.
- Intermediate values represent shades of gray.

A histogram divides the intensity range into bins and counts the number of pixels falling into each bin.

For example, a histogram may reveal whether an image is predominantly dark, predominantly bright, or has a wide range of intensities.

### Python example

```python
import cv2
import numpy as np

image = cv2.imread("image.jpg")if image is None:    raise FileNotFoundError("Could not load image.jpg")

gray = cv2.cvtColor(    image,    cv2.COLOR_BGR2GRAY)

histogram = cv2.calcHist(    [gray],    [0],    None,    [256],    [0, 256])

print("Histogram shape:", histogram.shape)

print("Total pixel count:", histogram.sum())
```

The histogram contains 256 bins, one for each possible intensity value.

The sum of the histogram counts should equal the number of pixels in the grayscale image.

### Quantization

Quantization reduces the number of intensity levels by grouping similar values into bins.

Instead of treating every possible pixel intensity as a separate level, an image-processing pipeline may group intensities into broader ranges.

This can simplify the representation, although some fine detail is lost.

## 16.5 Image aggregation statistics

Image aggregation statistics summarize numerical properties of pixel values.

They can be calculated for grayscale images or separately for each color channel.

### A. Mean

The mean is the average pixel intensity:

$$
 \mu=\frac{1}{N}\sum_{i=1}^{N}x_i
$$

It provides a basic measure of overall brightness for a grayscale image.

### B. Median

The median is the middle pixel intensity after the values are sorted.

It is less sensitive to extreme values than the mean.

### C. Variance

Variance measures how much pixel intensities vary around the mean.

$$
 \sigma^2=\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2
$$

A low variance may indicate that most pixels have similar intensities, while a high variance indicates a wider spread of brightness values.

### D. Skewness

Skewness describes the asymmetry of a distribution.

For pixel intensities, it can help characterize whether values are concentrated toward lower or higher intensities, relative to the distribution's tail.

### E. Kurtosis

Kurtosis describes aspects of the distribution's tail weight and concentration relative to its mean and variance. Its exact numerical interpretation depends on the convention used by the statistical library.

These statistics provide compact features for some image classification and analysis tasks, although they do not preserve spatial arrangement.

### Python example using NumPy and SciPy

```python
import cv2
import numpy as npfrom scipy.stats import skew, kurtosisimage = cv2.imread("image.jpg")if image is None:    raise FileNotFoundError("Could not load image.jpg")

gray = cv2.cvtColor(    image,    cv2.COLOR_BGR2GRAY)

pixels = gray.astype(float).ravel()mean_value = np.mean(pixels)median_value = np.median(pixels)variance_value = np.var(pixels)skewness_value = skew(pixels)kurtosis_value = kurtosis(pixels)

print("Mean:", mean_value)

print("Median:", median_value)

print("Variance:", variance_value)

print("Skewness:", skewness_value)

print("Kurtosis:", kurtosis_value)
```

The `.ravel()` method converts the two-dimensional grayscale image into a one-dimensional array of pixel intensities.

`np.var()` calculates population variance by default. `scipy.stats.kurtosis()` uses excess kurtosis by default, so a normal distribution has a kurtosis value of approximately zero under that convention.

## 16.6 Channel-wise image statistics

For color images, statistical features can be computed separately for each color channel.

For example, a color image can be represented by:

- Mean red-channel intensity
- Mean green-channel intensity
- Mean blue-channel intensity
- Variance of each channel

### Python example

```python
import cv2
import numpy as np

image = cv2.imread("image.jpg")if image is None:    raise FileNotFoundError("Could not load image.jpg")# OpenCV uses BGR channel order.channel_names = ["Blue", "Green", "Red"]for index, name in enumerate(channel_names):    channel = image[:, :, index].astype(float)    print(        name,        "Mean:", np.mean(channel),        "Variance:", np.var(channel)    )
```

Channel-wise statistics retain some color information, unlike grayscale-only statistics. However, they still summarize the image globally and do not identify where specific colors occur.

# Final Revision Notes: Connecting All Four Modules

The central idea of these lectures is that building a machine learning system involves much more than choosing an algorithm. The quality and representation of the input data are fundamental to the quality of the resulting model.

| Topic                        | Main purpose                              | Example                              |
| ---------------------------- | ----------------------------------------- | ------------------------------------ |
| Machine learning foundations | Understand how models learn               | Classification and regression        |
| ML taxonomy                  | Choose an appropriate learning approach   | Supervised vs. unsupervised learning |
| Data acquisition             | Collect usable data                       | Reading CSV and TSV files            |
| Structural inspection        | Understand data shape and types           | `shape`, `dtypes`, `head()`          |
| Data wrangling               | Clean and transform raw records           | Filtering and type casting           |
| Missing-value handling       | Address incomplete observations           | Mean, median, forward fill           |
| Data summarization           | Understand distributions and groups       | `groupby()`, `value_counts()`        |
| Numeric feature engineering  | Create useful numerical representations   | Binning and interaction features     |
| Statistical transformations  | Change numerical distributions            | Log and Box-Cox transformations      |
| Categorical engineering      | Encode category information               | One-hot and ordinal encoding         |
| Text engineering             | Convert documents into numerical features | BoW, N-grams, TF-IDF, Word2Vec       |
| Temporal engineering         | Extract time-related patterns             | Month, weekday, time bins            |
| Image engineering            | Extract numerical image properties        | Grayscale, histograms, statistics    |
| Model deployment             | Reuse trained preprocessing and models    | `joblib` persistence                 |

## Important practical principles

1. Understand before transforming. Inspect the data before deciding how to clean or encode it.
2. Choose preprocessing according to the data. A strategy appropriate for numerical measurements may be unsuitable for categories, text, dates, or images.
3. Keep training and prediction consistent. Use the same fitted scalers, encoders, feature definitions, and column order.
4. Avoid data leakage. Fit data-dependent transformations on training data only, then apply them to validation and test data.
5. Do not discard information unnecessarily. Binning, feature selection, grayscale conversion, and text preprocessing can simplify data but may remove useful information.
6. Evaluate the result. A transformation is useful when it supports the task, improves appropriate evaluation measures, or makes the model more reliable or interpretable.

These notes cover the topics present in your uploaded document, with expanded explanations and practical Python examples.&#x20;

Pasted text.txt
