# NFIRS Property Loss Prediction Model
## Machine Learning Engineer Nanodegree
Ethan C Vizitei
August 31, 2016

## I. Firefighting and property damage

### Overview

The National Fire Incident Reporting System (NFIRS) is both a data set and a
series of software systems built around collecting data on fires in the US
and analyzing it for many purposes (national initiative effectiveness, risk
management, etc).  It was established by the USFA (United States Fire Administration)
and is freely available to the public.  Many firefighting agencies in the United
States provide data on each incident they respond to into this national system.

I personally have worked with NFIRS before on the reporting side; as a firefighter
and later an officer in the Boone County Fire Protection District, all of the
reports I wrote and some of the ad-hoc software I implemented interfaced with
the ingestion end of the NFIRS database.  Some of the interesting information
available in it is a series of categorization fields for each incident
and the total resulting property damage.

This loss data is of particular interest because when prioritizing incidents
and when doing incident preplanning, having comparative predictions of actual
likely property loss for a given structure or residence and incident type
could be very valuable to a dispatch center, insurance agency, or first due fire
officer for a response area.

### Example Use Cases for a predictive property loss algorithm:

* During a large natural disaster when many fires are reported simultaneously in real time,
a dispatch agency may choose to prioritize response to incidents (or proactively increase
the default response level to a given incident) based upon predicted property damages.

* When performing a risk analysis, an insurance agent may wish to augment their
actuarial tables with predictive property loss for this particular structure
given a range of potential incidents, thus more finely calibrating the premium
to charge for a given range of coverage.

* Officers at a fire station often choose which structures in their runbox
to do detailed preplanning on based on perceived risk. Having predicted property
loss for a range of structures could help inform priority decisions for the
most important structures to perform detailed planning exercises for.

For this project I will be using just the NFIRS data from 2011 (which is
still a very large dataset), but in a production system this dataset would
benefit from being continually adjusted for inflation and augmented
in an ongoing manner with additional data from each year as data is added to
the NFIRS database.

- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_

### Intended Approach

For this project I will be taking the raw dbase files for the 2011
NFIRS data.  The dataset is very big (2.2 million basic records)
and we don't actually need all of it as some of those incidents
are not fires (Medical, hazmat, search and rescue, etc).

I intend to reduce the set of data under consideration to just
structural fire incidents, and extract/normalize just the features
that are relevant to predicting property loss.  Then I intend
to use the tensorflow framework to train a regression model
such that given a set of features for a target structure/incident,
we can make a reasonable prediction of likely real property loss.

*Possible Example Usecase:*
  A dispatch center receives a 911 call reporting a single-room fire
  inside a residential single-family dwelling at a given address.  
  That address is used by supporting software systems to query
  dimensions of the structure.  Questions from the dispatcher are used
  to establish what materials are on site and what the expected
  ignition source was, whether a fire detector has sounded, etc
  (standard questions currently asked in 911 calls during dispatch).  
  The data is fed into the model and produces an expected dollar value
  of $237,500.00 property loss from this incident.  Based on the
  unexpectedly high property loss estimate, a second alarm (
  extra fire trucks and personnel) are immediately dispatched rather
  than waiting for the first engine to arrive and size up the scene.

- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

### Cost function and Validation

To train the algorithm, I'll be using Mean Squared Error as
the cost function and trying to minimize it to find the regression
of best fit.  In layman's terms, this is taking the sum of the
squared differences between each real y value and the predicted y value
for the current function, divided by the number of elements (the
average of the squared error in prediction).  This is formally
written as:

  ![MSE][https://wikimedia.org/api/rest_v1/media/math/render/svg/67b9ac7353c6a2710e35180238efe54faf4d9c15]

Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration

The NFIRS data arrives in a set of tables that can be joined together.
The "basicincident.dbf" file has fields that apply to all incidents,
and then there are other incident fields which are different per type
of incident being described (basicaid, arson, fireincident,
  hazmat, wildlands, etc).  For the purposes of this analysis, we're looking
at fire incidents and the property loss they cause, so we really only
need the basicincident.dbf file and the fireincident.dbf file.

There are over 2 million records in the basicincident file, but only about 600k
have a record in the fireincident table.  We'll further select out
those incidents which result in no damage (there are many) and
those that are not of a "fire" incident type (even in fire incidents, there
  are things like false alarms which are common), so our final
data set is actually closer to 200k records.

The target column we'll be looking at is "PROP_LOSS", which represents the
dollar value loss of property destroyed in a given emergency incident.  After
trimming outliers (there are several high-dollar value losses that really skew
  the data over all), here's what the distribution looks like:

| stat  | PROP_LOSS $ value |
| count |  196574.000000    |
| mean  |    8914.580748    |
| std   |   12933.652034    |
| min   |       1.000000    |
| 25%   |    1000.000000    |
| 50%   |    3000.000000    |
| 75%   |   10000.000000    |
| max   |   60000.000000    |

We can see that even after chopping off a lot of high value outliers, the data
skews heavily to the low end, with the median being $3000 even though the mean
damage is up at $8914.

After combing the fire and basic incident tables into one tabular dataset, there
are still > 130 columns to consider, many of which are not strongly related
to the output target (property loss).

Most of the fields are categorical (the type of building, the type of fire,
the type of ignition source, etc), but there are a few numeric fields
(number of firetrucks, number of response personnel, square feet of structure).

One of the challenges we have for the numeric fields is that there's plenty
of missing data.  For example, square feet.  All buildings on fire have
some square footage, but not all incident reports include it.  Out of nearly
200000 records 106645 have no square foot value included.  The rest are
distributed like this:

| field | SQ_FEET  |
| count |    87153 |
| mean  |    10291 |
| std   |   579840 |
| min   |        1 |
| 25%   |      800 |
| 50%   |     1200 |
| 75%   |     1944 |
| max   | 10000000 |

There are several fields like this, with high outliers and skewed distributions,
and a lot of missing data.  The plan for these numeric fields is to replace
missing values with the median value of the dataset to keep a lot of 0s and -1s
from throwing off the relationship, and trimming extreme outliers.

- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization

One of the first things I wanted to look at was the distribution of the
target variable (the property loss value), because many of the values
tended low and I know for regression problems it's nice to have
a normal distribution.  Here's the distribution of the raw data:

![Property Loss](images/PropLossDistribution.png)

It skews to the right pretty significantly. I experimented with trying to
perform a regression on it with several learners, and when I compared
them to running the same regression exercise on the log of that
target variable the difference in r-squared scores was a full 0.1 for
nearly every learner in favor of the log version.  Here's
the same data run through a logarithmic transform first:

![Property Loss Log](images/PropLossLogDistribution.png)

- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
