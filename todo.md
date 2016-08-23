## Items appended as thought of

[x] Select topic
[x] Find Dataset (http://www.fema.gov/media-library-data/20130726-2126-31471-8394/nfirs_2011_120612.zip)
[x] install and play with tensorflow
[x] find dbf reader and explore data
[x] implement example tensorflow Regression for practice
[x] write introduction and explanation of firefighting dataset
[x] figure out how fire incidents and basic incidents are linked (fd ids and incident numbers)
[x] get indexes on sqlite tables to make them easier to query quickly
[x] update sqlite script to skip steps if sqlite files already exist
[x] provide code lookups from code DBF file for reference
[x] write data exploration section in project_report.md
[x] produce joined dataset of just incidents that are in both the fire incident file and the basic incident file
[x] produce series of scatter plots tying suspect features to property loss
[x] make a list of chosen features to train the model on
[x] fill in missing values with median values for numeric fields
[x] remove outliers from numeric fields
[x] use feature scaling on numeric fields
[x] switch all categories into 1-hot encoded vectors
[x] produce smaller dataset containing just features we care about for just relevant incidents
[x] split off a test set that will not be used during validation tests,
so that validation information doesn't bleed back into the model and overfit.
(30,000 datapoints should do it)
[x] spot check a series of common algorithms to see which approaches might be useful
[x] take some feature exploration graphs and import them into Exploratory
visualizations in the project report
[x] perform parameter tuning for Random Forest, Bagging, and GradientBoost regressions
[x] Write up algorithms and Techniques section
[x] line out data preprocessing
scripts in the Data Preprocessing Section of the writeup
[x] rework last transformation to normalize to log of residual and re-model/tune
[x] write up benchmarks
[x] write up implementation section
[x] write up refinement (cover outlier removal, feature scaling, logarithmic
  discovery, and model tuning)
[x] Check error rate against validation set
[x] Write up model validation and evaluation section
[x] write up justification with comparison to benchmarks
[x] produce a visualization of the predicted values vs the validation set
[x] write up free-form visualization section
[x] write up project Reflection
[x] Identify and document possible improvements to this model (particularly
   with respect to use cases)
[x] build a function that accepts un-scaled inputs and produces a un-logrithm'd
prediction value
[x]  compare un-logrithm'd
input labels and output labels
[x] produce a pdf report

### Post Review Round 1

[x] Include RMSE in metrics section (and calculate throughout model justification)
[x] Define R-squared metric in the metrics section
[x] include a MSE equation directly in the project report
[x] clarify why only 2011 data was included in the data exploration section
[x] add axes labels for data exploration visualizations
[ ] add an example for the ordered pairs benchmark
[ ] use a table to organize the various model exploration results
[x] include RMSE to minimize interpretation issues in the Results section
