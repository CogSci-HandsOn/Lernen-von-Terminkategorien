# Lernen von Terminkategorien

A project that tries to classify calendar events by type.

The only data available for classification is date, start time and end time. These data points are [mapped](res/label_mapping.json) to one or multiple labels out of eight possible. 

We used the scarce features to create many more by using week days, holidays, years, months, etc.
With the preprocessed data we tried some different supervised machine learning algorithms. 
Finally we failed. Probably the data wasn't sufficient. 
This project is still in progress though whenever someone finds time.