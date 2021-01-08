### AirBnB Seatle and Boston Analysis

Libraries Used:
    jupyter notebook(with Python3)<br>
    numpy<br>
    pandas<br>
    matplotlib (pyplot, ticker, style)<br>
    seaborn<br>
    warnings<br>
    DecisionTreeClassifier<br>
    RandomForestClassifier<br>
    model_selection(train_test_split, KFold, cross_val_score, GridSearchCV)<br>
    metrics (accuracy_score, confusion_matrix, f1_score)<br>


Files in the repository:
    Boston Data-zipped.zip (contains the listings-Boston.csv, calendar-Boston.csv, ratings-Boston.csv files)<br>
    Seattle Data-zipped.zip (contains the listings-Seattle.csv, calendar-Seattle.csv, ratings-Seattle.csv files<br>
    AirBnB Seattle Analysis.inpyb (provides the jupyter notebook code to run the analysis)<br>
    air_listings.csv (the cleaned listings dataset used in the notebook)<br>
    air_calendar.csv (the cleaned calendar dataset used in the notebook<br>
    air_reviews.csv (the cleaned reviews dataset used in the notebook)<br>

AirBnB Analysis:

Investigation Overview

Using AirBnB data for Seattle and Boston, this project will focus on the difference between Hosts and Superhosts and seeks to understand the answers to the following questions:

    1) What are the proportions of Hosts and Superhosts for Seattle and Boston?    <br>
    2) How do the Superhost compare to the Host for overall rating?<br>
    3) How do average number of stays compare for Superhosts and Hosts?<br>
    4) 4)	How do the number of listings compare?<br>
    5) Are Superhosts clustered into different neighbourhoods than hosts?<br>
    6) How do the daily rates compare for Hosts and Superhosts?<br>
    7) Can we use machine learning to predict which Hosts will are Superhosts?
    7) Using the data provided, what appears to be the most significant in predicting if a Host is a Superhost?<br>

As this data was not part of a randomized experiment, findings are not assumed to have causality.
Dataset.

Overview:

The dataset contains 3 files for each city (listings.csv, calendar.csv, ratings.csv).   They were downloaded from the following site:  http://insideairbnb.com/get-the-data.html

For the Seattle and Boston files, here is the relevant information about the files:

    listings.csv for each city was renamed to listings-Seattle.csv and listings-Boston.csv <br>
    calendar.csv for each city was renamed to calendar-Seattle.csv and calendar-Boston.csv <br>
    ratings.csv for each city was renamed to ratings-Seattle.csv and ratings-Boston.csv <br>

Data from both cities was combined and city name was added to each file, resulting in the following files:

    The calendar dataframe has with 5 columns and 2,702,460 rows, with data on price, availability and date.

    The listings datafame has 95 columns and 7,403 rows, with data on hosts, amenities, availabilty, price, type of host, location, type of accomodations, number of rooms and bathrooms, review scores, neighbourhoods, amenities and a number of other items.

    The reviews dataframe has 7 columns and 153,124 rows, with data on the reviewers of these properties.

 You can download the data yourself using the link above, but, you will need to rename the csv files to match the cities as described above to get the notebook to run properly.

Using a variety of data wrangling techniques, I evaluated the file, discovering a number of opportunities to improve the quality and tidyness of the data. The following issues were found with the datasets:

Quality Issues

Issues with the calendar dataframe:
    1) "available" is abbreviated as t or f. Change to True or False.<br>
    2) "price" has non-numeric characters and large number of missing values.<br>
    3) "calendar" is showing as an object and should be changed to datetime. We can also consider making datetime as an index to allow for time series analysis.<br>
    4) Since availability is = t when price is NaN (meaning available can be true when price is NaN, we will set null price values to 0.

Issues with listings dataframe:
    1) date_last_scraped should be datetime.<br>
    2) host_response_time, host_response_rate, host_acceptance_rate, host_listings_count, host_total_listings_count, price, weekly_price, monthly_price, security_deposit, cleaning_fee, extra_people, calculated_host_listings_count are all showing as objects and should be converted to numbers<br>
    3) There are a number of variables in the listing file with missing values. These will need to be addressed during cleaning.
    Missing Value Issues with listings dataframe requiring cleaning<br>
        * access has 72% missing values<br>
        * house_rules has 68% missing values<br>
        * missing monthly_price values should be set to 0<br>
        * missing weekly_price values should be set toto 0<br>
        * missing security_deposit values should be set to 0<br>
        * missing jurisdiction_names values should be set to "unknown"<br>
        * neighbourhood_group_cleansed values should be set to "unknown"<br>
        * missing has_availability values should be set to False<br>
        * misssing notes values should be set to "No notes"<br>
        * missing neighborhood_overview values should be set to "not entered"<br>
        * missing transit values should be set to "not provided"<br>
        * missing host_about values should be set to "not provided"<br>
        * missing cleaning_fee values should be set to 0<br>
        * missing space values should be set to "not provided"<br>
        * missing review scores (review_scores_accuracy, review_scores_checkin, review_scores_location, review_scores_value, review_scores_cleanliness, review_scores_communication, review_scores_rating) values should be set to their calculated mean<br>
        * missing reviews_per_month values should be set to 0 <br>
        * missing host_acceptance_rate values should be set to the calculated mean<br>
        * host_response_time missing values should be set to 999<br>
        * missing host response rate values should be set to 0<br>
        * missing neighbourhood values should be set to "not identified"<br>
        * missing medium_url values should be set to "missing url"<br>
        * missing xl_picture_url values should be set to "missing url"<br>
        * missing thumbnail_url values should be set to "missing url"<br>
        * missing host_neighbourhood values should be set to "not identified"<br>
        * missing bathrooms values should be set to 1<br>
        * missing host_location values should be set to "not identified"<br>
        * missing bedrooms values should be set to 1<br>
        * missing beds values should be set to 1<br>
        * missing property_type values should be set to "not identified"<br>
        * missing host_has_profile_pic values should be set to False<br>
        * missing host_name values should be set to "unknown"<br>
        * missing host_identity_verified values should be set to False<br>
        * missing host_is_superhost values should be set to False<br>
        * missing host_total_listings_count values should be set to 0<br>
        * missing host_thumbnail_url values should be set to "no url"<br>
        * missing host_picture_url values should be set to "no url"<br>

Issues with the reviews dataframe:
        1) The date field should be changed from object to datetime<br>
        2) Comments are missing some values that should be replaced with "No comment"<br>


Tidyness Issues:
    1) drop license in the listings dataframe as it has 100% missing values<br>
    2) drop square feet in the listings dataframe as it has 97% missing values<br>
    3) drop interaction in the listings dataframe as it is missing 72% of values and is not needed for this <br>
    4) drop market in the listings dataframe as it is not needed for this <br>


Using a combination of programmatic techniques, the data was cleaned and saved as:
    air_listings.csv<br>
    air_calendar.csv<br>
    air_reviews.csv<br>
All three files were saved as zipped files due to their size.


Main features of the dataset, driving my interest:

I am particularly interested in the difference between Hosts and Superhosts.  I became interested when reviewing the definitions of both on AirBnB's website.  I thought it would be interesting to evaluate the data based on the criterion identified on AirBnB's site.

There are a number of variables in the dataset that can support the investigation. We can start by looking at the proportion of hosts and superhosts for each city.  We can evaluate if there are differences in ratings, scores, and price per night.  We can also attempt to see if machine learning can identify if a host is is a Superhost.

Summary of Findings

The following findings were identified in the dataset.

The following items were of interest:

    * At 20%, there are approximately twice as many superhosts in Seattle than there are in Boston.<br>
    * Superhosts tend to be more highly rated as a group by guests <br>
    * Hosts appear to have a similar number of listings compared to superhosts <br>
    * While Superhosts tend to be clustered in similary neighborhoods in Seattle, they are not clustered in similar neighborhoods in Boston.<br>
    * Median prices per night for Hosts and Superhosts are quite similar.<br>


I decided to model the data using both DecisionTree and RainForest Classifiers.  I did this because I was attempting to assign a label to a host as "Host" or "Superhost." The model was first evaluated to determine which classifere would provide better predictability.  When it was determined that RainForest would work better, I optimized it.

The optimized model resulted in a predictive accuracy of 90.4%, correctly identifying a host as a Superhost.


Looking at the factors evaluated in the model, it was interesting to see:

Of top items affecting the prediction of a Host as Superhost, 4 items were associtated with reviews (number of reviews, review scores rating, reviews per month, review scores communication) and 2 with ratings(overall host rating, review.  Price, reviews per month, availability, and the number of listings also were present in the top 10.


Key Insights for Blog Post

This analysis focused on differences between Hosts and Superhosts using AirBnB data.

We hope to answer the blog post:

What are the proportions of Hosts and Superhosts for Seattle and Boston?<br>
How does the Superhost compare to the Host for overall rating<br>
How do average overall stays compare for Superhosts and Hosts?<br>
Are Superhosts clustered into different neighbourhoods than hosts<br>How do the daily rates compare for hosts and superhosts?
Can we use machine learning to predict which Hosts will are Superhosts?<br>
Using the data provided, what appears to be the most significant in predicting if a Host is a Superhost?  

Key Findings:

* At 20%, there are approximately twice as many superhosts in Seattle than there are in Boston.<br>
* Superhosts tend to be more highly rated as a group by guests<br>
* Hosts appear to have a similar number of listings compared to superhosts<br>
* While Superhosts tend to be clustered in similary neighborhoods in Seattle, they are not clustered in similar neighborhoods in Boston.<br>
* Median prices per night for Hosts and Superhosts are quite similar.

Using RandomForest and DecisionTree classifiers, we were able to determine that RandomForest would provide the greatest accuracy.  After tuning the RandomForest Classifer we were able to predict which hosts are superhosts using the available data with an accuracy of 90.4%
