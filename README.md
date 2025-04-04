# Prem-Cluster-Project

## Introduction
In this project I wanted to work on utilizing KNN (K-nearest-neighbors) classification to practice clustering. As I am have a passion for sports, especially soccer, I decided to look at some Premier League statistics and try to classify players into different groups of play-styles. In case you don't know, the Premier League is the league in England, which is one of the top soccer leagues in the world! My goal in the project is to use only the players' statistics to be able to cluster them into groups based on how they play and how the impact a game. Once I group the players, I will analyze the different groups and award them names as well as highlight the noteable players in each group. I am only going to focus on forwards (attackers) in this project, since they are the position in which I am most interested in analyzing and the most meaningful data available is related to forwards. Let's get started!

## Acquiring Data and Data Cleaning
### Acquiring Data
I retrieved all of my data from the Premier League Stats page: https://www.premierleague.com/stats . However, getting the data was not as straightforward as downloading some csv files. In order to obtain the data I used in the project, I used the Premier League website API, which allowed me access to retrieve the data in the form of json objects. With the use of a few steps of long python commands, I was able to retrieve all of the data for forwards into a pandas dataframe! I titled this as my base_df, since it is before any cleaning occurs.
### Data Cleaning
The majority of the work with setting this dataset up for analysis was just retrieving it from the web, and figuring out how to retrieve the necessary information from a messy json dictionary. Since we did that in the previous step, all that was left to do was to change the index to the playerId, and fill the null values. In order to fill the null values, I first had to understand what caused null values in this case. 
In other cases, I may have needed a complete other step to do a missing values analysis, and determine whether the data are MCAR, MAR, or NMAR. However, I know this simply from the way that I retrieved the data from the web. Since each stat was on a different page and therefore a different table, I had to merge a lot of different tables to get all the statistics that I wanted to use. Therefore, I used a left merge with the first dataframe I created which had all the data for players that scored goals. So, once I added a new statistic (column) to the base dataframe if the player was not included in the table with the new statistic, then that player's value would be null for that statistic. This means that the null values are MAR dependent on the `goals` column. I deduced that a player is only not included in a table if their value for that statistic is 0, so therefore I was able to assume that all null values can just be changed to 0. So I did exactly that, and changed all null values in the dataframe to 0, concluding the data cleaning process. The dataset is now clean and ready to be used for KNN classification!

## Clustering
### Determine Best Number of Components
The first step in the clustering process is to determine the best number of components to use from the features that finds the right balance between an expected variance ratio closer to 1 and not overfitting. To determine this number, we must conduct an analysis using our data.
##### Step One: Standardize Data
Our first step in this process is to use the SKlearn StandardScaler preprocessor to standardize our data within each statistic/column. Here is the code I used to accomplish this:
```
df_features = list(df.columns)
X = df.loc[: , df_features].values
X = StandardScaler().fit_transform(X)
```
Now, X is an array with each element being a list that contains all the standardized values for a specific feature. The reason that this step is necessary instead of just using the original data, is that standardizing the data puts all the features on the same scale so that our analysis and SKlearn machine learning processes can work more accurately.
##### Step 2: Find Variance for Different Components
Now that we have standardized our data, we want to find the variance for different number of components. In this context, a good variance value is one that is somewhat close to 1, because our "explained variance ratio" indicates the proportion of total variance represented by that number of components. Therefore, we want to find a component number that is close to 1, but once the increase in variance between components barely changes, then adding more components doesn't help too much. So we want to find the number of components right in the sweet spot.

To accomplish this task, we are using the SKlearn PCA module which gives us the variance for each number of components when fit to our data.
