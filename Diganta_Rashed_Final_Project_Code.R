library(tidyverse)
library(dplyr)
library(e1071)
library(data.table)
library(ggplot2)
library(caret)

# read in dataset
df = fread("players.csv")

# Observe dataframes and structure of dataset
colnames(df)
head(df)

df1 = df %>% filter(minutes > 0)
#Create vectors based on columns relevant to each player position in dataset

gk_stats <- c('name', 'minutes', 'team', 'now_cost', 'form', 'status', 'bonus', 'position', 'selected_by_percent', 'total_points', 'goals_conceded', 'expected_goals_conceded', 'clean_sheets', 'saves')
def_stats <- c('name', 'minutes', 'team', 'clean_sheets', 'now_cost', 'form', 'status', 'bonus', 'position', 'selected_by_percent', 'total_points', 'penalties_order', 'expected_assists', 'direct_freekicks_order', 'assists', 'expected_goal_involvements', 'corners_and_indirect_freekicks_order', 'expected_assists_per_90', 'expected_goals_per_90', 'ict_index', 'points_per_game', 'expected_goals', 'expected_goal_involvements_per_90', 'threat', 'creativity', 'goals_scored')
mid_stats <- c('name', 'minutes', 'team', 'now_cost', 'form', 'status', 'bonus', 'position', 'selected_by_percent', 'total_points', 'clean_sheets', 'goals_conceded', 'expected_assists', 'clean_sheets_per_90', 'penalties_order', 'direct_freekicks_order', 'starts_per_90', 'assists', 'expected_goal_involvements', 'corners_and_indirect_freekicks_order', 'expected_assists_per_90', 'expected_goals_per_90', 'ict_index', 'points_per_game', 'expected_goals', 'expected_goal_involvements_per_90', 'threat', 'creativity', 'goals_scored')
fwd_stats <- c('name', 'minutes', 'team', 'now_cost', 'form', 'status', 'bonus', 'position', 'selected_by_percent', 'total_points', 'direct_freekicks_order', 'expected_assists', 'starts_per_90', 'assists', 'expected_goal_involvements', 'corners_and_indirect_freekicks_order', 'expected_assists_per_90', 'expected_goals_per_90', 'ict_index', 'points_per_game', 'penalties_order', 'expected_goals', 'expected_goal_involvements_per_90', 'threat', 'creativity', 'goals_scored')

#Subset dataframes by converting frame into data table 

dt = setDT(df1)

gk_df <- subset(df1, position == "GKP")
def_df <- subset(df1, position == "DEF")
mid_df <- subset(df1, position == "MID")
fwd_df <- subset(df1, position == "FWD")


# Here's the idea so far: write an SVM using total_points as target variable, and then create a naive bayes analysis as well; use the two to compare
# after that interpret the output, pick 3 stats 

# First, we create a SVM, but prior to that we convert "total_points" to a factor so it can be treated for classification
df1$total_points = as.factor(df1$total_points)

svm_model = svm(total_points ~ name, data = df1, kernel = "radial")

#Naive bayes model - No longer used**

nb_model = naiveBayes(total_points ~ ., data = df1)

# Prediction model 

svm_pred = predict(svm_model)
nb_pred = predict(nb_model, newdata = df1)

# plotting the SVM and Naive Bayes model

svm_pred <- factor(svm_pred, levels = levels(df1$total_points))


svm_matrix = confusionMatrix(svm_pred, df1$total_points)
print(svm_matrix)

nb_matrix = table(nb_pred, df1$total_points)
print(nb_matrix)
summary(nb_matrix)

# Regression SVM based on Two valuable stats for each position
gk_svm = svm(total_points ~ clean_sheets + now_cost, data = gk_df, kernel = "radial")
gk_pred = predict(gk_svm)
levels(gk_pred) = levels(gk_df$total_points)
gk_df$total_points = factor(gk_df$total_points)
gk_pred = factor(gk_pred, levels = levels(gk_df$total_points))
gk_matrix = confusionMatrix(gk_pred, gk_df$total_points)
print(gk_matrix)

def_svm = svm(total_points ~ clean_sheets + now_cost, data = def_df, kernel = "radial")
def_pred = predict(def_svm)
levels(def_pred) = levels(def_df$total_points)
def_df$total_points = factor(def_df$total_points)
def_pred = factor(def_pred, levels = levels(def_df$total_points))
def_matrix = confusionMatrix(def_pred, def_df$total_points)
print(def_matrix)
 
mid_svm = svm(total_points ~ assists + now_cost, data = mid_df, kernel = "radial")
mid_pred = predict(mid_svm)
levels(mid_pred) = levels(mid_df$total_points)
mid_df$total_points = factor(mid_df$total_points)
mid_pred = factor(mid_pred, levels = levels(mid_df$total_points))
mid_matrix = confusionMatrix(mid_pred, mid_df$total_points)
print(mid_matrix)

fwd_svm = svm(total_points ~ goals_scored + now_cost, data = fwd_df, kernel = "radial")
fwd_pred = predict(fwd_svm)
levels(fwd_pred) = levels(fwd_df$total_points)
fwd_df$total_points = factor(fwd_df$total_points)
fwd_pred = factor(fwd_pred, levels = levels(fwd_df$total_points))
fwd_matrix = confusionMatrix(fwd_pred, fwd_df$total_points)
print(fwd_matrix)


# Test levels of initial matrix and target variable 
df1$predicted_points = svm_pred
top_10_players = df1 %>% 
  arrange(desc(predicted_points)) %>% 
  head(10)
print(top_10_players)

gk_df$best_gks = gk_pred
top_gk = gk_df %>%
  arrange(desc(best_gks)) %>%
  head(10)
print(top_gk)

def_df$best_def = def_pred
top_def = def_df %>%
  arrange(desc(best_def)) %>%
  head(10)
print (top_def)

mid_df$best_mid = mid_pred
top_mid = mid_df %>%
  arrange(desc(best_mid)) %>%
  head(10)
print (top_mid)

fwd_df$best_fwd = fwd_pred
top_fwd = fwd_df %>%
  arrange(desc(best_fwd)) %>%
  head(10)
print (top_fwd)

