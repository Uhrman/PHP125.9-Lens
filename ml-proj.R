###########################################################
#
###########################################################
#
#
#
###########################################################
# Create and save edx set, validation set
###########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

movieLens_10M_dataset_path <- "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
dl <- tempfile()
download.file(movieLens_10M_dataset_path, dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Delete temporary objects

unlink(dl)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Check dataset variables

edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

###########################################################
# RMSE function
###########################################################

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

###########################################################
# RSME: model 0 - without RSME, just the average
###########################################################

mu_hat <- mean(edx$rating)
naive_rmse <- RMSE(validation$rating, mu_hat)

# Create the results table

rmse_results <- tibble(method = "Just the average", 
                       RMSE = round(naive_rmse, 6)
                       )

# Show intermediate result

rmse_results %>% knitr::kable()

###########################################################
# RSME: model 1 - Movie Effect Model
###########################################################

mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

movie_avgs %>% 
  ggplot(aes(b_i)) + 
  geom_histogram(bins = 30, color = "black")

predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Calculate RSME

model_1_rmse <- RMSE(predicted_ratings, validation$rating)

# Add new result to the results table

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",  
                                 RMSE = round(model_1_rmse, 6)
                                 )
                          )

# Show intermediate results

rmse_results %>% knitr::kable() 

###########################################################
# RSME: model 2 - Movie + User Effects
###########################################################

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RSME

model_2_rmse <- RMSE(predicted_ratings, validation$rating)

# Add new result to the results table

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effects Model",  
                                 RMSE = round(model_2_rmse,6)
                                 )
                          )

# Show intermediate results

rmse_results %>% knitr::kable()

#############################################################
# RSME: model 3 - Regularization Movie Effect 
#############################################################

# ???? Penalized Least Squares

lambdas <- seq(0, 10, 0.25)
mu <- mean(edx$rating)

movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + lambda), n_i = n()) 

# Calculate RSMEs

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- 
    validation %>% 
    left_join(movie_reg_avgs, by = "movieId") %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})

# Calculate RSMEs and show results and best lambda

qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

# Add new result to the results table

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie Effect Model",  
                                 RMSE = round(min(rmses), 6)
                                 )
                          )

# Show intermediate results

rmse_results %>% knitr::kable()

#############################################################
# RSME: model 4 - Regularized Movie + User Effect
#############################################################

# Choosing the penalty terms

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

# Calculate RSMEs and show results and best lambda

qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

# Add new result to the results table

rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Regularized Movie + User Effect Model", 
                                 RMSE = round(min(rmses), 6)
                                 )
                          )

#############################################################
# Final result
#############################################################

rmse_results %>% knitr::kable()
