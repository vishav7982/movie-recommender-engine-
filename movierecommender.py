from numpy import *
num_movies = 10
num_users = 5
#initializing rating matrix
ratings = random.randint(11, size=(num_movies, num_users))
#print (ratings)

#intializing did rating matrix which gives us 1 if a movie is rated BY USER

did_rate = (ratings != 0) * 1
#print(did_rate)

vishav_ratings = zeros((num_movies,1))

vishav_ratings[1] = 9
vishav_ratings[6] = 9
vishav_ratings[9] = 3
vishav_ratings[8] = 2


ratings = append(vishav_ratings,ratings,axis= 1)
did_rate = append(((vishav_ratings != 0)*1),did_rate,axis=1)


#function that normalizes a dataset

def normalise_ratings(ratings,did_rate):
    num_movies =ratings.shape[0]
    ratings_mean = zeros(shape = (num_movies,1))
    ratings_norm = zeros(shape = ratings.shape)

    for i in range(num_movies):
        idx = where(did_rate[i] == 1)[0]
        ratings_mean[i] = mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    return ratings_norm ,ratings_mean

ratings,ratings_mean = normalise_ratings(ratings,did_rate)

num_users = ratings.shape[1]
num_features = 3

#features of movies like comedy , action , romance
movie_features = random.randn(num_movies,num_features)
#user preferences matrix
user_pref = random.randn(num_users,num_features)
#making a column vector to feed our gradient descent function
initial_X_and_theta = r_[movie_features.T.flatten(),user_pref.T.flatten()]

def unroll_params(X_and_theta, num_users, num_movies, num_features):
    # Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
    # --------------------------------------------------------------------------------------------------------------
    # Get the first 30 (10 * 3) rows in the 48 X 1 column vector
    first_30 = X_and_theta[:num_movies * num_features]
    # Reshape this column vector into a 10 X 3 matrix
    X = first_30.reshape((num_features, num_movies)).transpose()
    # Get the rest of the 18 the numbers, after the first 30
    last_18 = X_and_theta[num_movies * num_features:]
    # Reshape this column vector into a 6 X 3 matrix
    theta = last_18.reshape(num_features, num_users).transpose()
    return X, theta

def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
    X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)

    # we multiply by did_rate because we only want to consider observations for which a rating was given
    difference = X.dot(theta.T) * did_rate - ratings
    X_grad = difference.dot(theta) + reg_param * X
    theta_grad = difference.T.dot(X) + reg_param * theta

    # wrap the gradients back into a column vector
    return r_[X_grad.T.flatten(), theta_grad.T.flatten()]

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
    X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)

    # we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
    cost = sum((X.dot(theta.T) * did_rate - ratings) ** 2) / 2
    # '**' means an element-wise power
    regularization = (reg_param / 2) * (sum(theta ** 2) + sum(X ** 2))
    return cost + regularization

from scipy import optimize

# regularization paramater

reg_param = 30

# perform gradient descent, find the minimum cost (sum of squared errors) and optimal values of X (movie_features) and Theta (user_prefs)

minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta, 								args=(ratings, did_rate, num_users, num_movies, num_features, reg_param), 								maxiter=100, disp=True, full_output=True )


cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]


# unroll once again

movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, num_users, num_movies, num_features)

# Make some predictions (movie recommendations). Dot product

all_predictions = movie_features.dot( user_prefs.T )

# add back the ratings_mean column vector to my (our) predictions

predictions_for_vishav = all_predictions[:, 0:1] + ratings_mean

print (predictions_for_vishav)

print (vishav_ratings)



