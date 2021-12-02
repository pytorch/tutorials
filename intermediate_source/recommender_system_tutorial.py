"""
Recommender system in PyTorch
=============================

"""


######################################################################
# PyTorch is basically a deep learning framework. This means that we can
# solve a huge amount of real-world problems using pyTorch.
# 


######################################################################
# Even before the re-rise of deep learning approaches, the field of
# machine learning and data analysis were beloved research topics of this
# field. These approaches were not just studied in an academy. There were
# a lot of applications, and among them, recommender system is a
# representative one that made a great success in real-world.
# 


######################################################################
# Recommender system is a model that predicts which item a user will like.
# As a prior knowledge, we will be given a data on user’s preference on
# some of the items that the user has already rated. Additional
# information on users or items could be given as an extra data. There are
# some different types of models, but the most popular form of the model
# is the one that predicts user’s preferences on items that the user has
# not rated yet. In other words, it is to predict the missing data from
# the known data. After then, for a specific user, we can sort the items
# that the user has not rated by their predicted preference in decreasing
# order, and suggest some of the first few items to the user. There has
# been a lot of methods to make such predictions, including a lot of ones
# that do not rely on deep learning.
# 


######################################################################
# In this tutorial, we’ll take a look at the classical recommender system.
# There once was a competition called *Netflix Prize* that drew
# researchers’ attraction. It was a competition to predict users’
# star-rating on the movies with some of the given rating data. We’ll
# follow the similar way with the similar data. Although it was not the
# final winner of the competition, we’ll take a deeper look on a model
# that made a historical, and the most important breakthrough in this
# field. It is a collaborative filtering method which uses a singular
# value decomposition.
# 


######################################################################
# Please keep in mind that neural networks will not appear in this
# tutorial. Instead, please enjoy the way how a deep learning framework is
# used on a work without deep learning!
# 


######################################################################
# The Problem
# ~~~~~~~~~~~
# 


######################################################################
# First of all, let’s define the problem. Suppose that you’re maintaing a
# website where users rate the movies. There are :math:`N` users and
# :math:`M` movies. Some of the users have already rated some movies.
# You’ll be given the rating data as a input of the problem. For the
# output, you have to predict the rest of the rating data; How the users
# will rate the movies which they haven’t rated yet.
# 


######################################################################
# How would you solve this problem? If you have no idea, then just think
# about the movies you like. Suppose that you like a movie with particular
# acting person, or a movie directed by a particular director, or a movie
# of a specific genre. Then you’ll probably like the ones with the same
# acting person, the same director, or the same genre. Or suppose that you
# have a friend who has the same list of favorite movies. If there’s a new
# movie that your friend fall in love with, then you’ll probably follow
# the same way. These are well known approach to this problem, and of
# course, there are tons of other ideas.
# 


######################################################################
# But how it is related with a recommendation? If you can predict a user’s
# rating on movies, then you can also predict the list of the movies the
# user will like. Predict the ratings, sort them, and just print the first
# few movies in the list which have ratings over a pre-determined
# threshold! So predicting the rating is simply everything of a
# recommender system.
# 


######################################################################
# MovieLens Dataset
# ~~~~~~~~~~~~~~~~~
# 


######################################################################
# In this tutorial, we will use MovieLens dataset. It contains movie
# ratings by users of a website called MovieLens. There are some additonal
# data as well but we will only use the ratings data here. Please refer to
# ``https://grouplens.org/datasets/movielens/`` for detailed information.
# 


######################################################################
# There are three bunches of data; A small one, medium-size one, and a
# large one. As you can see in the code below, each dataset has different
# names. Let’s use the smallest one for convenience.
# 

DATASET_NAME = 'ml-latest-small' # MovieLens Latest Datasets (Small) recommended for education and development
# DATASET_NAME = 'ml-25m' # MovieLens 25M Dataset recommended for new research
# DATASET_NAME = ''ml-latest' # MovieLens Latest Datasets (Full) recommended for education and development


######################################################################
# Now we will download the dataset file from the website. While unzipping
# the downloaded file, you can see the names of the files included. As
# mentioned above, we will just use ``ratings.csv`` data here.
# 

import os
os.system(f"wget -nv 'https://files.grouplens.org/datasets/movielens/{DATASET_NAME}.zip'")
os.system(f"unzip {DATASET_NAME}.zip")


######################################################################
# Let’s load the data. To make it easy to work on the data, let’s use
# ``pandas``.
# 

import pandas as pd
data = pd.read_csv(f"{DATASET_NAME}/ratings.csv")


######################################################################
# And let’s take a bit of look on the data.
# 

print(data.sample(10))
print('min-----')
print(data.min())
print('max-----')
print(data.max())
print(f'Number of ratings: {len(data)}')


######################################################################
# As you can see, each row of the data contains user information, movie
# information, rating information, and time information. The IDs are
# integer values and ratings are floating numbers. Though it might not be
# an exact number, it seems that the numbers of users and movies are
# aroung 610 and 193609, respectively. The dataset covers a large number
# of movies, but not so many of users; It surely is just some portion of
# the original data, suitable for introductory purpose. Ratings starts
# from 0.5 and does not exceed 5.0. We can easily guess that the unit is
# 0.5, and that there are 10 unique rating values. The number of entries
# is far from the number of combinations of users and movies (just 0.08%
# only!). We have to guess the rest from this really, really small portion
# of data.
# 


######################################################################
# Okay, let’s go further.
# 


######################################################################
# Ratings as a Matrix
# ~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# We can regard ratings as a matrix, with possibly some missing entries.
# The rows of the matrix correspond to the users, and columns correspond
# to the movies. If :math:`i`-th user rated :math:`j`-th movie with rating
# value of :math:`r_{ij}`, then :math:`(i, j)` element of the matrix has a
# value of :math:`r_{ij}`. If there is no rating information in the data
# for the same pair of user and movie, :math:`(i, j)` element is missing
# and we have to predict the value. Then we can think of a problem of
# filling some missing entries of a matrix.
# 


######################################################################
# As we saw in the sample data as above, there could be a large number of
# users and/or movies. The number of combination of them is even larger.
# This really is a huge problem space and we need an efficient algorithm.
# 


######################################################################
# One thing to mention is that, we will allow arbitrary values for
# predicted ratings. The value might not end with ‘0.0’ and ‘0.5’. It
# could be even smaller than 0.5, or bigger than 5.0. As we will see soon,
# it does not bother evaluating the result.
# 


######################################################################
# Evaluation
# ~~~~~~~~~~
# 


######################################################################
# After predicting missing values of the matrix, how can we evaluate the
# result? Or essentially, what does it mean to evaluate a recommendation?
# 


######################################################################
# It might not be the right time to explain this topic, but let’s think
# about it. It would be good if the movies the user actually likes is
# included in the recommendation. We measure it with a *recall* value. But
# what if the system just predicts that the user will like all of the
# movies? All of the actual favorite movies will be included in the
# recommendation, but we know that such recommendation is meaningless. We
# also have to measure the number of recommended movies that the user
# actually likes, by a metric called *precision*. In this case, the system
# might just predict nothing. Then we can say that all of the recommended
# movies were true ones, but it is just another meaningless case.
# 


######################################################################
# To overcome such cases, metrics designed to be reflecting both precison
# and recall are often used. But it also has a drawback in the basis.
# Suppose that the recommender system output a movie that the user will
# like. However, we cannot know the actual response of the user from the
# input data. We just have to actually recommend the movie to the user to
# know the result.
# 


######################################################################
# Then how can we evalute the result just with the given data? A widely
# used metric is an RMSE (Root Mean Square Error) score. We first select
# some of the known ratings. Let’s call them a test dataset. And we run
# the system with the rest of the data, called a train dataset. We’ll get
# a predicted rating for user-movie pairs in the test dataset. And we also
# know the actual rating. So we can measure the error, and can square
# them, get the average (mean value) value of them, and a square root of
# it. That’s the RMSE score.
# 


######################################################################
# For this purpose, let’s split the dataset into train and test dataset.
# For convenience, we will use 10% of the original data as a test dataset,
# as stated in ``test_portion`` in the code below.
# 


######################################################################
# Collaborative Filtering
# ~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# Let’s start with some mathematics, namely Singular Value Decomposition
# (SVD). Given a real matrix :math:`A` of size :math:`n \times m`, we want
# to decompose it into a product of three matrices as :math:`A = U S V^T`.
# Here, :math:`U` is of size :math:`n \times n`, :math:`S` is of size
# :math:`n \times m`, and :math:`V^T` is of size :math:`m \times m`. We
# call it a singular value decomposition, because matrix :math:`S`
# contains singular values. Interestingly, we can use this idea to
# represent an approximation of :math:`A`. If we choose :math:`k` largest
# singular values from :math:`S`, say :math:`\tilde{S}`, then we will get
# a :math:`k` dimensional (rank :math:`k`) approximation
# :math:`\tilde{A} = U \tilde{S} V^T`. It is known that this is an
# approximation that minimizes a certain error term called Frobenius norm.
# 


######################################################################
# Now back to recommender system. We will take a look on a method called
# collaborative filtering. It is a method to give a recommendation
# (filtering) by infering preference from the data of other users/movies
# (collaborate). How can we apply this approach to our case? A natural way
# is to train a model which best fits the given data. We can use SVD for
# this purpose. Say we want to represent preference in :math:`k`
# dimension. But as we have lots of missing entries in the ratings matrix,
# we’d like to minize the error between known ratings and predicted
# ratings only.
# 


######################################################################
# What is the meaning of approximating in :math:`k` dimension? Let’s think
# it this way. We’d like to extract latent features of users and movies.
# There are some characteristics, like genres, acting persons, directors,
# etc., that determines whether a user likes a movie or not. We can choose
# some representative ones of them. The number of such features will be
# way smaller than the numbers of users and movies. We’d like to
# characterize users and movies by these entries only. If we measure
# likeliness of the features on users and movies, then we can make
# prediction on a preference of a new user on a new movie.
# 


######################################################################
# Let’s implement this approach with PyTorch. Given the dimension of
# latent features, we are to find matrices for users and movies. The
# matrices represents the likeliness of the features on users and movies.
# After then, we can approximate the rating matrix by multiplying them.
# Which matrices we have to use? The one that minimizes the RMSE score
# mentioned above. We can define the model like the code below.
# 


######################################################################
# Though we can compute the result mathematically, it might not be
# feasible to compute them. Thus we will try a computational approach. We
# start from a randomly initialized matrices :math:`U` and :math:`V`, and
# improve it step-by-step. This is exactly the same as how we train a
# neural network based model. And this is the reason why we are using a
# deep learning framework to this problem.
# 


######################################################################
# Now train is over, and let’s see the result on test dataset.
# 


######################################################################
# The result is not really small, but it suffices to know that we’ve found
# an appropriate approximation. For your information, Netfilx Prize
# started with an RMSE score of around 1.00, and finally arrived at a
# score between 0.85 and 0.90. The datasets are different but you can
# understand where we’ve arrived in tutorial
# 


######################################################################
# Concluding Remarks
# ~~~~~~~~~~~~~~~~~~
# 


######################################################################
# In this tutorial, we’ve learned how to implement a classical recommender
# system with PyTorch. It is not a neural network based model, but we used
# a training based approach with the framework.
# 


######################################################################
# How can we make the result better? Note that there was a timestamp field
# in the input data, but we never used it. There also were some additional
# files, and there are information beyond the given dataset, like metadata
# on the movies. We’ll see how we can in following tutorials.
# 