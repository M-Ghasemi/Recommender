# -*- coding: utf-8 -*-
"""
'User Recommender System' based on the Collaborative Filtering Technique.
.. _The data is from guidetodatamining book:
    http://guidetodatamining.com.

supported methods:
    * cosine
    * pearson
    * manhattan
    * euclidean

Todo:
    * Adding/Refactoring docstrings
"""
import copy
import random

from math import sqrt
from collections import defaultdict

import settings
from db import mongodb as db


def check_db(db):
    collections = db.list_collection_names()
    collections_dict = {}
    message = None

    for col in collections:
        collections_dict[col] = str(db[col].estimated_document_count()) + ' documents'
    if collections:
        message = f'''The database is not empty. The following collections exist.
            {collections_dict}'''
    else:
        message = 'The database is empty.'
    print(message)

    insert_db = input('Do you want to insert the data from '
                      'http://guidetodatamining.com/chapter2/ into database?'
                      '(y/n)\nExisting data will be deleted!\n')

    if not insert_db == 'y':
        return

    import os
    import shutil

    shutil.unpack_archive(
        os.path.join(settings.BASE_DIR, 'data/dump.zip'),
        os.path.join(settings.BASE_DIR, 'data/dump'),
    )

    for path, dirs, files in os.walk(os.path.join(settings.BASE_DIR, 'data/dump')):
        db_name = path.split('/')[-1]
        for file in files:
            if file.split('.')[-1] != 'bson':
                continue

            collection_name = file.split('.')[0]

            db[collection_name].drop()

            restore_cmd = "mongorestore --collection {} --db {} {}".format(
                collection_name, db_name, os.path.join(path, file)
            )
            if settings.MONGO.get('USER'):
                restore_cmd += ' --user {}'.format(settings.MONGO['USER'])
            if settings.MONGO.get('PASSWORD'):
                restore_cmd += ' --password {}'.format(
                    settings.MONGO['PASSWORD'])

            print(os.system(restore_cmd))


def generate_2_related_users(common_ratings, user1_ratings, user2_ratings):
    def anonymize_user(user):
        user['ratings'] = {}
        user['id'] = random.randint(1111, 9999)
        user['location'] = 'zip code' + str(random.randint(1111, 9999))
        user['age'] = random.randint(10, 100)

    user1 = db.users.find_one({}, {'_id': 0})
    user2 = copy.deepcopy(user1)
    anonymize_user(user1)
    anonymize_user(user2)

    books = db.books.find({}, {'_id': 0, 'isbn': 1})
    books.limit(common_ratings + user1_ratings + user2_ratings)
    books = list(books)

    for i in range(common_ratings):
        user1['ratings'][books[i]['isbn']] = random.randint(1, 10)
        user2['ratings'][books[i]['isbn']] = random.randint(1, 10)

    if common_ratings:
        j_range = range(i + 1, i + 1 + user1_ratings)
    else:
        j_range = range(0, user1_ratings)

    for j in j_range:
        user1['ratings'][books[j]['isbn']] = random.randint(1, 10)

    if user1_ratings:
        k_range = range(j + 1, j + 1 + user2_ratings)
    else:
        if common_ratings:
            k_range = range(i + 1, i + 1 + user2_ratings)
        else:
            k_range = range(0, user2_ratings)

    for k in k_range:
        user2['ratings'][books[k]['isbn']] = random.randint(1, 10)

    return (user1, user2)


class UserRecommender:
    """
    This is a UserRecommender class based on Collaborative filtering Technique.
    It implemented for a book rating system.
    """

    def __init__(self, k=1, metric='pearson', n=5):
        """ initialize UserRecommender"""

        self.k = k  # Number of neighbors
        self.n = n  # Number of Items to recommend
        self.metric = metric  # The metric used for distance computation
        self.user = {}  # The user who we want to recommend - (active user)

        # The method that calculates the distance
        # It depends on the metric and it changes with metric changes
        self.compute_distance = getattr(self, metric)

        self.books = self.get_books_dict()  # A dictionary of all data base books

        # A list of users who have at least one common rating with the active user
        self.related_users = []

        # Distances computed for each metric, will be stored in a properly named variable
        self.cosine_distances = []
        self.pearson_distances = []
        self.manhattan_distances = []
        self.euclidean_distances = []

        # The name of various metrics
        self.euclidean_name = 'euclidean'
        self.pearson_name = 'pearson'
        self.manhattan_name = 'manhattan'
        self.cosine_name = 'cosine'

    def set_user(self, user_id):
        self.clean()
        self.user = db.users.find_one({'id': user_id})

    def get_dissimilarity_distance_metrics(self):
        """This method returns names of methods that calculate distance (not similarity)
        of two users."""
        return [self.manhattan_name, self.euclidean_name]

    def get_similarity_distance_metrics(self):
        """This method returns names of methods that calculate distance (not similarity)
        of two users."""
        return [self.cosine_name, self.pearson_name]

    def get_books_dict(self):
        """
        This method returns a dictionary of all data base books in bellow format:
        {
            'isb': '[title] by [author]'
        }

        """
        book_dict = {}
        for book in self.find_books():
            book_dict[book['isbn']] = book['title'] + ' by ' + book['author']

        return book_dict

    def find_books(self, isbn_list=None):
        """This method returns a list of data base books."""
        if isbn_list:
            books = list(db.books.find(
                {
                    'isbn': {
                        '$in': isbn_list
                    }
                }
            ))
        else:
            books = list(db.books.find())

        return books

    def find_related_users(self, user):
        """
        This method returns a list of users who have at least one common rating with the active user.
        """
        if self.related_users:
            return self.related_users

        criteria = {
            'id': {'$ne': user['id']},
            '$or': [
                {'ratings.' + key: {'$exists': True}} for key in user['ratings'].keys()
            ]
        }
        self.related_users = list(db.users.find(criteria))

        return self.related_users

    def get_user_description(self):
        """This method Returns a representation of active user"""
        user_description = self.user['location']
        if self.user.get('age'):
            user_description += ' (age: ' + str(self.user['age']) + ')'

        return user_description

    def user_ratings(self, n=5):
        """This method returns n top ratings of active user"""

        if self.user is None:
            print('Please first set "active user" by calling set_user method')
            return

        print("Ratings for " + self.get_user_description())
        print("Ratings_count: ", len(self.user['ratings']))

        ratings = []

        # user documents is as follow
        # user_sampel = {
        #     "ratings": {
        #         "0440234743": 4,
        #         "0452264464": 6,
        #         "0609804618": 0
        #     },
        #     "id": 9,
        #     "location": "germantown, tennessee, usa",
        #     "age": 23
        # }

        for isbn in self.user['ratings'].keys():
            ratings.append(
                (self.books[isbn], self.user['ratings'][isbn])
            )

        # Sort ratings in descending order
        ratings.sort(key=lambda rating: rating[1], reverse=True)

        # Print n top ratings in this format: [title] by [author]   rate
        for rating in ratings[:n]:
            print("{}\t{}".format(*rating))

    def pearson(self, user1, user2):
        """This method compute Pearson-Correlation-Coefficient for two users ratings"""

        rating1 = user1['ratings']
        rating2 = user2['ratings']

        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for key in set(rating1.keys()).intersection(set(rating2.keys())):
            n += 1
            x = rating1[key]
            y = rating2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += pow(x, 2)
            sum_y2 += pow(y, 2)
        if n == 0:
            return 0
        # now compute denominator
        denominator = (
            sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n)
        )

        try:
            pcc = (sum_xy - (sum_x * sum_y) / n) / denominator
        except ZeroDivisionError:
            # denominator = 0
            pcc = 0

        return pcc

    def minkowski(self, user1, user2, r):
        """This method compute Minkowski distance between two users"""

        rating1 = user1['ratings']
        rating2 = user2['ratings']

        distance = 0

        common_ratings = False
        for key in set(rating1.keys()).intersection(set(rating2.keys())):
            if rating1[key] and rating2[key]:
                distance += pow(abs(rating1[key] - rating2[key]), r)
                common_ratings = True

        minkowski_distance = pow(distance, 1 / r) if common_ratings else 0

        return minkowski_distance

    def manhattan(self, user1, user2):
        """This method compute Manhattan distance between two users"""
        return self.minkowski(user1, user2, 1)

    def euclidean(self, user1, user2):
        """This method compute Euclidean distance between two users"""
        return self.minkowski(user1, user2, 2)

    def cosine(self, user1, user2):
        """This method compute Cosine similarity between two users"""

        rating1 = user1['ratings']
        rating2 = user2['ratings']

        keys = set(rating1.keys())
        keys.update(set(rating2.keys()))

        x = sqrt(sum([rate ** 2 for rate in rating1.values()]))
        y = sqrt(sum([rate ** 2 for rate in rating2.values()]))
        x_dot_y = sum([rating1.get(key, 0) * rating2.get(key, 0)
                       for key in keys])

        try:
            cosine_similarity = x_dot_y / (x * y)
        except ZeroDivisionError:
            cosine_similarity = 0

        return cosine_similarity

    def compute_nearest_neighbor(self, user):
        """This method a sorted list of users based on their distance to active user"""

        # Distances computed for each metric, will be stored in a properly named variable
        distances_attribute_name = '{}_distances'.format(self.metric)
        # If the distances of current metric are computed before,
        # we will not compute them again
        distances = getattr(self, distances_attribute_name, [])

        # Return the distances if already already exist
        # or return an empty list if user has not rating to compare with others
        if distances or not user['ratings']:
            return distances

        related_users = self.find_related_users(user)

        for user2 in related_users:
            # compute_distance method is set according to the metric
            # (pearson, cosine, manhattan, etc.)
            d = self.compute_distance(user, user2)
            distances.append((user2['id'], d))

        # sort based on distance -- closest first
        # Some methods like euclidean and manhattan compute distance of two users,
        # So bigger numbers show more dissimilarity between two users.
        # But some other methods like cosine and pearson compute similarity of two users,
        # So bigger numbers show more simmilarity between two users.
        if self.metric in self.get_dissimilarity_distance_metrics():
            # sort in ascending order because bigger numbers show weaker similarity
            reverse_sort = False
        else:
            # sort in descending order because bigger numbers show stronger similarity
            reverse_sort = True
        distances.sort(
            key=lambda artistTuple: artistTuple[1], reverse=reverse_sort)

        # Distances computed for each metric, will be stored in a properly named variable
        # for further use.
        setattr(self, distances_attribute_name, distances)

        return getattr(self, distances_attribute_name)

    def set_metric(self, metric):
        """This method will set metric and so compute_distance method"""

        # set method of distance computation according to new metric
        self.compute_distance = getattr(self, metric)
        # set metric
        self.metric = metric

    def clean(self):
        """This method cleans all cached data"""
        self.related_users = []
        self.cosine_distances = []
        self.pearson_distances = []
        self.manhattan_distances = []
        self.euclidean_distances = []
        self.cosine_recommended_list = []
        self.pearson_recommended_list = []
        self.manhattan_recommended_list = []
        self.euclidean_recommended_list = []
        self.user = {}

    def get_total_distance(self, nearest, number_of_neibors):
        # This method returns sum of (similarity/dissimilarity) distances
        return float(sum((nearest[i][1] for i in range(number_of_neibors))))

    def recommend_according_to_similarity_metrics(self):
        """
        This method makes a recommendation list according to
        similarity metrics like Cosine and Pearson.
        ''''Note that Bigger number shows stronger similarity''''
        """

        recommendations = defaultdict(int)
        # first get list of users  ordered by nearness
        nearest = self.compute_nearest_neighbor(self.user)
        if not nearest:
            # No similar user found, so no suggestion can be given
            return []

        number_of_neibors = min(self.k, len(nearest))
        sum_of_similarities = self.get_total_distance(
            nearest, number_of_neibors)

        if sum_of_similarities == 0:
            # Actually there is no similar user, so no suggestion can be given
            return []

        # now iterate through the nearest neighbors for making recommendations
        for i in range(number_of_neibors):
            # Each neighbor affects recommendations as much as his similarity to active user
            weight = nearest[i][1] / sum_of_similarities
            neighbor = db.users.find_one({'id': nearest[i][0]})
            # Books that the neighbor likes, will be offered to the active user
            # according to their similarity
            for key in set(neighbor['ratings'].keys()).difference(
                set(self.user['ratings'].keys())
            ):
                recommendations[key] += neighbor['ratings'][key] * weight

        recommended_list = []
        for key, value in recommendations.items():
            # replacing book isbn with "[title] by [author]" and adding it to recommended_list
            recommended_list.append(
                (self.books[key], value)
            )

        recommended_list.sort(
            key=lambda artistTuple: artistTuple[1], reverse=True)

        return recommended_list

    def recommend_according_to_dissimilarity_metrics(self):
        """
        This method makes a recommendation list according to
        dissimilarity metrics like Manhattan and Euclidean
        ''''Note that Bigger number shows weaker similarity''''
        """

        recommendations = defaultdict(int)
        # first get list of users  ordered by nearness
        nearest = self.compute_nearest_neighbor(self.user)
        if not nearest:
            # No similar user found, so no suggestion can be given
            return []

        number_of_neibors = min(self.k, len(nearest))
        sum_of_dissimilarities = self.get_total_distance(
            nearest, number_of_neibors)

        if sum_of_dissimilarities == 0:
            sum_of_similarities = 0
        else:
            sum_of_similarities = sum(
                (
                    1 - nearest[i][1] / sum_of_dissimilarities
                    for i in range(number_of_neibors)
                )
            )
        # now iterate through the nearest neighbors
        for i in range(number_of_neibors):
            if sum_of_similarities:
                weight = (
                    1 - nearest[i][1] / sum_of_dissimilarities) / sum_of_similarities
            else:
                weight = 1 / number_of_neibors
            neighbor = db.users.find_one({'id': nearest[i][0]})
            for key in set(neighbor['ratings'].keys()).difference(
                set(self.user['ratings'].keys())
            ):
                recommendations[key] += neighbor['ratings'][key] * weight

        recommended_list = []
        for key, value in recommendations.items():
            # replacing book isbn with "[title] by [author]" and adding it to recommended_list
            recommended_list.append(
                (self.books[key], value)
            )

        recommended_list.sort(
            key=lambda artistTuple: artistTuple[1], reverse=True)

        return recommended_list

    def recommend_by_metric(self, metric):
        """This method makes a recommendation list according to metric"""

        self.set_metric(metric)

        # Some methods like euclidean and manhattan compute distance of two users,
        # So bigger numbers show more dissimilarity between two users.
        # But some other methods like cosine and pearson compute similarity of two users,
        # So bigger numbers show more simmilarity between two users.
        if self.metric in self.get_similarity_distance_metrics():
            recommended_list = self.recommend_according_to_similarity_metrics()
        else:  # ('manhattan', 'euclidean')
            recommended_list = self.recommend_according_to_dissimilarity_metrics()

        setattr(self, self.metric + '_recommended_list', recommended_list)

        return recommended_list

    def get_all_supported_metrics(self):
        """This method returns all supported similarity Metrics by UserRecommender class"""
        return [
            self.manhattan_name,
            self.euclidean_name,
            self.cosine_name,
            self.pearson_name
        ]

    def print_recommendations(self, metric, recommended_list):
        print("****", metric, "****")
        for item in recommended_list:
            print('--Book: {}, --Rank: {}'.format(item[0], round(item[1])))
        print()

    def recommend(self, metric, user_id=None, k=1, n=5, r=2):
        """
        This method prints a list of recommendations and stores all related informations
        for further use.
        """
        if user_id:
            if user_id != self.user.get('id'):
                self.set_user(user_id)
        elif not self.user:
            print(
                'User ID not provided\n' +
                'You should provide a user_id or set a user by calling set_user method'
            )
            return
        if metric == 'all':
            for metric in self.get_all_supported_metrics():
                recommended_list = self.recommend_by_metric(metric)
                self.print_recommendations(metric, recommended_list[:self.n])
        else:
            recommended_list = self.recommend_by_metric(metric)
            self.print_recommendations(metric, recommended_list[:self.n])
