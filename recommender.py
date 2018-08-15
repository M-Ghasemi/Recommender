"""
BookRecommender based on the Collaborative Filtering Technique.
.. _The data is from guidetodatamining book:
    http://guidetodatamining.com.

supported methods:
    * cosine
    * pearson
    * manhattan
    * euclidean

Todo:
    * Refactoring/Adding docstrings
"""
import copy
import random

from math import sqrt
from collections import defaultdict

import settings
from db import mongodb as db


def check_db(db):
    """
    checks and restores the Database.

    Args:
        db (MongoClient): Database cursor.
    """

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


def generate_2_related_users(common_ratings: int, user1_ratings: int, user2_ratings: int) -> tuple:
    """
    Generates two related users.
    uses a database sample user to create to related users.

    Args:
        common_ratings: the number of books that both users rated to them.
        user1_ratings: the number of books that only user 1 has rated to them.
        user2_ratings: the number of books that only user 2 has rated to them.

    Returns:
        tuple: two users dictionary.

    Example:
        >>> from pprint import pprint

        >>> user1, user2 = generate_2_related_users(4, 3, 2)

        >>> pprint(user1)
        {'age': 10,
         'id': 4879,
         'location': 'zip code9337',
         'ratings': {'0002005018': 3,
                     '0345417623': 3,
                     '0375406328': 5,
                     '0375759778': 10,
                     '0449005615': 3,
                     '0887841740': 7,
                     '1575663937': 1}}

        >>> pprint(user2)
        {'age': 72,
         'id': 5694,
         'location': 'zip code8059',
         'ratings': {'0002005018': 1,
                     '0345417623': 1,
                     '038078243X': 1,
                     '055321215X': 7,
                     '0887841740': 5,
                     '1575663937': 4}}

    """
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


class BookRecommender:
    """
    Book Recommender class based on Collaborative filtering Technique.
    Args:
        k (int): the number of nearest neighbors to compare with.
        metric (str): call BookRecommender.get_all_supported_metrics() to get a
            list of all supported metrics. Any other value will raise a
            ValueError exception.
        n (int): the number of books to be recommended.
    """

    def __init__(self, k: int = 1, metric: str = 'pearson', n: int = 5):

        self.k = k
        self.n = n
        self.metric = metric
        self.user = {}

        # setting distance/similarity method based on the metric
        try:
            self.compute_distance = getattr(self, metric)
        except AttributeError:
            raise ValueError('metric is not supported!')

        self.books = self.get_books_dict()

        self.related_users = []

        # caching  the distances computed for each metric
        self.cosine_distances = []
        self.pearson_distances = []
        self.manhattan_distances = []
        self.euclidean_distances = []

        self.euclidean_name = 'euclidean'
        self.pearson_name = 'pearson'
        self.manhattan_name = 'manhattan'
        self.cosine_name = 'cosine'

    def set_user(self, user_id: int):
        """
        cleans all computed data and sets the new user.
        """
        self.clean()
        self.user = db.users.find_one({'id': user_id})

    def get_dissimilarity_distance_metrics(self) -> list:
        """
        Returns:
            a list of the string names of the distance based methods.
        """
        return [self.manhattan_name, self.euclidean_name]

    def get_similarity_distance_metrics(self) -> list:
        """
        Returns:
            a list of the string names of the similarity based methods.
        """
        return [self.cosine_name, self.pearson_name]

    def get_books_dict(self) -> dict:
        """
        Returns:
            a dictionary of all the books in the database.
        """
        book_dict = {}
        for book in self.find_books():
            book_dict[book['isbn']] = book['title'] + ' by ' + book['author']

        return book_dict

    def find_books(self, isbn_list: list = None) -> list:
        """
        Returns:
            a list of the database books with isbn_list ids. if isbn_list is
            not provided, then returns all the database books.
        """
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

    def find_related_users(self, user: dict) -> list:
        """
        Args:
            user (dictionary): id, and ratings are required fields
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }

        Returns:
            a list of users with at least one common rating with the user
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

    def get_user_description(self) -> str:
        """
        Returns:
            a string representation (location and age) of the self.users
        """
        user_description = self.user['location']
        if self.user.get('age'):
            user_description += ' (age: ' + str(self.user['age']) + ')'

        return user_description

    def user_ratings(self, n: int = 5):
        """
        prints n top ratings of the self.user (descending)
        """

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

    def pearson(self, user1: dict, user2: dict) -> float:
        """
        Args:
            user1 (dict): user document.
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }
            user2 (dict): user document.
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }
        Returns:
            float number pearson correlation coefficient of two users ratings
        """

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

    def minkowski(self, user1: dict, user2: dict, r: int) -> float:
        """
        Args:
            user1 (dict): user document.
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }
            user2 (dict): user document.
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }
            r (int): root number of minkowski formula. (1 for manhattan, 2 for euclidean)
        Returns:
            float number minkowski distance of two users ratings
        """

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

    def manhattan(self, user1: dict, user2: dict) -> float:
        """
        Args:
            user1 (dict): user document.
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }
            user2 (dict): user document.
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }
        Returns:
            float number manhattan distance of two users ratings
        """
        return self.minkowski(user1, user2, 1)

    def euclidean(self, user1: dict, user2: dict) -> float:
        """
        Args:
            user1 (dict): user document.
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }
            user2 (dict): user document.
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }
        Returns:
            float number euclidean distance of two users ratings
        """
        return self.minkowski(user1, user2, 2)

    def cosine(self, user1: dict, user2: dict) -> float:
        """
        Args:
            user1 (dict): user document.
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }
            user2 (dict): user document.
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }
        Returns:
            float number cosine similarity of two users ratings
        """
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

    def compute_nearest_neighbor(self, user: dict):
        """
        computes, caches and returns the neighbors according to self.metric.
        Args:
            user (dict): user document.
                {
                    'id': user_id,
                    'ratings': {
                        'isbn': integer rate from 1 to 10,
                        'another isbn': integer rate from 1 to 10,
                    },
                }
        Returns:
            a sorted list of neighbors of the self.user. each entry is a tuple
            of id and distance/similarity. if self.user has no
            ratings then returns an empty list.
        """

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

    def set_metric(self, metric: str):
        """sets metric and also compute_distance method
        Args:
            metric (str): call BookRecommender.get_all_supported_metrics()
            to get a list of all supported metrics.
        """
        # set method of distance computation according to new metric
        self.compute_distance = getattr(self, metric)
        # set metric
        self.metric = metric

    def clean(self):
        """cleans all non default data"""
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
        """returns sum of (similarity/dissimilarity) distances"""
        return float(sum((nearest[i][1] for i in range(number_of_neibors))))

    def recommend_according_to_similarity_metrics(self) -> list:
        """
        makes a recommendation list according to similarity metrics like
        Cosine and Pearson.
        Note that this function suppose that self.metric is already set to one
        of the supported similarity metrics.
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

    def recommend_according_to_dissimilarity_metrics(self) -> list:
        """
        makes a recommendation list according to similarity metrics like
        euclidean and manhattan.
        Note that this function suppose that self.metric is already set to one
        of the supported similarity metrics.
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

    def recommend_by_metric(self, metric: str) -> list:
        """
        Args:
            metric (str): call BookRecommender.get_all_supported_metrics()
            to get a list of all supported metrics.
        Returns:
            a list of recommended book and their rank.
        """
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

    def get_all_supported_metrics(self) -> list:
        """
        Returns:
            a list all supported metrics
        """
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

    def recommend(self, metric: str, user_id: int = None, k: int = 1, n: int = 5, r: int = 2):
        """
        prints an n item list of recommended books.

        Args:
            metric (str): call BookRecommender.get_all_supported_metrics()
                to get a list of all supported metrics. Any other value will
                raise a ValueError exception.
            user_id (int): int number database user id.
            k (int): the number of nearest neighbors to compare with.
            n (int): the number of books to be recommended.
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
