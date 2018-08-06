# -*- coding: utf-8 -*-
"""This file loads guidetodatamining data into mongodb.
pleas note that it drops book and users collection.
"""
import os
import settings


def setup_db(db):
    insert_books, insert_users = 'Y', 'Y'
    if db.books.find_one():
        insert_books = input(
            "The books collection already exists and is not empty, "
            "this operation will delete all of it's records."
            "Press 'c' to cancel droping books collection:\n")
    if db.users.find_one():
        insert_users = input(
            "The users collection already exists and is not empty, "
            "this operation will delete all of it's records."
            "Press 'c' to cancel droping users collection:\n")

    isbn_list = []

    if insert_books != 'c':
        db.books.drop()
        with open(os.path.join(settings.BASE_DIR, 'data/BX-Books.csv'), 'r') as books:
            for item in books:
                book_fields = [field.strip('"') for field in item.split(';')]
                book = {
                    'isbn': book_fields[0],
                    'title': book_fields[1],
                    'author': book_fields[2],
                    'publication_date': book_fields[3],
                    'publisher': book_fields[4],
                    'thumbnails': book_fields[5:]
                }
                book['thumbnails'][-1] = book['thumbnails'][-1].rstrip('"\n')
                db.books.insert_one(book)
                isbn_list.append(book_fields[0])

    if insert_users != 'c':
        db.users.drop()
        with open(os.path.join(settings.BASE_DIR, 'data/BX-Users.csv'), 'r') as users:

            for item in users:
                user_fields = [field.strip('"') for field in item.split(';')]
                user = {
                    'id': int(user_fields[0]),
                    'location': user_fields[1],
                    'ratings': {}
                }

                try:
                    user['age'] = int(user_fields[2].strip('"\n'))
                except Exception:
                    user['age'] = None

                db.users.insert_one(user)

        with open(os.path.join(settings.BASE_DIR, 'data/BX-Book-Ratings.csv'), 'r') as ratings:
            for item in ratings:
                rating_fields = [field.strip('"') for field in item.split(';')]
                isbn = rating_fields[1]
                rate = int(rating_fields[2].strip('"\n'))
                if isbn not in isbn_list or rate == 0:
                    continue
                user_id = int(rating_fields[0])
                db.users.update_one(
                    {'id': user_id},
                    {
                        '$set': {
                            'ratings.' + isbn: rate
                        }
                    }
                )
