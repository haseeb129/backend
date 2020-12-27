# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from djongo import models


class User(models.Model):
    _id = models.ObjectIdField()
    # id = models.IntegerField(max_length=200)
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    objects = models.DjongoManager()

# from django.db import models

# Create your models here.
