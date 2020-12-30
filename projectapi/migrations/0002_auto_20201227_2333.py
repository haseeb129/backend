# Generated by Django 3.0.5 on 2020-12-27 18:33

from django.db import migrations, models
import djongo.models.fields


class Migration(migrations.Migration):

    dependencies = [
        ('projectapi', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='previousProjects',
            fields=[
                ('_id', djongo.models.fields.ObjectIdField(auto_created=True, primary_key=True, serialize=False)),
                ('user_id', models.CharField(max_length=255)),
                ('state', djongo.models.fields.JSONField()),
            ],
        ),
        migrations.DeleteModel(
            name='projectapi',
        ),
    ]
