# Generated by Django 2.2.4 on 2020-01-09 06:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bot', '0004_auto_20200109_1447'),
    ]

    operations = [
        migrations.AddField(
            model_name='album',
            name='result_path',
            field=models.CharField(default='aa', max_length=64),
            preserve_default=False,
        ),
    ]
