# Generated by Django 2.2.4 on 2020-01-09 07:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bot', '0005_album_result_path'),
    ]

    operations = [
        migrations.CreateModel(
            name='Imageupload',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image_file', models.ImageField(upload_to='images/')),
                ('date_of_upload', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
