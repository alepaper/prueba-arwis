# Generated by Django 3.1.4 on 2021-03-15 13:54

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='FreeUser',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(max_length=50, unique=True)),
                ('sex', models.CharField(max_length=5)),
                ('main_tessitura', models.CharField(max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='UserAudios',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text_file', models.FileField(upload_to='audio-files/free-users/text-files')),
                ('results_text_file', models.TextField(blank=True, null=True)),
                ('glissando_file', models.FileField(upload_to='audio-files/free-users/glissando-files')),
                ('results_glissando_file', models.TextField(blank=True, null=True)),
                ('results', models.TextField(blank=True, null=True)),
                ('free_user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='base.freeuser')),
            ],
        ),
    ]