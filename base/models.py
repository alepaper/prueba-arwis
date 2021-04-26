"""Base models for the project"""
from django.db import models

# Create your models here.


class FreeUser(models.Model):
    email = models.EmailField(max_length=50, unique=True)
    sex = models.CharField(max_length=5)
    main_tessitura = models.CharField(max_length=20)

    def __str__(self):
        return f'E-mail: {self.email}\nSex: {self.sex}\nTessitura: {self.main_tessitura}'


class UserAudios(models.Model):
    free_user = models.ForeignKey(FreeUser, on_delete=models.CASCADE)
    text_file = models.FileField(upload_to='audio-files/free-users/text-files')
    results_text_file = models.TextField(blank=True, null=True)
    glissando_file = models.FileField(
        upload_to='audio-files/free-users/glissando-files')
    results_glissando_file = models.TextField(blank=True, null=True)
    results = models.TextField(blank=True, null=True)

    def __str__(self):
        return f'-User: {self.free_user}\n-Text file: {self.text_file}\n\t*Results of text file: {self.results_text_file}\n-Glissando file: {self.glissando_file}\n\t*Results of glissando file: {self.results_glissando_file}\nResults: {self.results}\n'
