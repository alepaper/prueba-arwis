import numpy as np
# import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
from scipy import stats
import warnings


class MainClassifier:
    def __init__(self, sex):
        self.sex = sex
        return

    def classify(self, filename, speech=False, optionModeText=False, backtrack=False):
        detectNote = DetectNotesFromCQT(filename, backtrack=backtrack)
        # result,cqt,new_cqt=detectNote.result_Dict(plot=False,speech=speech,optionModeText=optionModeText)
        result = detectNote.result_Dict(
            plot=False, speech=speech, optionModeText=optionModeText)
        classification = ClassifyByNotes(result, self.sex, speech=speech)
        percentTessitura = classification.dic_classify(optionModeText)
        result['PercentTessitura'] = percentTessitura
        result['AllTessitura'] = classification.df['Tessitura'].values
        return result

    def classifyText(self, filename, optionModeText=False):
        result = self.classify(filename, speech=True,
                               optionModeText=optionModeText)
        self.resultText = result
        return result

    def classifyGlissando(self, filename):
        result = self.classify(filename, speech=False)
        self.resultGlissando = result
        return result

    def joinClassifiers(self, comb=False):
        join = JoinClassifiers(
            self.resultText['PercentTessitura'], self.resultGlissando['PercentTessitura'], self.resultGlissando['AllTessitura'])
        result_join = join.joinResults()
        return result_join


class JoinClassifiers:
    def __init__(self, resultTexto, resultGlissando, tessituraG):
        self.resultTexto = resultTexto
        self.resultGlissando = resultGlissando
        self.tessituras = tessituraG
        return

    def joinResults(self, weightTexto=0.4, weightGlissando=0.6):
        texto = self.resultTexto['PercentTessitura']
        glissando = self.resultGlissando['PercentTessitura']
        rangeTexto = self.resultTexto['IndexTessitura']
        rangeGlissando = self.resultGlissando['IndexTessitura']
        maxTexto = np.max(np.array(texto))
        maxGlissando = np.max(np.array(glissando))
        indexMaxTexto = np.where(texto == maxTexto)[0]
        indexMaxGlissando = np.where(glissando == maxGlissando)[0]
        tessiturasincludedTexto = np.where(np.array(texto) != 0)[0]
        tessiturasincludedGlissando = np.where(np.array(glissando) != 0)[0]
        intersectTessituras = tessiturasincludedTexto[np.in1d(
            tessiturasincludedTexto, tessiturasincludedGlissando)]
        warning = False
        if maxGlissando == 100 or (maxTexto == 100 and indexMaxTexto[0] == 1):
            weightGlissando = 0.7
            weightTexto = 0.3
        if maxGlissando == 100 and indexMaxGlissando[0] == 0 and tessiturasincludedTexto[0] != 0 and tessiturasincludedTexto[-1] == 2:
            # weightGlissando=0.3
            # weightTexto=0.7
            warning = True
        g = np.array(glissando)*weightGlissando
        t = np.array(texto)*weightTexto
        result = g+t
        # Normalizar a un rango de sólo dos
        indexSorted = sorted(range(len(result)),
                             key=lambda k: result[k], reverse=False)
        if indexSorted[0] == 1:
            result[indexSorted[1]] = 0
        else:
            result[indexSorted[0]] = 0
        result = result/sum(result)*100
        tessitura_max = np.max(np.array(result))
        tessitura_index = np.where(result == tessitura_max)[0]
        tessituraJoin = [self.tessituras[i] for i in tessitura_index]
        # Generar alerta si el resultado no está incluido en el rango de la voz del texto
        if len(intersectTessituras) == 0:
            warning = True
        resultJoin = {'Result': result, 'IndexTessitura': tessitura_index,
                      'Tessitura': tessituraJoin, 'Warning': warning}
        return resultJoin

    def joinResultsOld(self, weightTexto=0.4, weightGlissando=0.6, comb=False):
        if comb == True:
            texto = self.resultTexto['percentAllNormalize']
            glissando = self.resultGlissando['percentAllNormalize']
        else:
            texto = self.resultTexto['PercentTessitura']
            glissando = self.resultGlissando['PercentTessitura']
        rangeTexto = self.resultTexto['IndexTessitura']
        rangeGlissando = self.resultGlissando['IndexTessitura']

        maxTexto = np.max(np.array(texto))
        maxGlissando = np.max(np.array(glissando))
        indexMaxTexto = np.where(texto == maxTexto)[0]
        indexMaxGlissando = np.where(glissando == maxGlissando)[0]
        if len(indexMaxGlissando) == 1:
            weightGlissando = 0.9
            weightTexto = 0.1
        g = np.array(glissando)*weightGlissando
        t = np.array(texto)*weightTexto
        result = g+t
        tessitura_max = np.max(np.array(result))
        tessitura_index = np.where(result == tessitura_max)[0]
        tessitura = np.array(self.tessituras[tessitura_index]).tolist()
        resultJoin = {'Result': result,
                      'IndexTessitura': tessitura_index, 'Tessitura': tessitura}
        return resultJoin


class ClassifyByNotes:
    noteNames = ["C", "C#", "D", "D#", "E",
                 "F", "F#", "G", "G#", "A", "A#", "B"]
    noteNames_Latin = ["Do", "Do#", "Re", "Re#", "Mi",
                       "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
    tessitura = ['Soprano', 'Mezzosoprano',
                 'Contralto', 'Tenor', 'Barítono', 'Bajo']
    typeTessitura = ['Woman', 'Woman', 'Woman', 'Man', 'Man', 'Man']

    minRange = [30, 27, 24, 19, 16, 12]
    maxRange = [62, 59, 55, 47, 43, 40]
    minRangeSpeech = [32, 29, 26, 22, 19, 16]
    maxRangeSpeech = [44, 41, 38, 34, 31, 28]

    def __init__(self, dicNotes=None, Type=None, speech=False, initNote=False, minNote=None, maxNote=None, latin=False):
        if initNote == False:
            self.initFromResults(dicNotes, Type, speech)
        else:
            self.initFromNotes(minNote, maxNote, Type, speech, latin)
        return

    def initFromResults(self, dicNotes, Type, speech):
        self.result = dicNotes
        self.Type = Type
        self.note = dicNotes['ResultNotes']['Note']
        self.number = dicNotes['ResultNotes']['Number']
        self.nameNotes = dicNotes['BasicMusicNote']['NoteName']
        self.nameNotesLatin = dicNotes['BasicMusicNote']['NoteNameLatin']
        self.modeResult = dicNotes['ResultNotes']['Mode']
        self.speech = speech
        if speech:
            self.minR = self.minRangeSpeech
            self.maxR = self.maxRangeSpeech
        else:
            self.minR = self.minRange
            self.maxR = self.maxRange
        self.rangeNotes = {'Tessitura': self.tessitura,
                           'Type': self.typeTessitura, 'MinRange': self.minR, 'MaxRange': self.maxR}
        self.rangeNotes_Woman = {
            'Tessitura': self.tessitura[0:3], 'Type': self.typeTessitura[0:3], 'MinRange': self.minR[0:3], 'MaxRange': self.maxR[0:3]}
        self.rangeNotes_Man = {
            'Tessitura': self.tessitura[3:6], 'Type': self.typeTessitura[3:6], 'MinRange': self.minR[3:6], 'MaxRange': self.maxR[3:6]}
        self.df_Man = pd.DataFrame(self.rangeNotes_Man, columns=[
                                   'Tessitura', 'Type', 'MinRange', 'MaxRange'])
        self.df_Woman = pd.DataFrame(self.rangeNotes_Woman, columns=[
                                     'Tessitura', 'Type', 'MinRange', 'MaxRange'])
        minResult = min(self.number)
        maxResult = max(self.number)
        self.minResult = minResult
        self.maxResult = maxResult
        self.rangeResult = np.array(range(minResult, maxResult+1))
        self.df = self.df_Woman if self.Type == 'Woman' else self.df_Man
        self.rangeTessituras = [np.array(
            range(mi, ma+1)) for mi, ma in zip(self.df['MinRange'], self.df['MaxRange'])]
        return

    def initFromNotes(self, minNote, maxNote, Type, speech=False, latin=False, modeNote=None):
        self.Type = Type
        self.speech = speech
        self.modeNote = modeNote
        if speech:
            self.minR = self.minRangeSpeech
            self.maxR = self.maxRangeSpeech
        else:
            self.minR = self.minRange
            self.maxR = self.maxRange
        self.rangeNotes = {'Tessitura': self.tessitura,
                           'Type': self.typeTessitura, 'MinRange': self.minR, 'MaxRange': self.maxR}
        self.rangeNotes_Woman = {
            'Tessitura': self.tessitura[0:3], 'Type': self.typeTessitura[0:3], 'MinRange': self.minR[0:3], 'MaxRange': self.maxR[0:3]}
        self.rangeNotes_Man = {
            'Tessitura': self.tessitura[3:6], 'Type': self.typeTessitura[3:6], 'MinRange': self.minR[3:6], 'MaxRange': self.maxR[3:6]}
        if Type == 'Man':
            self.df_Man = pd.DataFrame(self.rangeNotes_Man, columns=[
                                       'Tessitura', 'Type', 'MinRange', 'MaxRange'])
            self.df = self.df_Man
        else:
            self.df_Woman = pd.DataFrame(self.rangeNotes_Woman, columns=[
                                         'Tessitura', 'Type', 'MinRange', 'MaxRange'])
            self.df = self.df_Woman
        notebase = self.noteNames_Latin if latin else self.noteNames
        minResult = UtilsFunctions.compute_NumberFromNote(minNote, notebase)
        maxResult = UtilsFunctions.compute_NumberFromNote(maxNote, notebase)
        self.minResult = minResult
        self.maxResult = maxResult
        self.rangeResult = np.array(range(minResult, maxResult+1))
        self.rangeTessituras = [np.array(
            range(mi, ma+1)) for mi, ma in zip(self.df['MinRange'], self.df['MaxRange'])]
        return

    def classify(self):
        # Buscar intersección entre el resultado y los rangos
        rangeTessitura = [[np.array(range(mi, ma+1))]
                          for mi, ma in zip(self.df['MinRange'], self.df['MaxRange'])]
        intersection, highest, lowest = self.intersection_Range()
        differenceBetweenAll = [[self.differenceList(intersection[i], intersection[j]) for j in range(
            0, self.df.shape[0]) if abs(j-i) == 1] for i in range(0, self.df.shape[0])]
        differenceBetweenAllO = differenceBetweenAll.copy()
        countdifferenceAll = [sum([len(i) for i in d])
                              for d in differenceBetweenAll]
        if len(highest) > 0:
            countdifferenceAll[0] = countdifferenceAll[0]+len(highest)
        if len(lowest) > 0:
            countdifferenceAll[-1] = countdifferenceAll[-1]+len(lowest)
        if countdifferenceAll[0] >= countdifferenceAll[-1]:
            countdifferenceAll[1] = max(
                0, countdifferenceAll[1]+countdifferenceAll[-1]-len(differenceBetweenAll[1][1]))
            countdifferenceAll[-1] = 0
        else:
            if countdifferenceAll[0] > 0:
                countdifferenceAll[1] = max(
                    0, countdifferenceAll[1]+countdifferenceAll[0]-len(differenceBetweenAll[1][0]))
                countdifferenceAll[0] = 0
        countdifferenceAll[0] = max(
            0, countdifferenceAll[0]*(countdifferenceAll[0]+1)/2)
        if self.speech and countdifferenceAll[0] == 0 and countdifferenceAll[-1] == 0 and countdifferenceAll[1] != 0 and (len(intersection[1])-len(intersection[0])) <= 4:
            countdifferenceAll[0] = max(
                0, 2*len(intersection[0])-len(rangeTessitura[0][0]))
            countdifferenceAll[1] = countdifferenceAll[1]+len(intersection[1])
        sumcountdifAll = sum(countdifferenceAll)
        if sumcountdifAll == 0:  # Está contenido en todos los rangos, le pongo un porciento pesando las notas más altas
            rest = intersection[0][0]-rangeTessitura[0][0][0]
            countdifferenceAll[0] = 0+(rest)*(rest+1)/2
            countdifferenceAll[1] = max(0, len(intersection[0])-rest)
            countdifferenceAll[2] = 0
            sumcountdifAll = sum(countdifferenceAll)
        percentTessituraAll = [
            count/sumcountdifAll for count in countdifferenceAll]
        if max(percentTessituraAll) == 0.5:
            if percentTessituraAll[0] == 0.5:
                percentTessituraAll[0] = 0.6
                percentTessituraAll[1] = 0.4
            else:
                percentTessituraAll[1] = 0.6
                percentTessituraAll[2] = 0.4
        percentTessituraAll100 = [i*100 for i in percentTessituraAll]
        return percentTessituraAll100, highest, lowest

    def classify_Text_Mode(self):
        # Buscar el rango donde está incluida la moda más alto
        rangeTessitura = [[np.array(range(mi, ma+1))]
                          for mi, ma in zip(self.df['MinRange'], self.df['MaxRange'])]
        intersection, highest, lowest = self.intersection_Range()
        modeIncluded = [i for i, iR in enumerate(
            intersection) if self.modeResult in iR]
        classifyByMode = modeIncluded[0]
        # El mínimo Rango donde cae el máximo
        maxIncluded = [i for i, iR in enumerate(
            intersection) if self.maxResult in iR]
        maxMinRange = maxIncluded[-1]
        percentTessituraAll = [0]*len(rangeTessitura)
        if classifyByMode == maxMinRange:
            percentTessituraAll[classifyByMode] = 1
        else:
            if classifyByMode == 2 and maxMinRange == 0:
                percentTessituraAll[2] = 0.6
                percentTessituraAll[1] = 0.4
            else:
                if classifyByMode == 0 and maxMinRange == 2:
                    percentTessituraAll[0] = 0.6
                    percentTessituraAll[1] = 0.4
                else:
                    percentTessituraAll[classifyByMode] = 0.8
                    percentTessituraAll[maxMinRange] = 0.2
        percentTessituraAll100 = [i*100 for i in percentTessituraAll]
        return percentTessituraAll100, highest, lowest

    def dic_classify(self, optionModeText=False):
        if optionModeText and self.speech:
            percentTessitura, highest, lowest = self.classify_Text_Mode()
        else:
            percentTessitura, highest, lowest = self.classify()
        # Tessitura con mayor porciento
        maxT = max(percentTessitura)
        indexTessitura = [i for i, p in enumerate(
            percentTessitura) if p == maxT]
        tessituraResult = [self.df['Tessitura'][i] for i in indexTessitura]
        dicClassification = {'Tessitura': tessituraResult, 'IndexTessitura': indexTessitura,
                             'PercentTessitura': percentTessitura, 'HighestNotes': highest, 'LowestNotes': lowest}
        return dicClassification

    def intersection_Range(self):
        intersection = [self.rangeResult[np.in1d(
            self.rangeResult, rt)] for rt in self.rangeTessituras]
        highest = [r for r in self.rangeResult if r >
                   self.rangeTessituras[0][-1]]
        lowest = [r for r in self.rangeResult if r <
                  self.rangeTessituras[-1][0]]
        intersection[0] = np.array(np.concatenate(
            (intersection[0], highest)), np.int64)
        intersection[-1] = np.array(np.concatenate(
            (intersection[-1], lowest)), np.int64)
        return intersection, highest, lowest

    def differenceList(self, list1, list2):
        list_difference = [item for item in list1 if item not in list2]
        return list_difference


class UtilsFunctions:

    noteNames = ["C", "C#", "D", "D#", "E",
                 "F", "F#", "G", "G#", "A", "A#", "B"]
    noteNames_Latin = ["Do", "Do#", "Re", "Re#", "Mi",
                       "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

    @staticmethod
    def compute_Notes(notes, noteBases=noteNames):
        mNoteRealFinal = [noteBases[x % 12]+str(x//12+1) for x in notes]
        return mNoteRealFinal

    @staticmethod
    def compute_Note(note, latin=False):
        if latin == False:
            noteBases = UtilsFunctions.noteNames
        else:
            noteBases = UtilsFunctions.noteNames_Latin
        return noteBases[note % 12]+str(note//12+1)

    @staticmethod
    def createNotes(noteNames, numberInit=1, numberEnd=6):
        allNotes = [n+str(i) for i in range(numberInit, numberEnd+1)
                    for n in noteNames]
        return allNotes

    @staticmethod
    def compute_NumberFromNote(note, noteBases=noteNames):
        allNotes = np.array(UtilsFunctions.createNotes(noteBases))
        number = np.where(allNotes == note)[0]
        return number[0]


class DetectNotesFromCQT:

    noteNames = ["C", "C#", "D", "D#", "E",
                 "F", "F#", "G", "G#", "A", "A#", "B"]
    noteNames_Latin = ["Do", "Do#", "Re", "Re#", "Mi",
                       "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

    def __init__(self, pathFile, sr=44100, nfft=2048, overlap=0.5, n_bins=84, mag_exp=1, pre_post_max=5, cqt_threshold=-30, backtrack=False, fragment=False):
        self.nfft = nfft                           # length of the FFT window
        self.overlap = overlap                     # Hop overlap percentage
        # Number of samples between successive frames
        self.hop_length = int(nfft*(1-overlap))
        self.n_bins = n_bins                       # Number of frequency bins
        self.mag_exp = mag_exp                     # Magnitude Exponent
        # Pre- and post- samples for peak picking
        self.pre_post_max = pre_post_max
        # -61    # Threshold for CQT dB levels, all values below threshold are set to -120 dB
        self.cqt_threshold = cqt_threshold
        self.backtrack = backtrack
        self.sr = sr
        if fragment == False:
            self.x, self.sr = librosa.load(pathFile, sr=sr, mono=True)
            self.duration = (self.x.shape[0]/self.sr)
        return

    # CQT Function
    def calc_cqt(self, x, sr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C = librosa.cqt(x, sr=sr, hop_length=self.hop_length,
                            fmin=None, n_bins=self.n_bins)
            C_mag = librosa.magphase(C)[0]**self.mag_exp
            CdB = librosa.core.amplitude_to_db(C_mag, ref=np.max)
        return CdB

    # CQT Threshold
    def cqt_thresholded_(self, cqt, threshold):
        new_cqt = np.copy(cqt)
        new_cqt[new_cqt < threshold] = -120
        return new_cqt

    def cqt_thresholded(self, cqt):
        return self.cqt_thresholded_(cqt, self.cqt_threshold)

    # Quitar filas menores que C2(12)
    def cqt_from_limit(self, cqt, limit='C2'):
        row = UtilsFunctions.compute_NumberFromNote(limit)
        cqt[np.array(range(0, row)), :] = -120
        return cqt

    # Onset Envelope from Cqt
    def calc_onset_env(self, cqt):
        return librosa.onset.onset_strength(S=cqt, sr=self.sr, aggregate=np.mean, hop_length=self.hop_length)

    # Onset from Onset Envelope
    def calc_onset(self, cqt):
        onset_env = self.calc_onset_env(cqt)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env,
                                                  sr=self.sr, units='frames',
                                                  hop_length=self.hop_length,
                                                  backtrack=self.backtrack,
                                                  pre_max=self.pre_post_max,
                                                  post_max=self.pre_post_max)
        onset_boundaries = np.concatenate([[0], onset_frames, [cqt.shape[1]]])
        onset_times = librosa.frames_to_time(
            onset_boundaries, sr=self.sr, hop_length=self.hop_length)
        return [onset_times, onset_boundaries, onset_env]

    # Onset from the CQT thresholded
    def calc_onset_CQT(self):
        return self.calc_onset_CQT_other(self.x, self.sr)

    def calc_onset_CQT_other(self, x, sr):
        CdB = self.calc_cqt(x, sr)
        new_cqt = self.cqt_thresholded(CdB)
        onsets = self.calc_onset(new_cqt)
        new_cqt[:, onsets[1][0]:onsets[1][1]] = -120
        self.cqt = new_cqt
        self.onsets = onsets
        self.CdB = CdB
        return CdB, new_cqt, onsets

    def estimate_tempo(self):
        tempo, beats = librosa.beat.beat_track(y=None, sr=self.sr, onset_envelope=self.onsets[2], hop_length=self.hop_length,
                                               start_bpm=120.0, tightness=100, trim=True, bpm=None,
                                               units='frames')
        tempo = int(2*round(tempo/2))
        return tempo

    def detect_silents(self, cqt, limitwidth=2):
        maxbyColumn = cqt.max(axis=0)
        columnEmpty = np.where(maxbyColumn == -120)[0]
        consecutiveEmpty = np.split(
            columnEmpty, np.where(np.diff(columnEmpty) != 1)[0]+1)
        consecutiveEmpty = [c for c in consecutiveEmpty if c.size > limitwidth]
        return consecutiveEmpty

    def calc_onset_without_silents(self, cqt, onset_boundaries):
        #onset_times, onset_boundaries, onset_env=self.calc_onset(cqt)
        onsets = np.array([[onset_boundaries[i], onset_boundaries[i+1]-1]
                           for i, o in enumerate(onset_boundaries[0:-1])])
        newonsets = []
        # silents for each onsets and readjust
        for o in onsets:
            cqto = cqt[:, o[0]:o[1]+1]
            rangeOnset = np.array(range(o[0], o[1]+1))
            maxbyColumn = cqt.max(axis=0)
            columnEmpty = np.where(maxbyColumn == -120)[0]
            ofiltred = np.setdiff1d(rangeOnset, columnEmpty)
            newO = np.split(ofiltred, np.where(np.diff(ofiltred) != 1)[0]+1)
            for no in newO:
                newonsets.append(no)
        if len(newonsets) > 0:
            newonset_boundaries = np.array(
                [np.array([o[0], o[-1]]) for o in newonsets if len(o) > 0])
        else:
            newonset_boundaries = np.array([])
        #onset_timesI = librosa.frames_to_time(newonset_boundaries[:,0], sr=self.sr, hop_length=self.hop_length)
        #onset_timesE = librosa.frames_to_time(newonset_boundaries[:,1], sr=self.sr, hop_length=self.hop_length)
        return newonset_boundaries  # , onset_timesI, onset_timesE

    def detect_bands(self, cqt, consecutiveEmpty):
        # Buscar inicio de índices de bandas entre silencios
        if len(consecutiveEmpty) > 0:
            firstFrame = self.onsets[1][1]
            countBands = len(consecutiveEmpty)-1
            endFrame = np.size(
                cqt, 1)-1 if (consecutiveEmpty[-1][-1] < np.size(cqt, 1)-1) else consecutiveEmpty[-1][0]-1
            indexInitBands = [e[-1]+1 for e in consecutiveEmpty if (
                e[-1]+1 >= firstFrame) and (e[-1] < np.size(cqt, 1)-1)]
            indexEndBands = [
                e[0]-1 for e in consecutiveEmpty if (e[0] >= firstFrame) and (e[-1] <= endFrame)]
            if indexEndBands == [] or indexEndBands[-1] != endFrame:
                indexEndBands.append(endFrame)
            if len(indexEndBands) != len(indexInitBands) and indexInitBands[0] > indexEndBands[0]:
                if len(indexEndBands) > 1:
                    indexEndBands.pop(0)
                else:
                    indexInitBands = [firstFrame]+indexInitBands
            indexBands = np.vstack(
                (np.array(indexInitBands), np.array(indexEndBands))).T
            d = indexBands[:, 1]-indexBands[:, 0]
            indexDeleted = np.where(d < 5)
            if indexDeleted != []:
                for i in indexDeleted:
                    indexBands = np.delete(indexBands, i, axis=0)
            # Eliminar bandas inconsistentes por diferencias muy grandes de con respecto a las otras
        return indexBands

    def detect_cqt_firstBand_onset_2(self, cqt):
        wR = 2
        consecutiveEmpty = self.detect_silents(cqt)
        if len(consecutiveEmpty) == 0:
            indexBands = [[self.onsets[1][1], cqt.shape[1]-1]]
        else:
            indexBands = self.detect_bands(cqt, consecutiveEmpty)
        oI = self.onsets[1]
        onsetbyiB = [
            oI[np.searchsorted(oI, [b[0]-1], side='right')[0]] for b in indexBands]
        new_cqt = np.copy(self.CdB)
        columns = np.array(range(0, np.size(new_cqt, 0)))
        indexposibleBand = 0
        listBands = []
        listBandsRow = []
        for bi, b in enumerate(indexBands):
            deleteband = False
            # Tomar de la matriz original sin cambios de dB
            subCQT = new_cqt[:, b[0]:b[1]+1]
            subCQT30 = cqt[:, b[0]:b[1]+1]
            # Debe comenzar en onset más cercano a b[0]
            init = cqt[:, onsetbyiB[bi]]
            colInint = onsetbyiB[bi]
            if init.max() == -120:
                init = cqt[:, b[0]]
                colInint = b[0]
            posibleFirst = np.where(init > -120)[0]
            consposibleFirst = np.split(
                posibleFirst, np.where(np.diff(posibleFirst) != 1)[0]+1)
            # Chequear el primero si está muy pequeño seguir al próximo si hay próximo
            sub = cqt[consposibleFirst[0], colInint]
            centerBand = consposibleFirst[0][sub.argmax()]
            # Si es una banda aislada o si está muy baja...
            ncols = 12
            sub5 = cqt[max(centerBand-2, 0):min(centerBand+2+1, cqt.shape[0]),
                       colInint:min(colInint+ncols, colInint+(b[1]-b[0])-1)]
            maxsub5 = np.max(sub5, axis=0)
            maxsub5Row = np.max(sub5, axis=1)
            maxsub5 = [1 if m != -120 else 0 for m in maxsub5]
            # Aislada es que tenga filas encima y abajo vacías y columnas al final también
            colsempty = (sum(maxsub5) < ncols)
            colsnotempty = [mi for mi, m in enumerate(maxsub5) if m != -0]
            rowsempty = [mi for mi, m in enumerate(maxsub5Row) if m == -120]
            isolated = len(rowsempty) > 0 and (
                rowsempty[0] == 0 and rowsempty[-1] == sub5.shape[0]-1) and colsempty
            if np.sum(maxsub5) <= 5 or isolated:
                if len(consposibleFirst) > indexposibleBand+1:
                    indexposibleBand = 1
                    sub = cqt[consposibleFirst[1], colInint]
                    centerBand = consposibleFirst[1][sub.argmax()]
                else:
                    deleteband = True
            ##################################################################################
            # Borrar antes de la primera banda
            if bi == 0:
                new_cqt[:, 0:colInint] = -120
            anterior = centerBand
            if bi == 0:
                previousBandCenter = centerBand
            # Si la banda está inconsistente como ruido quitar y poner vacío
            if centerBand < 12 or (bi > 0 and len(listBands) > 0 and (b[0]-listBands[-1][1]) < 12 and abs(previousBandCenter-centerBand) > 12) or deleteband:
                ###poner -120
                for index in range(b[0], b[1]+1):
                    new_cqt[:, index] = -120
            else:
                noise = True
                newnoise = False
                while noise:
                    indexRow = []
                    noise = False
                    if newnoise:
                        if len(consposibleFirst) > indexposibleBand+1:
                            indexposibleBand = indexposibleBand+1
                            sub = cqt[consposibleFirst[indexposibleBand], colInint]
                            centerBand = consposibleFirst[indexposibleBand][sub.argmax(
                            )]
                        else:
                            for i in range(b[0], b[1]+1):
                                new_cqt[:, i] = -120
                            break
                    for i in range(b[0], b[1]+1):
                        if i != b[0]:
                            if abs(int(subCQT[:, i-b[0]].argmax())-int(anterior)) < 5:
                                centerBand = subCQT[:, i-b[0]].argmax()
                            else:
                                anterior = centerBand
                                centerBand = anterior-2 + \
                                    subCQT[max(
                                        0, anterior-2):min(anterior+2, np.size(new_cqt, 0)-1), i-b[0]].argmax()
                                if centerBand > 10:
                                    c = subCQT30[max(
                                        0, anterior-2):min(anterior+2, np.size(new_cqt, 0)-1), i-b[0]]
                            if centerBand <= 10:
                                noise = True
                                newnoise = True
                                indexposibleBand = indexposibleBand+1
                                break
                        indexRowN = np.array(
                            range(max(0, centerBand-wR), min(centerBand+wR+1, np.size(new_cqt, 0))))
                        indexRow.append(indexRowN)
                    if not noise and len(indexRow) > 0:
                        listBands.append(b)
                        listBandsRow.append(indexRow)
                        for i in range(b[0], b[1]+1):
                            new_cqt[~np.isin(
                                columns, indexRow[i-b[0]]), i] = -120
                        previousBandCenter = centerBand
            # Poner -120 entre bandas
            for i in range(b[0], b[1]+1):
                if bi != 0:
                    for index in range(indexBands[bi-1][1]+1, b[0]):
                        new_cqt[:, index] = -120
        # Poner -120 al final de la última banda
        lastIndex = indexBands[-1][1]
        for i in range(lastIndex+1, new_cqt.shape[1]):
            new_cqt[:, i] = -120
        # Poner -120 en los espacios
        for ce in consecutiveEmpty:
            new_cqt[:, ce] = -120
        return new_cqt

    def calc_newonset_with_spaces(self, onsets, cqt, emptyFrame=5):
        consecutiveEmpty = self.detect_silents(cqt, limitwidth=emptyFrame)
        first_last_ce = [[c[0], c[-1]] for c in consecutiveEmpty]
        first_ce = np.array([c[0] for c in consecutiveEmpty])
        last_ce = [c[-1] for c in consecutiveEmpty]
        newonsets = np.array([onsets[1][0]])
        indexOnsetEmpty = []
        for i in range(1, len(onsets[1])-1):
            if onsets[1][i] > newonsets[-1]:
                m = np.where((first_ce >= onsets[1][i]) & (
                    first_ce < onsets[1][i+1]))[0]
                newonsets = np.append(newonsets, [onsets[1][i]])
                if m.size > 0:
                    for index in m:
                        if first_ce[index] > onsets[1][i]:
                            newonsets = np.append(newonsets, [first_ce[index]])
                        indexOnsetEmpty.append(len(newonsets)-1)
                        newonsets = np.append(
                            newonsets, [min(last_ce[index]+1, onsets[1][-1])])

        if onsets[1][-1] > newonsets[-1]:
            newonsets = np.append(newonsets, onsets[1][-1])
        return newonsets, indexOnsetEmpty

    # Adiciona divisiones de onset donde hay espacios superiores a emptyFrame=5
    # Devuelve los nuevos onsets y los índices de los onsets vacíos adicionados
    def onsets_with_blank_spaces(self, onsets, new_cqt, emptyFrame=5):
        emptyFrame = emptyFrame
        newonsets = np.array([onsets[1][0]])
        newonsets = np.append(newonsets, [onsets[1][1]])
        onsetsEmpty_Index = []
        for i in range(2, len(onsets[1])):
            # Chequear vacíos seguidos primero que sean mayores a emptyFrame
            subCQT = new_cqt[:, onsets[1][i-1]:onsets[1][i]]
            maxbyColumn = subCQT.max(axis=0)
            columnWithdata = np.where(maxbyColumn > -120)
            columnWithoutdata = np.where(maxbyColumn == -120)[0]
            consecutiveEmpty = np.split(columnWithoutdata, np.where(
                np.diff(columnWithoutdata) != 1)[0]+1)
            consecutiveEmpty = [
                c for c in consecutiveEmpty if c.size > emptyFrame]
            for indexC, c in enumerate(consecutiveEmpty):
                diffAnt = newonsets[len(newonsets)-1]-c[0]
                if diffAnt != 0:
                    newonsets = np.append(
                        newonsets, newonsets[len(newonsets)-1]+c[0])
                onsetsEmpty_Index.append(len(newonsets)-1)
                newonsets = np.append(
                    newonsets, newonsets[len(newonsets)-1]+len(c))
            diffAnt = newonsets[len(newonsets)-1]-onsets[1][i]
            if diffAnt != 0:
                newonsets = np.append(newonsets, onsets[1][i])
        self.newonsets = newonsets
        return newonsets, onsetsEmpty_Index

    def compute_time(self, onset):
        onsetTime = librosa.frames_to_time(
            onset, sr=self.sr, hop_length=self.hop_length)
        return onsetTime

    # Eliminar repetido
    # Asociar tiempo con onsets-juntados
    # Quitar los vacíos
    def delete_repeted_empty_Note(self, maxNotebyOnset, onsetInit, onsetEnd):
        onsetTime = self.compute_time(self.newonsets)
        aNote = np.array(maxNotebyOnset, dtype=np.uint32)
        diffNote = np.diff(aNote, 1)
        diff = np.array([1])
        diff = np.append(diff, diffNote)
        diffF = np.array(diffNote)
        diffF = np.append(diffF, [1])
        arrayNotesnp = aNote[diff != 0]
        # Asociar tiempo con onsets-juntados
        cond = (diff != 0)
        condF = (diffF != 0)
        onsetInit = onsetInit[cond]
        onsetEnd = onsetEnd[condF]
        # Quitar los vacíos
        index_empty = np.where(arrayNotesnp == 0)[0]
        arrayNotesnp = arrayNotesnp[arrayNotesnp != 0]
        onsetInit = onsetInit[sorted(
            list(set(range(len(onsetInit))) - set(index_empty)))]
        onsetEnd = onsetEnd[sorted(
            list(set(range(len(onsetEnd))) - set(index_empty)))]
        return arrayNotesnp, onsetInit, onsetEnd

    # combinar notas inconsistentes seguidas que cambian indistintamente con sólo una diferencia de +-1
    # Buscar índices con cambios de signo en notas seguidas
    def delete_incoherents_Note(self, arrayNotesnp, onsetInit, onsetEnd):
        newNotes = [arrayNotesnp[0]]
        newOnsetInit = [onsetInit[0]]
        newOnsetEnd = [onsetEnd[0]]
        inc = 0
        diffOnset = onsetEnd-onsetInit
        if arrayNotesnp.size > 2:
            for i in range(1, arrayNotesnp.size-1):
                if inc != 1:
                    if inc == 0 and arrayNotesnp[i-1] == arrayNotesnp[i+1] and abs(int(arrayNotesnp[i])-int(arrayNotesnp[i-1])) == 1 and onsetEnd[i-1]+1 == onsetInit[i] and onsetEnd[i]+1 == onsetInit[i+1]:
                        if diffOnset[i-1]+diffOnset[i+1] >= diffOnset[i]:
                            newOnsetEnd[-1] = onsetEnd[i+1]
                            inc = 1
                        else:
                            newNotes[-1] = arrayNotesnp[i]
                            newOnsetEnd[-1] = onsetEnd[i]
                    else:
                        newNotes.append(arrayNotesnp[i])
                        newOnsetInit.append(onsetInit[i])
                        newOnsetEnd.append(onsetEnd[i])
                        inc = 0
                else:
                    inc = 2
            if inc == 0:
                newNotes.append(arrayNotesnp[-1])
                newOnsetInit.append(onsetInit[-1])
                newOnsetEnd.append(onsetEnd[-1])

        arrayNotesnp = np.array(newNotes)
        onsetInit = np.array(newOnsetInit)
        onsetEnd = np.array(newOnsetEnd)
        return arrayNotesnp, onsetInit, onsetEnd

    # Calcular nota por cada onset nuevo, sin tener en cuenta los vacíos
    # Máximo de los promedios de las notas por onset
    def compute_Note_by_onset(self, new_cqt, newonsets):
        onsetTime = self.compute_time(newonsets)
        onsetInit = newonsets[0:len(newonsets)-1]
        onsetEnd = newonsets[1:len(newonsets)]-1
        maxNotebyOnset = []
        for i in range(0, len(onsetInit)):
            if onsetInit[i] < onsetEnd[i]:
                subCQT = new_cqt[:, onsetInit[i]:onsetEnd[i]]
                maxbyRow = subCQT.max(axis=1)
                rowWithdata = np.where(maxbyRow > -120)
                # Calcular promedio
                average = subCQT.mean(axis=1)[rowWithdata]
                if average.size > 0:
                    maxNote = rowWithdata[0][average.argmax()]
                    maxNotebyOnset.append(maxNote)
                else:
                    maxNotebyOnset.append(0)  # Si no hay nota en el onset
            else:
                maxNotebyOnset.append(0)  # Si no hay nota en el onset
        return maxNotebyOnset, onsetInit, onsetEnd

        # Calcular nota por cada onset nuevo, sin tener en cuenta los vacíos
    # Máximo de los promedios de las notas por onset
    def compute_Note_by_onset_CdB(self, new_cqt, newonsets):
        onsetTime = self.compute_time(newonsets)
        onsetInit = newonsets[0:len(newonsets)-1]
        onsetEnd = newonsets[1:len(newonsets)]-1
        maxNotebyOnset = []
        for i in range(0, len(onsetInit)):
            if onsetInit[i] < onsetEnd[i]:
                subCQT = new_cqt[:, onsetInit[i]:onsetEnd[i]]
                subCQT_CdB = self.CdB[:, onsetInit[i]:onsetEnd[i]]
                maxbyRow = subCQT.max(axis=1)
                rowWithdata = np.where(maxbyRow > -120)
                # Calcular promedio
                average = subCQT_CdB.mean(axis=1)[rowWithdata]
                if average.size > 0:
                    maxNote = rowWithdata[0][average.argmax()]
                    maxNotebyOnset.append(maxNote)
                else:
                    maxNotebyOnset.append(0)  # Si no hay nota en el onset
            else:
                maxNotebyOnset.append(0)  # Si no hay nota en el onset
        return maxNotebyOnset, onsetInit, onsetEnd

    def convert_N_to_Note(self, x, latin=False):
        if latin:
            return self.noteNames_Latin[x % 12]+str(x//12+1)
        else:
            return self.noteNames[x % 12]+str(x//12+1)

    def compute_Notes(self, notes):
        mNoteRealFinal = [self.convert_N_to_Note(x) for x in notes]
        return mNoteRealFinal

    def compute_Notes_Latin(self, notes):
        mNoteRealFinal = [self.convert_N_to_Note(x, latin=True) for x in notes]
        return mNoteRealFinal

     # Juntar los que son pequeños, recalcular los índices de los vacíos.
    def onsets_with_longer_size(self, newonsets, onsetsEmpty_Index, numberFrame=8):
        onsetsLonger = np.copy(newonsets)
        onsetsDiff = np.diff(onsetsLonger) - 1
        for i in range(2, len(onsetsLonger)-1):
            if i not in onsetsEmpty_Index and onsetsDiff[i-1] < numberFrame:
                if (i-1) not in onsetsEmpty_Index:
                    onsetsLonger[i] = -1
                else:
                    if (i+1) in onsetsEmpty_Index:
                        onsetsLonger[i+1] = -1
        indexJ = np.where(onsetsLonger == -1)[0]
        onsetsLonger = onsetsLonger[onsetsLonger != -1]
        onsetsEmpty_Index = [
            i-np.count_nonzero(indexJ < i) for i in onsetsEmpty_Index]
        self.newonsets = onsetsLonger
        return onsetsLonger, onsetsEmpty_Index

    def delete_short_notes(self, arrayNotesnp, onsetInit, onsetEnd, limitframe=5):
        newNotes = []
        newOnsetInit = []
        newOnsetEnd = []
        diffOnset = onsetEnd-onsetInit
        ant = False
        if arrayNotesnp.size > 1:
            for i in range(0, arrayNotesnp.size):
                if ant == True:
                    ant = False
                else:
                    if diffOnset[i] > limitframe:
                        newNotes.append(arrayNotesnp[i])
                        newOnsetInit.append(onsetInit[i])
                        newOnsetEnd.append(onsetEnd[i])
                    else:
                        if i < arrayNotesnp.size-1 and abs(int(arrayNotesnp[i+1])-int(arrayNotesnp[i])) == 1 and onsetInit[i+1]-onsetEnd[i] < 2 and diffOnset[i+1] > limitframe:
                            ant = True
                            newNotes.append(arrayNotesnp[i+1])
                            newOnsetInit.append(onsetInit[i])
                            newOnsetEnd.append(onsetEnd[i+1])
                        else:
                            if i > 0 and abs(int(arrayNotesnp[i-1])-int(arrayNotesnp[i])) == 1 and onsetInit[i]-onsetEnd[i-1] < 2 and diffOnset[i-1] > limitframe:
                                newOnsetEnd[-1] = onsetEnd[i]

        arrayNotesnp = np.array(newNotes)
        onsetInit = np.array(newOnsetInit)
        onsetEnd = np.array(newOnsetEnd)
        return arrayNotesnp, onsetInit, onsetEnd

    def delete_noise_under_mean_speech(self, arrayNotesnp, onsetInit, onsetEnd):
        index = []
        if self.speech and arrayNotesnp.size > 5:
            mean = np.mean(arrayNotesnp, axis=0)
            for i in range(0, arrayNotesnp.size):
                if int(mean)-int(arrayNotesnp[i]) >= 5:
                    index.append(i)
            arrayNotesnp = np.delete(arrayNotesnp, index)
            onsetInit = np.delete(onsetInit, index)
            onsetEnd = np.delete(onsetEnd, index)
        if arrayNotesnp.size > 3 and abs(int(arrayNotesnp[arrayNotesnp.size-1])-int(arrayNotesnp[arrayNotesnp.size-2])) > 10:
            arrayNotesnp = arrayNotesnp[0:-1]
            onsetInit = onsetInit[0:-1]
            onsetEnd = onsetEnd[0:-1]
        if arrayNotesnp.size > 4 and abs(int(arrayNotesnp[arrayNotesnp.size-2])-int(arrayNotesnp[arrayNotesnp.size-3])) > 10:
            arrayNotesnp = arrayNotesnp[0:len(arrayNotesnp)-2]
            onsetInit = onsetInit[0:len(onsetInit)-2]
            onsetEnd = onsetEnd[0:len(onsetEnd)-2]
        return arrayNotesnp, onsetInit, onsetEnd

    def delete_join_short_notes(self, arrayNotesnp, onsetInit, onsetEnd, limitframe=5):
        #arrayNotesnp, onsetInit, onsetEnd=self.delete_noise_under_mean_speech(arrayNotesnp, onsetInit, onsetEnd)
        # Quitar ruidos intermedios
        if self.speech:
            w = 8
        else:
            w = 12
        indexNoises = [i for i in range(1, arrayNotesnp.size) if abs(
            int(arrayNotesnp[i])-int(arrayNotesnp[i-1])) >= w]
        add = False
        if indexNoises != []:
            bandsIndex = []
            for j, index in enumerate(indexNoises):
                if add == False:
                    if j < len(indexNoises)-1 and indexNoises[j+1] == index+1:
                        bandsIndex.append([index])
                        add = True
                    else:
                        if bandsIndex == [] or (bandsIndex != [] and index-1 not in bandsIndex[-1]):
                            diffb = [i for i in range(index, arrayNotesnp.size) if abs(int(arrayNotesnp[i])-int(
                                arrayNotesnp[index])) < w and abs(int(arrayNotesnp[i])-int(arrayNotesnp[index-1])) > w]
                            consecutivediff = np.split(
                                diffb, np.where(np.diff(diffb) != 1)[0]+1)
                            bandsIndex.append(consecutivediff[0])
                else:
                    add = False
            if abs(int(arrayNotesnp[arrayNotesnp.size-1])-int(arrayNotesnp[arrayNotesnp.size-2])) > 5 and (bandsIndex == [] or (arrayNotesnp.size-1 not in bandsIndex[-1])):
                bandsIndex.append([arrayNotesnp.size-1])
            flat_bi = [item for sublist in bandsIndex for item in sublist]
            arrayNotesnp = np.delete(arrayNotesnp, flat_bi)
            onsetInit = np.delete(onsetInit, flat_bi)
            onsetEnd = np.delete(onsetEnd, flat_bi)
        return arrayNotesnp, onsetInit, onsetEnd

    # Esto puede ser si la persona es bajo, entonces se calcularán las notas más abajo
    def recomputeBands(self, cqt):
        maxCQT = np.argmax(cqt, axis=0)
        maxCQT0 = [m for m in maxCQT if m != 0]
        (_, idx, counts) = np.unique(
            maxCQT0, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode_MaxCQT = maxCQT0[index]
        min_MaxCQT = np.min(maxCQT0)
        max_MaxCQT = np.max(maxCQT0)
        mean_MaxCQT = np.mean(maxCQT0)
        subCQT = cqt[0:min(max_MaxCQT, mode_MaxCQT+12), :]
        maxCQT = np.argmax(subCQT, axis=0)
        new_cqt = cqt.copy()
        new_cqt[:, :] = -120
        for c, r in enumerate(maxCQT):
            new_cqt[r, c] = cqt[r, c]
        return new_cqt

    def notes_from_fragment_CQT(self, subCQT, subCdB, limit):
        resultNotesG = []
        maxNote = -1
        noteName = ''
        noteNameLatin = ''
        subCdB[0:6, :] = -120
        subCQT[0:6, :] = -120
        if subCQT.max() != -120:
            meanG = np.mean(subCdB, axis=1)
            # subCQT[subCQT==-120]=-40
            meanCQT = np.mean(subCQT, axis=1)
            avgG = np.where(meanG > limit)[0]
            avgCQT = np.where(meanCQT > limit)[0]
            avg = avgCQT
            # FirstBand (El de mayor Db de los primeros consecutivos)
            consecutiveCQT = np.split(avg, np.where(np.diff(avg) != 1)[0]+1)
            if len(consecutiveCQT) > 0 and len(avg) > 0:
                consecutiveCQT = [c for c in consecutiveCQT if len(c) > 1]
                if len(consecutiveCQT) == 0:
                    consecutiveCQT = np.split(
                        avg, np.where(np.diff(avg) != 1)[0]+1)
                rangeConsecutive = consecutiveCQT[0]
                if len(rangeConsecutive) > 12:
                    rangeConsecutive = rangeConsecutive[0:12]
                indexMaxCQT = np.argmax(meanG[np.array(rangeConsecutive)])
                maxNote = consecutiveCQT[0][indexMaxCQT]
        if maxNote >= 0:
            noteName = UtilsFunctions.compute_Note(maxNote, latin=False)
            noteNameLatin = UtilsFunctions.compute_Note(maxNote, latin=True)
        result = {'Note': maxNote, 'NoteName': noteName,
                  'NoteLatin': noteNameLatin}
        return result

    def main_note_in_fragment(self, pathFile='', speech=False, latin=False, y=None, sr=44100, limit=-38):
        if y is None:
            if pathFile == '':
                y = self.x
                sr = self.sr
            else:
                y, sr = librosa.load(pathFile, sr=self.sr, mono=True)
        CdB = self.calc_cqt(y, sr)
        cqt = self.cqt_thresholded(CdB)
        onsets = self.calc_onset(cqt)
        onsetsR = self.calc_onset_without_silents(cqt, onsets[1])
        # Longitud del cqt (Importante cuando es un fragmento muy pequeño)
        longfragment = cqt.shape[1]
        toplength = 1
        if longfragment > 10:
            toplength = 5
        onsetsF = np.array([o for o in onsetsR if (o[1]-o[0]) > toplength])
        onsetsInit = onsetsEnd = resultNotes = notesInt = notesLatin = []
        if onsetsF.size > 0:
            onsetsInit = onsetsF[:, 0]
            onsetsEnd = onsetsF[:, 1]
            for i, e in zip(onsetsInit, onsetsEnd):
                subCQT = cqt[:, i:e+1]
                subCdB = CdB[:, i:e+1]
                result = self.notes_from_fragment_CQT(subCQT, subCdB, limit)
                resultNotes.append(result['Note'])
        timeI = librosa.frames_to_time(
            onsetsInit, sr=self.sr, hop_length=self.hop_length)
        timeF = librosa.frames_to_time(
            onsetsEnd, sr=self.sr, hop_length=self.hop_length)
        notesInt = self.compute_Notes(
            resultNotes) if len(resultNotes) > 0 else []
        notesLatin = self.compute_Notes_Latin(
            resultNotes) if len(resultNotes) > 0 else []
        data = {'Number': resultNotes, 'Note': notesInt, 'NoteLatin': notesLatin,
                'OnsetInit': onsetsInit, 'OnsetEnd': onsetsEnd, 'TimeI': timeI, 'TimeF': timeF}
        df = pd.DataFrame(data, columns=[
                          'Number', 'Note', 'NoteLatin', 'OnsetInit', 'OnsetEnd', 'TimeI', 'TimeF'])
        df = df.drop(df[df['Number'] < 8].index)
        mainNoteNumber = -1
        mainNote = mainNoteLatin = ''
        mainTimeI = mainTimeF = 0
        if len(df.index) > 0:
            df['DifOnsets'] = df['OnsetEnd']-df['OnsetInit']
            indexmax = df['DifOnsets'].idxmax()
            mainNoteNumber = df.loc[indexmax, 'Number']
            mainNote = df.loc[indexmax, 'Note']
            mainNoteLatin = df.loc[indexmax, 'NoteLatin']
            mainTimeI = df.loc[indexmax, 'TimeI']
            mainTimeF = df.loc[indexmax, 'TimeF']
        resultNote = {'mainNoteNumber': mainNoteNumber, 'mainNote': mainNote,
                      'mainNoteLatin': mainNoteLatin, 'TimeI': mainTimeI, 'TimeF': mainTimeF}
        return resultNote

    def notes_by_semifragment(self, pathFile='', speech=False, latin=False, y=None, sr=44100, limit=-38):
        if y is None:
            if pathFile == '':
                y = self.x
                sr = self.sr
            else:
                y, sr = librosa.load(pathFile, sr=self.sr, mono=True)
        CdB = self.calc_cqt(y, sr)
        cqt = self.cqt_thresholded(CdB)
        onsets = self.calc_onset(cqt)
        onsetsR = self.calc_onset_without_silents(cqt, onsets[1])
        onsetsF = np.array([o for o in onsetsR if (o[1]-o[0]) > 5])
        onsetsInit = onsetsEnd = resultNotes = notesInt = notesLatin = []
        if onsetsF.size > 0:
            onsetsInit = onsetsF[:, 0]
            onsetsEnd = onsetsF[:, 1]
            for i, e in zip(onsetsInit, onsetsEnd):
                subCQT = cqt[:, i:e+1]
                subCdB = CdB[:, i:e+1]
                result = self.notes_from_fragment_CQT(subCQT, subCdB, limit)
                resultNotes.append(result['Note'])
        timeI = librosa.frames_to_time(
            onsetsInit, sr=self.sr, hop_length=self.hop_length)
        timeF = librosa.frames_to_time(
            onsetsEnd, sr=self.sr, hop_length=self.hop_length)
        notesInt = self.compute_Notes(
            resultNotes) if len(resultNotes) > 0 else []
        notesLatin = self.compute_Notes_Latin(
            resultNotes) if len(resultNotes) > 0 else []
        data = {'Number': resultNotes, 'Note': notesInt, 'NoteLatin': notesLatin,
                'OnsetInit': onsetsInit, 'OnsetEnd': onsetsEnd, 'TimeI': timeI, 'TimeF': timeF}
        df = pd.DataFrame(data, columns=[
                          'Number', 'Note', 'NoteLatin', 'OnsetInit', 'OnsetEnd', 'TimeI', 'TimeF'])
        df = df.drop(df[df['Number'] < 8].index)
        if len(df.index) > 1:
            df['DifOnsets'] = df['OnsetEnd']-df['OnsetInit']
            indexmax = df['DifOnsets'].idxmax()
            if indexmax == 0:
                df = df[:-1]
            elif indexmax == len(df.index)-1:
                df = df.drop(df.head(1).index)
            else:
                df = df[:-1]
                df = df.drop(df.head(1).index, inplace=True)

        dataresult = {'Number': df['Number'].values, 'Note': df['Note'].values,
                      'NoteLatin': df['NoteLatin'].values, 'TimeI': df['TimeI'].values, 'TimeF': df['TimeF'].values}
        return dataresult

    # Para calcular las notas en un fragmento pequeño
    def notes_by_onsets_fragment(self, pathFile='', speech=False, latin=False, y=None, sr=44100, limit=-38):
        if y is None:
            if pathFile == '':
                y = self.x
                sr = self.sr
            else:
                y, sr = librosa.load(pathFile, sr=self.sr, mono=True)
        CdB = self.calc_cqt(y, sr)
        cqt = self.cqt_thresholded(CdB)
        onsets = self.calc_onset(cqt)
        onsetsR = self.calc_onset_without_silents(cqt, onsets[1])  # onsets[1]
        onsetsF = np.array([o for o in onsetsR if (o[1]-o[0]) > 5])
        #onsetsF=np.array([[onsetsR[i],onsetsR[i+1]-1] for i, o in enumerate(onsetsR[0:-1]) if (onsetsR[i+1]-onsetsR[i])>5])
        if onsetsF.size > 0:
            onsetsInit = onsetsF[:, 0]
            onsetsEnd = onsetsF[:, 1]
        else:
            onsetsInit = []
            onsetsEnd = []
        resultNotes = []
        for i, e in zip(onsetsInit, onsetsEnd):
            subCQT = cqt[:, i:e+1]
            subCdB = CdB[:, i:e+1]
            result = self.notes_from_fragment_CQT(subCQT, subCdB, limit)
            resultNotes.append(result['Note'])
        timeI = librosa.frames_to_time(
            onsetsInit, sr=self.sr, hop_length=self.hop_length)
        timeF = librosa.frames_to_time(
            onsetsEnd, sr=self.sr, hop_length=self.hop_length)
        notesInt = self.compute_Notes(
            resultNotes) if len(resultNotes) > 0 else []
        notesLatin = self.compute_Notes_Latin(
            resultNotes) if len(resultNotes) > 0 else []
        data = {'Number': resultNotes, 'Note': notesInt, 'NoteLatin': notesLatin,
                'OnsetInit': onsetsInit, 'OnsetEnd': onsetsEnd, 'TimeI': timeI, 'TimeF': timeF}
        df = pd.DataFrame(data, columns=[
                          'Number', 'Note', 'NoteLatin', 'OnsetInit', 'OnsetEnd', 'TimeI', 'TimeF'])
        df = df.drop(df[df['Number'] < 8].index)
        minNote, maxNote, maxMin, maxMinLatin, modeNote, modeNoteN, modeNoteL = self.computeStatisticNotes(
            df['Number'].values)
        dataresult = {'Number': df['Number'].values, 'Note': df['Note'].values, 'NoteLatin': df['NoteLatin'].values, 'TimeI': df['TimeI'].values, 'TimeF': df['TimeF'].values, 'MinMaxNumber': [
            minNote, maxNote], 'MinMaxNote': maxMin, 'MinMaxLatin': maxMinLatin, 'Mode': modeNote, 'ModeNote': modeNoteN, 'ModeLatin': modeNoteL}
        return dataresult, df

    def notes_by_onsets_text(self, pathFile, speech=False, latin=False, limit=-38):
        if pathFile == '':
            y = self.x
            sr = self.sr
        else:
            y, sr = librosa.load(pathFile, sr=self.sr, mono=True)
        CdB = self.calc_cqt(y, sr)
        cqt = self.cqt_thresholded(CdB)
        onsets = self.calc_onset(cqt)
        onsetsR = onsets[1]
        onsetsF = [o for i, o in enumerate(
            onsetsR[0:-1]) if (onsetsR[i+1]-onsetsR[i]) >= 8]
        onsetsF.append(onsetsR[-1])
        onsetsInit = np.array(onsetsF[0:-1])
        onsetsEnd = np.array(onsetsF[1:])-1
        resultNotes = []
        for i, e in zip(onsetsInit, onsetsEnd):
            subCQT = cqt[:, i:e+1]
            subCdB = CdB[:, i:e+1]
            result = self.notes_from_fragment_CQT(subCQT, subCdB, limit)
            resultNotes.append(result['Note'])
        timeI = list(range(0, len(onsetsInit)))
        timeF = list(range(1, len(onsetsInit)+1))
        notesInt = self.compute_Notes(resultNotes)
        notesLatin = self.compute_Notes_Latin(resultNotes)
        data = {'Number': resultNotes, 'Note': notesInt, 'NoteLatin': notesLatin, 'OnsetInit': onsetsInit,
                'OnsetEnd': onsetsEnd, 'DiffOnset': onsetsEnd-onsetsInit, 'TimeI': timeI, 'TimeF': timeF}
        df = pd.DataFrame(data, columns=[
                          'Number', 'Note', 'NoteLatin', 'OnsetInit', 'OnsetEnd', 'DiffOnset', 'TimeI', 'TimeF'])
        df = self.deleteNotewithNoise(df)
        minNote, maxNote, maxMin, maxMinLatin, modeNote, modeNoteN, modeNoteL = self.computeStatisticNotes(
            df['Number'].values)
        dataresult = {'Number': df['Number'].values, 'Note': df['Note'].values, 'NoteLatin': df['NoteLatin'].values, 'TimeI': df['TimeI'].values, 'TimeF': df['TimeF'].values, 'MinMaxNumber': [
            minNote, maxNote], 'MinMaxNote': maxMin, 'MinMaxLatin': maxMinLatin, 'Mode': modeNote, 'ModeNote': modeNoteN, 'ModeLatin': modeNoteL}
        return dataresult, df

    def computeStatisticNotes(self, resultNotes, minfreq=1):
        modeNote = -1
        minNote = -1
        maxNote = -1
        if len(resultNotes) > 0:
            (idNote, idx, counts) = np.unique(
                resultNotes, return_index=True, return_counts=True)
            index = idx[np.argmax(counts)]
            modeNote = resultNotes[index]
            # Eliminar bajas frecuencias de apareción (Pueden ser ruidos)
            rNote = idNote
            if np.max(counts) > minfreq:
                rNote = idNote[counts > minfreq]
            minNote = np.min(rNote)
            maxNote = np.max(rNote)
        modeNoteN = self.compute_Notes(
            [modeNote])[0] if modeNote > -1 else ['']
        modeNoteL = self.compute_Notes_Latin(
            [modeNote])[0] if modeNote > -1 else ['']
        maxMin = self.compute_Notes(
            [minNote, maxNote]) if minNote > -1 else ['', '']
        maxMinLatin = self.compute_Notes_Latin(
            [minNote, maxNote]) if minNote > -1 else ['', '']
        return minNote, maxNote, maxMin, maxMinLatin, modeNote, modeNoteN, modeNoteL

    def deleteNotewithNoise(self, df):
        # Quitar notas vacías
        df = df.drop(df[df['Number'] < 0].index)
        resultNotes = df['Number']
        # Nota fuera de lugar si hay una diferencia superior a -12 con las del lado
        diff = np.diff(resultNotes)
        diffNext = np.concatenate([diff, [0]])
        diffPrevious = np.concatenate([[0], diff])
        #cond=np.logical_and(np.abs(diffNext)<12, np.abs(diffPrevious)<12)
        indexCond = np.where((np.abs(diffNext) >= 12) &
                             (np.abs(diffPrevious) >= 12))[0]
        df = df.drop(df.index[indexCond])
        return df

    def notes_by_seconds(self, pathFile, speech=False, latin=False, seconds=1):
        y, sr = librosa.load(pathFile, sr=self.sr, mono=True)
        CdB = self.calc_cqt(y, sr)
        cqt = self.cqt_thresholded(CdB)
        silents = self.detect_silents(cqt)
        duration = y.shape[0]/sr
        time = int(seconds*sr)
        countdiv = int(duration/seconds)+((int(duration/seconds)
                                           * seconds) < duration)  # (duration % seconds > 0)
        fragmentsCQT = [y[i*time:min((i+1)*time, len(y))]
                        for i in range(0, countdiv)]
        resultNotes = []
        resultNotesNames = []
        for f in fragmentsCQT:
            note_result = self.notes_from_fragment(
                '', fragment=True, songFragment=f)
            resultNotes.append(note_result['Note'])
            resultNotesNames.append(note_result['NoteName'])
        timeI = list(range(0, countdiv))
        timeF = list(range(1, countdiv+1))
        # Eliminar las notas vacías
        resultNotes = np.array(resultNotes)
        cond = resultNotes > 0
        timeI = np.array(timeI)[cond]
        timeF = np.array(timeF)[cond]
        resultNotes = resultNotes[cond]
        minNote = np.min(resultNotes)
        maxNote = np.max(resultNotes)
        (_, idx, counts) = np.unique(resultNotes,
                                     return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        modeNote = resultNotes[index]
        maxMin = self.compute_Notes([minNote, maxNote])
        maxMinLatin = self.compute_Notes_Latin([minNote, maxNote])
        notesInt = self.compute_Notes(resultNotes)
        notesLatin = self.compute_Notes_Latin(resultNotes)
        result = {'Number': resultNotes, 'NoteNames': notesInt, 'NoteLatin': notesLatin, 'MinMaxNumber': [
            minNote, maxNote], 'MinMaxNote': maxMin, 'MinMaxLatin': maxMinLatin, 'Mode': modeNote}
        return result

    def notes_from_file(self, speech=False):
        self.speech = speech
        CdB, cqt, onsets = self.calc_onset_CQT()
        if cqt.max() == -120:  # Vacío
            data = {'Number': [], 'Note': [], 'OnsetInit': [],
                    'OnsetEnd': [], 'DiffOnset': [], 'TimeI': [], 'TimeF': []}
            df = pd.DataFrame(data, columns=[
                              'Number', 'Note', 'OnsetInit', 'OnsetEnd', 'DiffOnset', 'TimeI', 'TimeF'])
            dataresult = {'Number': [], 'Note': [], 'NoteLatin': [], 'TimeI': [
            ], 'TimeF': [], 'MinMaxNumber': [], 'MinMaxNote': [], 'MinMaxLatin': []}
            return data, dataresult, df
        self.onsets = onsets
        cqt = self.cqt_from_limit(cqt)
        self.cqt = cqt
        self.new_cqt = self.detect_cqt_firstBand_onset_2(cqt)
        self.new_cqt = self.cqt_from_limit(self.new_cqt)
        if self.new_cqt.max() == -120:  # No se encontraron notas, puede ser un bajo o estar vacío
            self.new_cqt = self.recomputeBands(cqt)
        cqt_th = self.cqt_thresholded_(self.new_cqt, self.cqt_threshold)
        self.new_cqt = cqt_th
        onsets = self.calc_onset(cqt_th)
        newonsets, onsetsEmpty_Index = self.calc_newonset_with_spaces(
            onsets, cqt_th)
        self.newonsets = newonsets
        maxNotebyOnset, onsetInit, onsetEnd = self.compute_Note_by_onset_CdB(
            cqt_th, newonsets)
        arrayNotesnp, onsetInit, onsetEnd = self.delete_repeted_empty_Note(
            maxNotebyOnset, onsetInit, onsetEnd)
        arrayNotesnp, onsetInit, onsetEnd = self.delete_incoherents_Note(
            arrayNotesnp, onsetInit, onsetEnd)
        arrayNotesnp, onsetInit, onsetEnd = self.delete_short_notes(
            arrayNotesnp, onsetInit, onsetEnd)
        arrayNotesnp, onsetInit, onsetEnd = self.delete_noise_under_mean_speech(
            arrayNotesnp, onsetInit, onsetEnd)
        arrayNotesnp, onsetInit, onsetEnd = self.delete_join_short_notes(
            arrayNotesnp, onsetInit, onsetEnd)
        mNoteRealFinal = self.compute_Notes(arrayNotesnp)
        numberNote = arrayNotesnp
        notes = mNoteRealFinal
        notesLatin = self.compute_Notes_Latin(arrayNotesnp)
        time = self.compute_time(self.newonsets)
        timeI = librosa.frames_to_time(
            onsetInit, sr=self.sr, hop_length=self.hop_length)
        timeF = librosa.frames_to_time(
            onsetEnd, sr=self.sr, hop_length=self.hop_length)
        ####Max and Min
        minNote = np.min(numberNote)
        maxNote = np.max(numberNote)
        maxMin = self.compute_Notes([minNote, maxNote])
        maxMinLatin = self.compute_Notes_Latin([minNote, maxNote])
        (idNote, idx, counts) = np.unique(
            numberNote, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        modeNote = numberNote[index]
        modeNoteN = self.compute_Notes([modeNote])[0]
        modeNoteL = self.compute_Notes_Latin([modeNote])[0]
        data = {'Number': numberNote, 'Note': notes, 'OnsetInit': onsetInit,
                'OnsetEnd': onsetEnd, 'DiffOnset': onsetEnd-onsetInit, 'TimeI': timeI, 'TimeF': timeF}
        df = pd.DataFrame(data, columns=[
                          'Number', 'Note', 'OnsetInit', 'OnsetEnd', 'DiffOnset', 'TimeI', 'TimeF'])
        dataresult = {'Number': numberNote, 'Note': notes, 'NoteLatin': notesLatin, 'TimeI': timeI, 'TimeF': timeF, 'MinMaxNumber': [
            minNote, maxNote], 'MinMaxNote': maxMin, 'MinMaxLatin': maxMinLatin, 'Mode': modeNote, 'ModeNote': modeNoteN, 'ModeLatin': modeNoteL}
        return data, dataresult, df

    def result_Dict(self, speech=False, plot=False, save=False, img='', optionModeText=False):
        if optionModeText and speech:
            dataresult, df = self.notes_by_onsets_fragment(
                pathFile='', speech=speech)
        else:
            data, dataresult, df = self.notes_from_file(speech)

        numberNote = np.array(range(12, 72))
        noteMusic = [self.convert_N_to_Note(i) for i in numberNote]
        datamusic = {'NoteName': self.noteNames,
                     'NoteNameLatin': self.noteNames_Latin}
        result = {'BasicMusicNote': datamusic, 'ResultNotes': dataresult}

        # if plot==True or save==True:
        #     time=self.compute_time(self.newonsets)
        #     plt.figure()
        #     plt.hlines(df['Number'], df['TimeI'], df['TimeF'], color='red', lw=4)
        #     plt.xticks(np.arange(0, time[len(time)-1], 1))
        #     plt.yticks([12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71],
        #                ['C2','C#2','D2','D#2','E2','F2','F#2','G2','G#2','A2','A#2','B2','C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3','A#3','B3','C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4','C5','C#5','D5','D#5','E5','F5','F#5','G5','G#5','A5','A#5','B5','C6','C#6','D6','D#6','E6','F6','F#6','G6','G#6','A6','A#6','B6'])
        #     plt.ylim(ymin=df['Number'].min()-1, ymax=df['Number'].max()+1)
        # if plot==True:
        #     plt.show()
        # if save==True:
        #     plt.savefig(img)
        #     plt.close()
        return result  # ,self.cqt, self.new_cqt
