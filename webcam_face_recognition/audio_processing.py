import os
import numpy as np
import librosa
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import speech_recognition as sr
from pydub import AudioSegment


class Person:
    def __init__(self, name):
        self.name = name
        self.audio = 0
        self.audio_file = 0
        self.audio_path = 0
        audios = os.listdir("audios")
        for audio in audios:
                audio_name = audio.split(".")
                audio_name = audio_name[0]
                if audio_name == name:
                     self.audio_file = name +".wav"
                     self.audio_path = os.path.join('audios', self.audio_file)
    def extract_features(self):
        audio, sample_rate = librosa.load(self.audio_path, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    
    def save_audio(self):
        audio = self.audio
        self.audio_path = os.path.join('audios', self.audio_file)
        wav_data = audio.get_wav_data()
        audio_segment = AudioSegment(wav_data)#, frame_rate=audio.sample_rate)
        audio_segment.export(self.audio_path, format="wav")
        print(f"{self.name}'s audio saved: {self.audio_path}")
    
    def recognize_audio(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
                print("Say something:")
                self.audio = r.listen(source, phrase_time_limit=7)
                self.audio_file = self.name + ".wav"
                    
        try:
            text = r.recognize_google(self.audio, language="en-EN")  # Recognize in English
            print(f"Recognized text for {self.name}: {text}")
        except sr.UnknownValueError:
            print("Couldn't understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Speech Recognition service; {e}")
    def check_other_persons(self,persons):
            audios = os.listdir("audios")
            for audio in audios:
                audio_name = audio.split(".")
                audio_name = audio_name[0]
                person = Person(audio_name)
                persons.append(person)
            return persons

class VoiceComparator:
    def __init__(self, persons, person):
        self.persons = persons
        self.model = KNeighborsClassifier(n_neighbors=len(persons))
        self.person = person
        self.train_model()

    def train_model(self):
        X = [person.extract_features() for person in self.persons]
        y = [person.name for person in self.persons]
        self.model.fit(X, y)

    def compare_voice(self, bool):
        if bool:
            person_to_compare = self.person
            unknown_features = person_to_compare.extract_features().reshape(1, -1)
            distances = []
            for person in self.persons:
                person_features = person.extract_features().reshape(1, -1)
                distance = np.linalg.norm(person_features - unknown_features)
                distances.append(distance)

            closest_person_index = np.argmin(distances)
            closest_person = self.persons[closest_person_index]
            print(f"The voice is most similar to {closest_person.name}")
        #     if person_to_compare.name == closest_person.name:
        #     print(f"The voice is {closest_person.name}'s voice.")
        #    else:
        #        print(f"The voice is probably {closest_person.name}'s voice, so face and voice didn't match")
        else:
            print("The voice has been added.")



# persons_diger = [
#     Person("fr1", "fr1.wav"),
#     Person("fr2", "fr2.wav"),
#     Person("busra", "busra.wav"),
#     Person("kenan", "kenan.wav"),
#     Person("orhan", "orhan.wav")
# ]

# comparator = VoiceComparator(persons)
# comparator.compare_voice("busra.wav")
# persons = []
# audios = os.listdir("audios")
# for audio in audios:
#     audio_name = audio.split(".")
#     audio_name = audio_name[0]
#     person = Person(audio_name,audio)
#     persons.append(person)
# print("*")


