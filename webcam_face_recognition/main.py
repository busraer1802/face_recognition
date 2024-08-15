from recognition import FaceRecognition
from audio_processing import Person, VoiceComparator
import os 

persons = []
if __name__ == '__main__':
        fr = FaceRecognition()
        fr.run_recognition()
        person = Person(fr.face_names[0])
        audios = os.listdir("audios")
        is_audio_exist = False
        for audio in audios:
                audio_name = audio.split(".")
                audio_name = audio_name[0]
                if audio_name == fr.face_names[0]:
                        is_audio_exist = True
        person.recognize_audio()
        persons = person.check_other_persons(persons)
        vc = VoiceComparator(persons,person)
        if not is_audio_exist:
                person.save_audio()
        vc.compare_voice(is_audio_exist)
        
        
