import numpy as np
from IPython.display import Audio 
from PIL import Image
import os
from pydub import AudioSegment

def voice_control(filename, base_path):
    song = AudioSegment.from_file(os.path.join(base_path,filename))
    # increase volume by 10 dB
    song_10_db_inc = song + 10    
    # reduce volume by 10 dB
    song_10_db_reduce = song - 10
    # but let's make him *very* quiet
    song = song - 36
    # save the output
    song_10_db_inc.export(f"{os.path.join(base_path,filename)}_10db_inc", "wav")
    song_10_db_reduce.export(f"{os.path.join(base_path,filename)}_10db_reduce", "wav")




if __name__ == '__main__':
    # 음성 변환을 위한 for문
    for qnum in range(1, 12):
        for disease in ["SCI","MCI", "AD"]:
            base_path = "./data/Spick_test_folder/record_renamed/" + str(qnum) + "//" + disease + "//"
            recordFileRoot=base_path

            files = os.listdir(base_path)
            for file in files:
                if len(file)<10:
                    print(file)
                    voice_control(file,base_path)
