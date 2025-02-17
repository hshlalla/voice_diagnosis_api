'''
Created on 2022. 2. 25.

@author: khs
'''

#import tempfile, os, soundfile, librosa, math, time, multiprocessing, string, random, shutil
import os, soundfile, tempfile, warnings
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('agg')
warnings.filterwarnings(action='ignore', category=UserWarning, module='librosa')
# voice_cut2 기준(최종 결정 완료)에 따른 문항별 음원 지속 시간
end_time = [9, 10, 12, 18, 60, 60, 60, 60, 60, 60, 60]
# ex) 1번 문항은 9초로 통일, 3번 문항은 12초로 통일..
sr = 48000 # sampling rate

def imageprocessing(wav_file, save_file, sr):
    Mel_Spectrogram(wav_file, save_file, sr)
    crop_mel(save_file)
    
def execute_mel(file_path, step, recordFileRoot):
    try:
        step = int(step) - 1
        
        mp4_file = os.path.join(recordFileRoot, file_path)

        y, _ = librosa.load(mp4_file, sr=48000)
        
        duration = len(y)/48000

        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=True).name
        if (end_time[step] > duration):
            final_duration=end_time[step]*sr
            y2 = np.concatenate((y, np.zeros(abs(final_duration-len(y)))), axis=0)


        else:
            cutDuration = sr * end_time[step]
            y2 = y[:cutDuration]


        soundfile.write(temp_file, y2, sr, format='WAV')

        imageprocessing(temp_file, "{}.png".format(mp4_file), sr)
    except Exception as ex:
        result = {"resultCode": "9999", "msg": '{}: {}'.format(type(ex).__name__, str(ex))}
        
        return result

# 잘린 파일을 mel-spectrogram으로 변환하기
def Mel_Spectrogram(wav_path, save_file, sr):
    y, sr = librosa.load(wav_path, sr=sr)

    sec = 60

    index = sr * sec

    y_segment = y[0:index]

    S = librosa.feature.melspectrogram(y=y_segment, sr=sr, win_length=3000,
                                       n_fft=3000)  # win_length, n_fft = 3000으로 통일.(<-가천대에서 결정)
    sum_s=np.sum(S)
    plt.gcf().set_size_inches(3, 3.24)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    plt.savefig(save_file, dpi=100)
    plt.close()

def crop_mel(img_file):
    img = plt.imread(img_file)
    img_cropped = img[24:324, :, :]

    plt.gcf().set_size_inches(3, 3)
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    plt.imshow(img_cropped)

    plt.savefig(img_file, dpi=100)
    plt.close()

def get_file_list(userid, golden_standard):
    print(userid)
    filelist=[]

    # 문항 개수로 테스트 해보려면 이곳수정
    for qnum in ([1,2,3,4,5,6,7,8,9,10,11]):
        path = f"./data/Spick_test_folder/record_renamed/{qnum}/{golden_standard}/"
        filelist.append(f"{path}{userid}_{qnum}")

    return filelist    

if __name__ == '__main__':
    for qnum in range(10, 12):
        for disease in ["SCI","MCI", "AD"]:
            base_path = "./data/Spick_test_folder/record_renamed/" + str(qnum) + "//" + disease + "//"
            recordFileRoot=base_path
            # 아이디 가져오기
            files = os.listdir(base_path)
            for file in files:
                if len(file)==18:
                    execute_mel(file,qnum,base_path)
                elif len(file)==21:
                    execute_mel(file,qnum,base_path)



