# -*- coding: utf-8 -*-

'''
Created on 2018. 7. 2.

@author: khs
'''

import os, sys
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

def main():
    os.environ['DEMENTIA_CONFIG_FILE'] = sys.argv[1] if len(sys.argv) > 1 else 'configure.json'
    import web
    
    web.webStart()

def test():
    pass

"""
def get_result():
    os.environ['DEMENTIA_CONFIG_FILE'] = sys.argv[1] if len(sys.argv) > 1 else 'configure.json'
    import diagnosis

    for qnum in range(1, 2):
        for disease in ["MCI", "AD", "SCI"]:
            base_path = "D://DATA//dementia_ai_clinicalTrial//record_renamed//" + str(qnum) + "//" + disease + "//"
            files = os.listdir(base_path)
            #print(len(files))


            for fileidx in range(0, len(files)):
                #print(files)
                filelist = []
                username = os.path.splitext(files[fileidx])[0]  # 확장자 제거 안함
                print(base_path)
                print(username)

                username = username.split("_")[0]
                print(username)

                user_filelist = get_file_list(username, disease)

                result = diagnosis.execute(user_filelist)

                print(result)
"""

"""
def test_makemel():
    os.environ['DEMENTIA_CONFIG_FILE'] = sys.argv[1] if len(sys.argv) > 1 else 'configure.json'
    import diagnosis

    for qnum in range(1, 2):
        for disease in ["MCI", "AD", "SCI"]:
            base_path = "D://DATA//dementia_ai_clinicalTrial//record_renamed//" + str(qnum) + "//" + disease + "//"
            files = os.listdir(base_path)
            #print(len(files))


            for fileidx in range(0, len(files)):
                #print(files)
                filelist = []
                username = os.path.splitext(files[fileidx])[0]  # 확장자 제거 안함
                print(base_path)
                print(username)

                username = username.split("_")[0]
                print(username)

                ##if username.endswith(".wav"):
                #if (username[-1] == 'R'):
                #    username = username[:-4]
                #elif (username[-2] == "."):
                #    username = username[:-2]
                #else:
                #    print("can't find user name---", username)
                #
                #print(username)


                files1 = os.listdir(base_path)
                user_file_name = list(filter(lambda x: username+"_" in x, files1))
                base_path3 = "D://DATA//dementia_ai_clinicalTrial//record_renamed//" + str(qnum) + "//" + disease + "//"
                #filelist.append("D" + base_path3 + user_file_name[0][:-4])  # 앞의 한 글자가 날아가버려서 앞에 더미 문자 하나 추가해줌.(원본 api를 건들지 않고 결과를 뽑으려고 깔끔하진 않지만 여기에서 추가함)

                diagnosis.execute("D" + base_path3 + user_file_name[0], 1)
                #print("D" + base_path3 + user_file_name[0],"done")

                for qnum2 in range(2, 12):
                    base_path2 = "D://DATA//dementia_ai_clinicalTrial//record_renamed//" + str(qnum2) + "//" + disease + "//"
                    files2 = os.listdir(base_path2)
                    user_file_name = list(filter(lambda x: username in x, files2))
                    base_path2 = "D://DATA//dementia_ai_clinicalTrial//record_renamed//" + str(qnum2) + "//" + disease + "//"
                    #filelist.append("D" + base_path2 + user_file_name[0][:-4])

                    diagnosis.execute("D" + base_path2 + user_file_name[0], qnum2)
                    print("D" + base_path2 + user_file_name[0], "done")

"""


def test_getResults():
    TPcount = 0
    TNcount = 0
    FPcount = 0
    FNcount = 0
    classifi_result =""

    os.environ['DEMENTIA_CONFIG_FILE'] = sys.argv[1] if len(sys.argv) > 1 else 'configure.json'

    for qnum in range(1, 2):
        for disease in ["MCI", "AD", "SCI"]:
            base_path = "D://DATA//dementia_ai_clinicalTrial//record_renamed//" + str(qnum) + "//" + disease + "//"
            files = os.listdir(base_path)
            #print(files)
            # print(len(files))


            for fileidx in range(0, len(files)):
                # print(files)
                filelist = []

                username = os.path.splitext(files[fileidx])
                #print(username)
                if username[1]=='.png':
                    username = os.path.splitext(files[fileidx])[0]
                    #print(base_path)
                    #print(username)

                    username = username.split("_")[0]
                    #print(username)

                    user_filelist = get_file_list(username, disease)

                    result = diagnosis.execute(user_filelist)

                    print(result)

                    if disease == "SCI":
                        actual_class = "normal"
                    else:  # AD or MCI
                        actual_class = "abnormal"


                    if result['dementiaClass'] == "NOR":
                        classifi_result = "normal"
                    else:#AD or MCI
                        classifi_result = "abnormal"


                    print(username, ", actual: ", actual_class, ", predicted: ", classifi_result)

                    if actual_class == "normal":
                        if classifi_result == "normal":
                            print("TN")
                            TNcount=TNcount+1
                        else:
                            print("FP")
                            FPcount=FPcount+1
                    else:#실제 AD or MCI
                        if classifi_result == "normal":
                            print("FN")
                            FNcount = FNcount + 1
                        else:
                            print("TP")
                            TPcount = TPcount + 1

    print(TPcount, TNcount, FPcount, FNcount)
    # 정확도
    accuracy = (TPcount + TNcount) / (TPcount + FNcount + FPcount + TNcount)
    # 민감도
    sensitivity = (TPcount) / (TPcount + FNcount)
    # 특이도
    specificity = (TNcount) / (FPcount + TNcount)

    print("정확도", accuracy * 100, "%")
    print("민감도", sensitivity * 100, "%")
    print("특이도", specificity * 100, "%")

    # confusion matrix
    # 클래스의 레이블
    labels = [1, 0]

    # Confusion matrix 생성
    y_true = [1] * TPcount + [0] * TNcount + [1] * FNcount + [0] * FPcount
    y_pred = [1] * TPcount + [0] * TNcount + [0] * FNcount + [1] * FPcount

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Abnormal', 'Normal'],
                yticklabels=['Abnormal', 'Normal'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()





def get_file_list(userid, golden_standard):
    filelist=[]

    for qnum in range(1,12):
        path = f"sD://DATA//dementia_ai_clinicalTrial//record_renamed//{qnum}//{golden_standard}//"
        filelist.append(path+userid+"_"+str(qnum))

    return filelist



if __name__ == '__main__':
    #main()
#    test()
    import local_test_diagnosis

    test_getResults()
    #get_result()
    #user_filelist = get_file_list(userid, golden_standard)

    #result = diagnosis.execute(user_filelist)

    #print(result)

    """
    result = diagnosis.execute(["sD://DATA//dementia_ai_clinicalTrial//record_renamed//1//AD//r10a01_1",
                               "sD://DATA//dementia_ai_clinicalTrial//record_renamed//2//AD//r10a01_2",
                               "sD://DATA//dementia_ai_clinicalTrial//record_renamed//3//AD//r10a01_3",
                               "sD://DATA//dementia_ai_clinicalTrial//record_renamed//4//AD//r10a01_4",
                               "sD://DATA//dementia_ai_clinicalTrial//record_renamed//5//AD//r10a01_5",
                               "sD://DATA//dementia_ai_clinicalTrial//record_renamed//6//AD//r10a01_6",
                               "sD://DATA//dementia_ai_clinicalTrial//record_renamed//7//AD//r10a01_7",
                               "sD://DATA//dementia_ai_clinicalTrial//record_renamed//8//AD//r10a01_8",
                               "sD://DATA//dementia_ai_clinicalTrial//record_renamed//9//AD//r10a01_9",
                               "sD://DATA//dementia_ai_clinicalTrial//record_renamed//10//AD//r10a01_10",
                               "sD://DATA//dementia_ai_clinicalTrial//record_renamed//11//AD//r10a01_11"
                     ])

    """


    print("finished.")
    #test_makemel()  # make mel-spectrogram
"""
    directoryA = "D://KOTCA//test_dataset_audio//"
    directoryB = "D://KOTCA//test_dataset_image//"

    copy_directory(directoryA, directoryB)
    remove_files_by_extension(directoryA, ".png")
    remove_files_by_extension(directoryB, ".wav")

    test_getResults()
"""