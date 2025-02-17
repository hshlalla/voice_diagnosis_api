#import tempfile, os, soundfile, librosa, math, time, multiprocessing, string, random, shutil
import os, json, utils
import pandas as pd
import numpy as np
from keras.models import load_model, Model
from keras.preprocessing import image
from keras import backend as K
from datetime import datetime
import modules.mel as mel
import rescale_score
import spider_web

# 어떤 버전으로 작업했는데 명시할껏
# dementia_version = mfds_dev

now=datetime.now()
# model_folder="./src/"
model_folder="/data/models/spick_used_models/"
models = {
    'q3': {
        'final_model':  {
            'MCI_AD': model_folder+"final_model/q3/MCI_AD_q3.h5",
            'NOR_AD': model_folder+"final_model/q3/SCI_AD_q3.h5",
            'NOR_AB': model_folder+"final_model/q3/normal_abnormal_q3.h5",
            'NOR_MCI': model_folder+"final_model/q3/SCI_MCI_q3.h5",
        },
        'seq': [8, 3, 5]
    },
    'q10': {
        'final_model':  {
            'MCI_AD': model_folder+"final_model/q10/MCI_AD_ver1_q10.h5",
            'NOR_AD': model_folder+"final_model/q10/normal_AD_ver1_q10.h5",
            'NOR_AB': model_folder+"final_model/q10/normal_abnormal_ver1_q10.h5",
            'NOR_MCI': model_folder+"final_model/q10/normal_MCI_ver1_q10.h5",
        },
        'seq': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    },
    'q11': {
        'final_model':  {
            'MCI_AD': model_folder+"final_model/q11/MCI_AD.h5",
            'NOR_AD': model_folder+"final_model/q11/normal_AD.h5",
            'NOR_AB': model_folder+"final_model/q11/normal_abnormal.h5",
            'NOR_MCI': model_folder+"final_model/q11/normal_MCI.h5",
        },
        'seq': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    },
    # 'MCI_AD': [load_model(os.path.join(model_folder,'MA', item)) for item in sorted(os.listdir(model_folder+'MA'))],
    # 'NOR_AD':  [load_model(os.path.join(model_folder,'NA', item)) for item in sorted(os.listdir(model_folder+'NA'))],
    # 'NOR_AB':  [load_model(os.path.join(model_folder,'NAB', item)) for item in sorted(os.listdir(model_folder+'NAB'))],
    # 'NOR_MCI':  [load_model(os.path.join(model_folder,'NM', item)) for item in sorted(os.listdir(model_folder+'NM'))],

    # 폴더가 더러워...
    'MCI_AD': [os.path.join(model_folder,'MA', item) for item in sorted(os.listdir(model_folder+'MA')) if item.startswith("save_model") and item.endswith(".h5") and not item.endswith("backbone.h5")],
    'NOR_AD': [os.path.join(model_folder, 'NA', item) for item in sorted(os.listdir(model_folder + 'NA')) if item.startswith("save_model") and item.endswith(".h5") and not item.endswith("backbone.h5")],
    'NOR_AB': [os.path.join(model_folder, 'NAB', item) for item in sorted(os.listdir(model_folder + 'NAB')) if item.startswith("save_model") and item.endswith(".h5") and not item.endswith("backbone.h5")],
    'NOR_MCI': [os.path.join(model_folder, 'NM', item) for item in sorted(os.listdir(model_folder + 'NM')) if item.startswith("save_model") and item.endswith(".h5") and not item.endswith("backbone.h5")],

}
#logger.debug('============ {} ============='.format(time.time() - stime))
#os.makedirs('processing', exist_ok=True)

# voice_cut2 기준(최종 결정 완료)에 따른 문항별 음원 지속 시간
end_time = [9, 10, 12, 18, 60, 60, 60, 60, 60, 60, 60]
# ex) 1번 문항은 9초로 통일, 3번 문항은 12초로 통일..
sr = 48000 # sampling rate

def execute(fileList: list, birthday, gender, isSpider=False):
    print("시작")
    try:
#        unique_key = '{}_{}'.format(time.time(), ''.join(random.sample(string.ascii_lowercase, k=6)))
    
        result_proba_dict = dict()
        result_proba_list_dict = dict()
        result_proba_list_spider=dict()
        
        for model_class in models["q11"]['final_model'].keys(): #4가지 유형 불러옴
            img_data_array = {}
            img_columns_array = {}
            model_class1, model_class2 = model_class.split('_')
    #        img_array_all = []

    # 여기서 수정해야함.모델이 11개 로드되는 포문이므로 10번문제가 빠졌을때 11번 문제가 10번 모델에 적용됨.
            for step_no, seq in enumerate(models['q{}'.format(len(fileList))]['seq']):
                #원본
                loaded_model = load_model(models[model_class][seq - 1])

                #img_name = os.path.join(configure.recordFileRoot, "{}.png".format(fileList[step_no][1:]))
                img_name = os.path.join(recordFileRoot, "{}".format(fileList[step_no]))
                # img_name = fileList[step_no]  
                feature_set, features_column, processed_img = feature_extract(loaded_model, seq, img_name)
                #spider_web
                if isSpider and (len(fileList) == 10 or len(fileList) == 11):
                    spider_web.spider_web_calculate(
                        result_proba_list_spider,
                        loaded_model,
                        processed_img,
                        model_class1,
                        model_class2,
                        step_no,
                        model_class)                   

                K.clear_session()
                #(디버그 툴로 꼭 모델과, 문제가 정확하게 매칭되고 있는지 확인할것 문제 개수에 따른 실수 확율 매우 높음)
                img_data_array["data" + str(seq-1)] = feature_set
                img_columns_array["col" + str(seq-1)] = features_column


            final_df = final_classification(img_data_array, img_columns_array, ques_no= len(fileList))

#            final_df.to_csv('{}.csv'.format(model_class))
#            pred = models['final_model'][model_class].predict(final_df)
            
            
            model_name = models["q{}".format(len(fileList))]['final_model'][model_class]
            dementia_proba2 = load_model(model_name).predict(final_df)[0,0]
            dementia_proba1 = 1 - dementia_proba2
            result_proba_dict[model_class] = int(dementia_proba1*100000)/1000
                
            
            model_class1_proba = result_proba_list_dict.pop(model_class1, [])
            model_class1_proba.append(dementia_proba1)
            result_proba_list_dict[model_class1] = model_class1_proba
            
            model_class2_proba = result_proba_list_dict.pop(model_class2, [])
            model_class2_proba.append(dementia_proba2)
            result_proba_list_dict[model_class2] = model_class2_proba

        ab_proba = result_proba_list_dict.pop('AB')[0]
        result_proba_list_dict['MCI'].append(ab_proba) #abnormal 부분 점수 원본값으로 적용
        result_proba_list_dict['AD'].append(ab_proba/2) #abnormal 부분 점수 원본값으로 적용

        sum_dict = {k: sum(v) for k, v in result_proba_list_dict.items()}
        # 성별 점수, 나이점수 보정
        print(now.year)
        age=now.year-int(birthday)
        print(age)
        gender=gender

        #23명 데이터 확인후 가중치 조정
        # sum_dict["NOR"]=sum_dict["NOR"]*0.7
        # individualScore에서 나오는 sum_dict 형태 : {"AD": 0.225, "NOR": 0.871, "MCI": 0.236}
        individualScore={"classScore":result_proba_dict, "sum_dict":{k:int(v*1000/3)/1000 for k,v in sum_dict.items()}}
        if bool(result_proba_list_spider):
            verbalAbility, shortTermMemory, longTermMemory, concentration, calculationAbility, conceptualThinking, visualComprehension=spider_web.spider_web_ability(result_proba_list_spider,fileList)
            
            individualScore.update({'compositeScore': {
                "verbalAbility":verbalAbility,
                "shortTermMemory":shortTermMemory,
                "longTermMemory":longTermMemory,
                "concentration":concentration,
                "calculationAbility":calculationAbility,
                "conceptualThinking":conceptualThinking,
                "visualComprehension":visualComprehension}})
        
        #NOR vs MCI+AD
        if sum_dict["NOR"]>(sum_dict["AD"]+sum_dict["MCI"]):
            dementia_class="NOR"
            dementia_proba= sum_dict["NOR"]
        elif sum_dict["MCI"]>sum_dict["AD"]:
            dementia_class="MCI"
            dementia_proba= sum_dict["MCI"]
        else:
            dementia_class="AD"
            dementia_proba= sum_dict["AD"]
            
        #"AD"와 "MCI"를 더한값과 normal을 비교하게 되서 아래 MAX로 dementia_class를 정하는 방식은 주석처리
        #dementia_class = max(sum_dict, key=sum_dict.get)
        #식약처 normal vs abnormal 표현으로 바꾸기 위해서 dementia_proba 변경
        #dementia_proba = sum_dict[dementia_class] / len(result_proba_list_dict[dementia_class])
        
        individualScore['proba'] = int(sum_dict[dementia_class]*1000 / len(result_proba_list_dict[dementia_class])) / 10
        score = rescale_score.rescale_30_score(dementia_class,dementia_proba,sum_dict)
        score100 = rescale_score.rescale_100_score(dementia_class, dementia_proba, sum_dict)

        results = dict(
            dementiaClass=dementia_class,
            dementiaProba=int(dementia_proba*1000)/10,
            individualScore=utils.json_dumps(individualScore),
            score=score,
            score100=score100
            )

        #logger.debug('{}'.format(results))
        print(results)
        print(eval(results['individualScore']))
    
    except Exception as ex:
        #logger.exception('')
        result = {"resultCode": "9999", "msg": '{}: {}'.format(type(ex).__name__, str(ex))}
        print(result)

def feature_extract(loaded_model, step_num, img_path):
    flatten = loaded_model.layers[-2].output  # 원하는 레이어를 자동으로 가져옴
    test_model = Model(inputs=loaded_model.input, outputs=flatten)
    img = image.load_img(img_path, target_size=(300, 300))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0  # Normalize

    # Extract features
    feature_set = test_model.predict(img_tensor)
    features_column = [f"img_f{i}_{step_num}" for i in range(feature_set.shape[1])]

    return feature_set, features_column, img_tensor

# def final_classification(img_data, img_column, ques_no=11):
#     img_array_all = np.append(img_data,img_column)
#     #print(img_array_all)
#     # 문제 순서 변경시 여기서 변경
#     if ques_no == 11:
#         data_all = np.concatenate(
#             [img_array_all[7]["data7"][0], img_array_all[2]["data2"][0], img_array_all[0]["data0"][0],
#              img_array_all[6]["data6"][0], img_array_all[4]["data4"][0], img_array_all[5]["data5"][0],
#              img_array_all[10]["data10"][0], img_array_all[3]["data3"][0], img_array_all[9]["data9"][0],
#              img_array_all[1]["data1"][0], img_array_all[8]["data8"][0]])
#         columns_all = np.concatenate(
#             [img_array_all[18]["col7"], img_array_all[13]["col2"], img_array_all[11]["col0"],
#              img_array_all[17]["col6"], img_array_all[15]["col4"], img_array_all[16]["col5"],
#              img_array_all[21]["col10"], img_array_all[14]["col3"], img_array_all[20]["col9"],
#              img_array_all[12]["col1"], img_array_all[19]["col8"]])
#     elif ques_no == 10:
#         data_all = np.concatenate(
#             [img_array_all[7]["data7"][0], img_array_all[2]["data2"][0], img_array_all[0]["data0"][0],
#              img_array_all[6]["data6"][0], img_array_all[4]["data4"][0], img_array_all[5]["data5"][0],
#              img_array_all[9]["data10"][0], img_array_all[3]["data3"][0],
#              img_array_all[1]["data1"][0], img_array_all[8]["data8"][0]])
#         columns_all = np.concatenate(
#             [img_array_all[17]["col7"], img_array_all[12]["col2"], img_array_all[10]["col0"],
#              img_array_all[16]["col6"], img_array_all[14]["col4"], img_array_all[15]["col5"],
#              img_array_all[19]["col10"], img_array_all[13]["col3"],
#              img_array_all[11]["col1"], img_array_all[18]["col8"]])
#     elif ques_no == 3:
#         data_all = np.concatenate(
#             [img_array_all[0]["data7"][0], img_array_all[1]["data2"][0], img_array_all[2]["data4"][0]])
#         columns_all = np.concatenate(
#             [img_array_all[3]["col7"], img_array_all[4]["col2"], img_array_all[5]["col4"]])
#         #문제제거 하면서 dict가 땡겨지므로 index할때 주의

#     df = pd.DataFrame(data_all.astype(float)).T
#     df.columns = columns_all
#     return df

def final_classification(img_data, img_column, ques_no=11):
    # 데이터 구조화
    #img_data_dict = {f"data{step_num}": img_data[step_num] for step_num in range(len(img_data))}
    #img_column_dict = {f"col{step_num}": img_column[step_num] for step_num in range(len(img_column))}

    # 문제별 매핑
    SEQUENCE_MAPPING = {
        11: ["data7", "data2", "data0", "data6", "data4", "data5", "data10", "data3", "data9", "data1", "data8"],
        10: ["data7", "data2", "data0", "data6", "data4", "data5", "data10", "data3", "data1", "data8"],
        3:  ["data7", "data2", "data4"],
    }

    # 데이터 처리
    if ques_no in SEQUENCE_MAPPING:
        selected_data_keys = SEQUENCE_MAPPING[ques_no]
        data_all, columns_all = process_data(img_data, img_column, selected_data_keys)
    else:
        raise ValueError(f"Unsupported question number: {ques_no}")

    # 데이터프레임 생성
    df = pd.DataFrame(data=data_all,columns=columns_all)
    return df


def process_data(img_data_dict, img_column_dict, keys):
    """
    Helper function to process data and columns.
    """
    data = np.hstack([img_data_dict[key] for key in keys])
    columns = np.hstack([img_column_dict[key.replace("data", "col")] for key in keys])
    return data, columns


def get_file_list(userid, golden_standard,ques_num=11):
    print(userid)
    filelist=[]
    if ques_num==3:
        # 문항 개수로 테스트 해보려면 이곳수정
        for qnum in ([8,3,5]):
            path = f"./data/Spick_test_folder/record_renamed/{qnum}/{golden_standard}/"
            filelist.append(f"{path}{userid}_{qnum}.png")
    elif ques_num==10:
        # 문항 개수로 테스트 해보려면 이곳수정
        for qnum in ([1,2,3,4,5,6,7,8,9,11]):
            path = f"./data/Spick_test_folder/record_renamed/{qnum}/{golden_standard}/"
            filelist.append(f"{path}{userid}_{qnum}.png")
    else:
        # 문항 개수로 테스트 해보려면 이곳수정
        for qnum in ([1,2,3,4,5,6,7,8,9,10,11]):
            path = f"./data/Spick_test_folder/record_renamed/{qnum}/{golden_standard}/"
            filelist.append(f"{path}{userid}_{qnum}.png")
    return filelist

def get_file_list_inc(userid, golden_standard):
    filelist=[]

    # 문항 개수로 테스트 해보려면 이곳수정
    for qnum in ([1,2,3,4,5,6,7,8,9,10,11]):
        path = f"./data/Spick_test_folder/record_renamed/{qnum}/{golden_standard}/"
        filelist.append(f"{path}{userid}_{qnum}_10db_inc.png")

    return filelist

def get_file_list_reduce(userid, golden_standard):
    filelist=[]

    # 문항 개수로 테스트 해보려면 이곳수정
    for qnum in ([1,2,3,4,5,6,7,8,9,10,11]):
        path = f"./data/Spick_test_folder/record_renamed/{qnum}/{golden_standard}/"
        filelist.append(f"{path}{userid}_{qnum}_10db_reduce.png")

    return filelist



if __name__ == '__main__':
    #pass

    #recordFileRoot = "/home/hsh/Downloads/Spick_test_folder/MCI"
    #file_lists=set(list(map(lambda x : x.split("_")[0].split(".")[0],os.listdir(recordFileRoot))))
    birthday="1987"
    gender=["F"]
    
    # 파일 한명만 검사할 때
    # base_path = "./data/Spick_test_folder/hong/"
    # base_path = "/data/dataset/음성/voice_to_image/dataset20230927/ALL/1/AD"
    # recordFileRoot=base_path
    # files=sorted(list(filter(lambda x : x.endswith(".png"),os.listdir(base_path))))

    base_path = "/data/dataset/음성/voice_to_image/dataset20230927/ALL/"
    recordFileRoot = base_path
    files = list()
    for i in range(0, 11):
        file = base_path + f"{i+1}/AD/0b233392-7574-4b16-bb25-45f5e5dcb8dd_{i}_R.png"
        files.append(file)

    final=execute(files,birthday, gender, isSpider=True)


    # #식약처용
    # for qnum in range(1, 2):
    #     for disease in ["SCI","MCI", "AD"]:
    #         base_path = "./data/Spick_test_folder/record_renamed/" + str(qnum) + "//" + disease + "//"
    #         recordFileRoot=base_path
    #         files=list(filter(lambda x : x.endswith(".png"),os.listdir(base_path)))


    #         for fileidx in range(0, len(files)):
    #             # print(files)
    #             filelist = []

    #             username = os.path.splitext(files[fileidx])
    #             #print(username)
    #             if username[1]=='.png':
    #                 username = os.path.splitext(files[fileidx])[0]
    #                 #print(base_path)
    #                 #print(username)

    #                 username = username.split("_")[0]
    #                 #print(username)

    #                 file_lists = get_file_list(username, disease, ques_num=3)
    #                 file_lists2 = get_file_list(username, disease, ques_num=10)
    #                 file_lists3 = get_file_list(username, disease)
                    

    #                 execute(file_lists, birthday,gender)
    #                 execute(file_lists2, birthday,gender)
    #                 execute(file_lists3, birthday,gender)
                


    
    # save_result=[]

    # for file_name in file_lists:
    #     if len(file_name)>10:
    #         final=execute([f"/{file_name}_0_R",
    #                         f"/{file_name}_1_R",
    #                         f"/{file_name}_2_R",
    #                         f"/{file_name}_3_R",
    #                         f"/{file_name}_4_R",
    #                         f"/{file_name}_5_R",
    #                         f"/{file_name}_6_R",
    #                         f"/{file_name}_7_R",
    #                         f"/{file_name}_8_R",
    #                         f"/{file_name}_9_R",
    #                         f"/{file_name}_10_R"
    #                         ])
            
    #     elif len(file_name)>3:
    #         final=execute([f"/{file_name}.1",
    #                         f"/{file_name}.2",
    #                         f"/{file_name}.3",
    #                         f"/{file_name}.4",
    #                         f"/{file_name}.5",
    #                         f"/{file_name}.6",
    #                         f"/{file_name}.7",
    #                         f"/{file_name}.8",
    #                         f"/{file_name}.9",
    #                         f"/{file_name}.10",
    #                         f"/{file_name}.11"
    #                         ])
    #     else:
    #         continue
    #     save_result.append(final)

    # with open('./persons.json', 'w') as f : 
	#     json.dump(save_result, f, indent=4)
    # final=execute(["/1694758181080_o3uyZ4",
    #                "/1694758250760_vvW2lP",
    #                "/1694758342430_yPKM6v",        
    #                ])
