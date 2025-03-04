
import os, utils
import pandas as pd
import numpy as np
from keras.models import load_model, Model
from keras.preprocessing import image
from keras import backend as K
from datetime import datetime

import rescale_score
import configure
import spider_web

logger = configure.getLogger('diagnosis')
now=datetime.now()
#time = time.time()
models = {
    'q3': {
        'final_model':  {
            'MCI_AD': load_model("models/final_model/q3/MCI_AD_q3.h5"),
            'NOR_AD': load_model("models/final_model/q3/SCI_AD_q3.h5"),
            'NOR_AB': load_model("models/final_model/q3/normal_abnormal_q3.h5"),
            'NOR_MCI': load_model("models/final_model/q3/SCI_MCI_q3.h5")
        },
        'seq': [8, 3, 5]
    },
    'q10': {
        'final_model':  {
            'MCI_AD': load_model("models/final_model/q10/MCI_AD_ver1_q10.h5"),
            'NOR_AD': load_model("models/final_model/q10/normal_AD_ver1_q10.h5"),
            'NOR_AB': load_model("models/final_model/q10/normal_abnormal_ver1_q10.h5"),
            'NOR_MCI': load_model("models/final_model/q10/normal_MCI_ver1_q10.h5")
        },
        'seq': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    },
    'q11': {
        'final_model':  {
            'MCI_AD': load_model("models/final_model/q11/MCI_AD.h5"),
            'NOR_AD': load_model("models/final_model/q11/normal_AD.h5"),
            'NOR_AB': load_model("models/final_model/q11/normal_abnormal.h5"),
            'NOR_MCI': load_model("models/final_model/q11/normal_MCI.h5")
        },
        'seq': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    },
    'MCI_AD': [load_model(os.path.join('models', 'MA', item)) for item in sorted(os.listdir('models/MA'))],
    'NOR_AD':  [load_model(os.path.join('models', 'NA', item)) for item in sorted(os.listdir('models/NA'))],
    'NOR_AB':  [load_model(os.path.join('models', 'NAB', item)) for item in sorted(os.listdir('models/NAB'))],
    'NOR_MCI':  [load_model(os.path.join('models', 'NM', item)) for item in sorted(os.listdir('models/NM'))],
    # #test
    # 'MCI_AD': [(os.path.join('models', 'MA', item)) for item in sorted(os.listdir('models/MA'))],
    # 'NOR_AD': [(os.path.join('models', 'NA', item)) for item in sorted(os.listdir('models/NA'))],
    # 'NOR_AB': [(os.path.join('models', 'NAB', item)) for item in sorted(os.listdir('models/NAB'))],
    # 'NOR_MCI': [(os.path.join('models', 'NM', item)) for item in sorted(os.listdir('models/NM'))],
}
#logger.debug('============ {} ============='.format(time.time() - stime))

#os.makedirs('processing', exist_ok=True)

# voice_cut2 기준(최종 결정 완료)에 따른 문항별 음원 지속 시간
end_time = [9, 10, 12, 18, 60, 60, 60, 60, 60, 60, 60]
# ex) 1번 문항은 9초로 통일, 3번 문항은 12초로 통일..
sr = 48000 # sampling rate

def execute(fileList: list, birthday, gender, isSpider=True,):
    '''
    fileList는 리스트 형태이며, 각 요소는 녹음파일의  configure.recordFileRoot 기준 full path 문자열.
    따라서 첫번째 녹음파일의 full path는 configure.recordFileRoot + fileList[0]
    configure.recordFileRoot값은 configure.json에서 recordFileRoot키의 값을 세팅하면됨.
    예)
        configure.recordFileRoot = '/data'
        fileList[0] = '/record/2020/02/25/322341234_DFEF123'
        일경우 실제 파일 full path는 /data/record/2020/02/25/322341234_DFEF123


    tempfile 사용예
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
        K.models.save_model(model.model, fd.name, overwrite=True)
        keras_model = fd.read()

    tempfile.gettempdir()
    tempfile.mktemp(suffix, prefix, dir)

    결과 샘플
    {
        "dementiaClass": "정상",
        "dementiaProba": 0.9
    }

    {
        "dementiaClass": "MCI",
        "dementiaClass": 0.7
    }
    '''
    try:
        result_proba_dict = dict()
        result_proba_list_dict = dict()
        result_proba_list_spider=dict()
        for model_class in models["q11"]['final_model'].keys(): #4가지 유형 불러옴
            img_data_array = {}
            img_columns_array = {}
            model_class1, model_class2 = model_class.split('_')


    # 여기서 수정해야함.모델이 11개 로드되는 포문이므로 10번문제가 빠졌을때 11번 문제가 10번 모델에 적용됨.
            for step_no, seq in enumerate(models['q{}'.format(len(fileList))]['seq']):
                ##테스트 할때만 사용
                # loaded_model = load_model(models[model_class][seq-1])
                #원본
                loaded_model = models[model_class][seq - 1]

                img_name = os.path.join(configure.recordFileRoot, "{}.png".format(fileList[step_no][1:]))
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
                #디버그 툴로 꼭 확인할것 문제 제거시 실수 확율 매우 높음)
                img_data_array["data" + str(seq-1)] = feature_set
                img_columns_array["col" + str(seq-1)] = features_column

            final_df = final_classification(img_data_array, img_columns_array, ques_no= len(fileList))

#            final_df.to_csv('{}.csv'.format(model_class))
#            pred = models['final_model'][model_class].predict(final_df)

            dementia_proba2 = models["q{}".format(len(fileList))]['final_model'][model_class].predict(final_df)[0,0]
            dementia_proba1 = 1 - dementia_proba2
            result_proba_dict[model_class] = int(dementia_proba1*100000)/1000
            
            
            model_class1_proba = result_proba_list_dict.pop(model_class1, [])
            model_class1_proba.append(dementia_proba1)
            result_proba_list_dict[model_class1] = model_class1_proba
            
            model_class2_proba = result_proba_list_dict.pop(model_class2, [])
            model_class2_proba.append(dementia_proba2)
            result_proba_list_dict[model_class2] = model_class2_proba
            
        ab_proba = result_proba_list_dict.pop('AB')[0]
        # abnormal 부분 반으로 점수 줬으나 실험결과 MCI는 원점수를 주는것이 선별 정확도가 더 높아 /2를 제거
        # 이로인해 합계가 100이 안되는 상황 발생.
        result_proba_list_dict['MCI'].append(ab_proba) #abnormal 부분 점수 반으로 나눔
        result_proba_list_dict['AD'].append(ab_proba/2) #abnormal 부분 점수 반으로 나눔

        sum_dict = {k: sum(v) for k, v in result_proba_list_dict.items()}
        # 성별 점수, 나이점수 보정
        age=now.year-birthday
        gender=gender



        # #모델 4개에서 나온값이 더해지면서 key가 정렬이 이상해지는 부분 재 sort
        # result_proba_list_spider=dict(sorted(result_proba_list_spider.items(),key=lambda x:(x[0].split("_")[0],int(x[0].split("_")[1]))))
        # sum_array_for_one_ques=np.mean(np.array(list(result_proba_list_spider.values())),axis=1).reshape(3,len(fileList))
        # #sum_dict_for_one_ques = {k: sum(v)/3 for k, v in result_proba_list_spider.items()}

        # # normal만 가지고 할경우
        # if len(fileList)==10:
        #     #10문제일때 능력치 산출방법 (10번문제 제외)
        #     #언어능력 1 (10%) + 2 (10%) + 3 (10%) + 4 (10%) + 5 (10%) + 6 (10%) + 7 (10%) + 9 (15%) + 11 (15%)
        #     verbalAbility=np.sum(sum_array_for_one_ques[2,[0,1,2,3,4,5,6,8,9]]*[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.15,0.15])
        #     #단기기억력 : 1 (20%) + 2 (20%) + 3 (20%) + 11 (40%)
        #     shortTermMemory=np.sum(sum_array_for_one_ques[2,[0,1,2,9]]*[0.2,0.2,0.2,0.4])
        #     #장기기억력 : 9 (100%)
        #     longTermMemory=np.sum(sum_array_for_one_ques[2,[8]])
        #     #집중력 : 1 (5%) + 2 (5%) + 3 (5%) + 4 (5%) + 5 (5%) + 6 (5%) + 7 (5%) + 8 (50%) + 9 (7.5%)  + 11 (7.5%)
        #     concentration=np.sum(sum_array_for_one_ques[2,[0,1,2,3,4,5,6,7,8,9]]*[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.5,0.075,0.075])
        #     #계산능력 : 8 (100%)
        #     calculationAbility=np.sum(sum_array_for_one_ques[2,[7]])
        #     #개념적 사고력 : 6 (50%) + 7 (50%)
        #     conceptualThinking=np.sum(sum_array_for_one_ques[2,[5,6]]*[0.5,0.5])
        #     #시각적 이해력 : 4 (50%) + 5 (50%)
        #     visualComprehension=np.sum(sum_array_for_one_ques[2,[3,4]]*[0.5,0.5])
        # else:
        #     #11문제일때 스파이더 능력치 산출방법
        #     #언어능력 : 1 (10%) + 2 (10%) + 3 (10%) + 4 (10%) + 5 (10%) + 6 (10%) + 7 (10%) + 9 (10%) + 10 (10%) + 11 (10%)
        #     verbalAbility=np.mean(sum_array_for_one_ques[2])
        #     #단기기억력 : 1 (20%) + 2 (20%) + 3 (20%) + 11 (40%)
        #     shortTermMemory=np.sum(sum_array_for_one_ques[2,[0,1,2,10]]*[0.2,0.2,0.2,0.4])
        #     #장기기억력 : 9 (50%) + 10 (50%)
        #     longTermMemory=np.sum(sum_array_for_one_ques[2,[8,9]]*[0.5,0.5])
        #     #집중력 : 1 (5%) + 2 (5%) + 3 (5%) + 4 (5%) + 5 (5%) + 6 (5%) + 7 (5%) + 8 (50%) + 9 (5%) + 10 (5%) + 11 (5%)
        #     concentration=np.sum(sum_array_for_one_ques[2,[0,1,2,3,4,5,6,7,8,9,10]]*[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.5,0.05,0.05,0.05])
        #     #계산능력 : 8 (100%)
        #     calculationAbility=np.sum(sum_array_for_one_ques[2,[7]])
        #     #개념적 사고력 : 6 (50%) + 7 (50%)
        #     conceptualThinking=np.sum(sum_array_for_one_ques[2,[5,6]]*[0.5,0.5])
        #     #시각적 이해력 : 4 (50%) + 5 (50%)
        #     visualComprehension=np.sum(sum_array_for_one_ques[2,[3,4]]*[0.5,0.5])
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
        
        # 식약처 인증후 점수 산정방법 변경하였음. normal vs abnormal(MCI+AD).
        #"sum_dict": {"AD": 0.2915666898091634, "NOR": 0.3684277335802714, "MCI": 0.8182206352551779}
        if sum_dict["NOR"]>(sum_dict["AD"]+sum_dict["MCI"]):
            dementia_class="NOR"
            dementia_proba= sum_dict["NOR"]
        elif sum_dict["MCI"]>sum_dict["AD"]:
            dementia_class="MCI"
            dementia_proba= sum_dict["MCI"]
        else:
            dementia_class="AD"
            dementia_proba= sum_dict["AD"]


            #dementia_class="AB"
            #dementia_proba= (sum_dict["MCI"]+sum_dict["AD"])/sum(sum_dict.values())
            
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

        logger.debug('{}'.format(results))

        return results
        
    #    logger.debug('{}'.format(time.time() - stime))
    except Exception as ex:
        logger.exception('')
        result = {"resultCode": "9999", "msg": '{}: {}'.format(type(ex).__name__, str(ex))}
        
        return result
    

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
if __name__ == '__main__':
    #pass
    final=execute(["/0b233392-7574-4b16-bb25-45f5e5dcb8dd_0_R",
                   "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_1_R",
                   "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_2_R",
                   "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_3_R",
                   "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_4_R",
                   "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_5_R",
                   "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_6_R",
                   "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_7_R",
                   "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_8_R",
                   "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_9_R",
                   "/0b233392-7574-4b16-bb25-45f5e5dcb8dd_10_R"
                   ])
    # final=execute(["/1694758181080_o3uyZ4",
    #                "/1694758250760_vvW2lP",
    #                "/1694758342430_yPKM6v",        
    #                ])