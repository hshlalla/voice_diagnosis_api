import numpy as np

def spider_web_calculate(spider_dict,model,loaded_img,class_1,class_2,step_num,model_class):
#    print("start spiderweb")
    # flowFromDirectory 하게되면 알파벳순으로 결정되므로 Ex(M_A_model에서 AD가 0으로 mci가 1로 나옴.)
    # model class와 뒤바뀌는점 주의
    one_ques_proba1, one_ques_proba2 =model.predict(loaded_img)[0]
    ques_class1_proba = spider_dict.pop("{}_{}".format(class_1,step_num), [])
    ques_class1_proba.append(one_ques_proba2)
    spider_dict["{}_{}".format(class_1,step_num)] = ques_class1_proba

    ques_class2_proba = spider_dict.pop("{}_{}".format(class_2,step_num), [])
    ques_class2_proba.append(one_ques_proba1)
    spider_dict["{}_{}".format(class_2,step_num)] = ques_class2_proba
    if model_class=="NOR_AB":
        ab_proba_for_one_ques = spider_dict.pop("{}_{}".format('AB',step_num))[0]
        spider_dict['MCI_'+str(step_num)].append(ab_proba_for_one_ques) 
        spider_dict['AD_'+str(step_num)].append(ab_proba_for_one_ques)

def spider_web_ability(result_spider_dict,fileList):
    #모델 4개에서 나온값이 더해지면서 key 정렬이 이상해지는 부분 재 sort
    result_spider_dict=dict(sorted(result_spider_dict.items(),key=lambda x:(x[0].split("_")[0],int(x[0].split("_")[1]))))
    sum_array_for_one_ques=np.mean(np.array(list(result_spider_dict.values())),axis=1).reshape(3,len(fileList))
    #sum_dict_for_one_ques = {k: sum(v)/3 for k, v in result_proba_list_spider.items()}
    
    #normal만 가지고 할경우
    if len(fileList)==10:
        #10문제일때 능력치 산출방법 (10번문제 제외)
        #언어능력 1 (10%) + 2 (10%) + 3 (10%) + 4 (10%) + 5 (10%) + 6 (10%) + 7 (10%) + 9 (15%) + 11 (15%)
        verbalAbility=np.sum(sum_array_for_one_ques[2,[0,1,2,3,4,5,6,8,9]]*[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.15,0.15])
        #단기기억력 : 1 (20%) + 2 (20%) + 3 (20%) + 11 (40%)
        shortTermMemory=np.sum(sum_array_for_one_ques[2,[0,1,2,9]]*[0.2,0.2,0.2,0.4])
        #장기기억력 : 9 (100%)
        longTermMemory=np.sum(sum_array_for_one_ques[2,[8]])
        #집중력 : 1 (5%) + 2 (5%) + 3 (5%) + 4 (5%) + 5 (5%) + 6 (5%) + 7 (5%) + 8 (50%) + 9 (7.5%)  + 11 (7.5%)
        concentration=np.sum(sum_array_for_one_ques[2,[0,1,2,3,4,5,6,7,8,9]]*[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.5,0.075,0.075])
        #계산능력 : 8 (100%)
        calculationAbility=np.sum(sum_array_for_one_ques[2,[7]])
        #개념적 사고력 : 6 (50%) + 7 (50%)
        conceptualThinking=np.sum(sum_array_for_one_ques[2,[5,6]]*[0.5,0.5])
        #시각적 이해력 : 4 (50%) + 5 (50%)
        visualComprehension=np.sum(sum_array_for_one_ques[2,[3,4]]*[0.5,0.5])
    elif len(fileList)==11:
        #11문제일때 스파이더 능력치 산출방법
        #언어능력 : 1 (10%) + 2 (10%) + 3 (10%) + 4 (10%) + 5 (10%) + 6 (10%) + 7 (10%) + 9 (10%) + 10 (10%) + 11 (10%)
        verbalAbility=np.mean(sum_array_for_one_ques[2])
        #단기기억력 : 1 (20%) + 2 (20%) + 3 (20%) + 11 (40%)
        shortTermMemory=np.sum(sum_array_for_one_ques[2,[0,1,2,10]]*[0.2,0.2,0.2,0.4])
        #장기기억력 : 9 (50%) + 10 (50%)
        longTermMemory=np.sum(sum_array_for_one_ques[2,[8,9]]*[0.5,0.5])
        #집중력 : 1 (5%) + 2 (5%) + 3 (5%) + 4 (5%) + 5 (5%) + 6 (5%) + 7 (5%) + 8 (50%) + 9 (5%) + 10 (5%) + 11 (5%)
        concentration=np.sum(sum_array_for_one_ques[2,[0,1,2,3,4,5,6,7,8,9,10]]*[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.5,0.05,0.05,0.05])
        #계산능력 : 8 (100%)
        calculationAbility=np.sum(sum_array_for_one_ques[2,[7]])
        #개념적 사고력 : 6 (50%) + 7 (50%)
        conceptualThinking=np.sum(sum_array_for_one_ques[2,[5,6]]*[0.5,0.5])
        #시각적 이해력 : 4 (50%) + 5 (50%)
        visualComprehension=np.sum(sum_array_for_one_ques[2,[3,4]]*[0.5,0.5])

    return verbalAbility, shortTermMemory, longTermMemory, concentration, calculationAbility, conceptualThinking, visualComprehension


if __name__ == '__main__':
    pass