def spider_rescale_100_score(dementia_class,dementia_proba,sum_dict):
    dementia_proba=int(dementia_proba*100)
    if dementia_class == "NOR":
        if (98 < dementia_proba ):
            score = 99
        elif (96 < dementia_proba <= 98):
            score = 98
        elif (94 < dementia_proba <= 96):
            score = 97
        elif (92 < dementia_proba <= 94):
            score = 96        
        elif (90 < dementia_proba <= 92):
            score = 95        
        elif (88 < dementia_proba <= 90):
            score = 94        
        elif (86 < dementia_proba <= 88):
            score = 93        
        elif (84 < dementia_proba <= 86):
            score = 92        
        elif (82 < dementia_proba <= 84):
            score = 91        
        elif (80 < dementia_proba <= 82):
            score = 90
        elif (78 < dementia_proba <= 80):
            score = 89
        elif (76 < dementia_proba <= 78):
            score = 88
        elif (74 < dementia_proba <= 76):
            score = 87
        elif (72 < dementia_proba <= 74):
            score = 86
        elif (70 < dementia_proba <= 72):
            score = 85
        elif (68 < dementia_proba <= 70):
            score = 84
        elif (60 < dementia_proba <= 65):
            score = 83
        elif (55 < dementia_proba <= 60):
            score = 82
        elif (50 < dementia_proba <= 55):
            score = 81
        else:
            score = 80
    
    elif dementia_class == "MCI":
        if (40 < dementia_proba <= 60) and sum_dict["NOR"] > sum_dict["AD"]:
            score =79
        elif (60 < dementia_proba <= 65) and sum_dict["NOR"] > sum_dict["AD"]:
            score = 78
        elif (65 < dementia_proba <= 70) and sum_dict["NOR"] > sum_dict["AD"]:
            score = 77
        elif (70 < dementia_proba <= 72) and sum_dict["NOR"] > sum_dict["AD"]:
            score = 76
        elif (90 < dementia_proba):
            score = 75
        elif (87 < dementia_proba <= 90):
            score = 74
        elif (84 < dementia_proba <= 87):
            score = 73
        elif (81 < dementia_proba <= 84):
            score = 71
        elif (78 < dementia_proba <= 81):
            score = 70
        elif (75 < dementia_proba <= 78):
            score = 69
        elif (72 < dementia_proba <= 75):
            score = 68
        elif (70 < dementia_proba <= 72) and sum_dict["NOR"] < sum_dict["AD"]:
            score = 69
        elif (60 < dementia_proba <= 70) and sum_dict["NOR"] < sum_dict["AD"]:
            score = 68
        elif (40 < dementia_proba <= 60) and sum_dict["NOR"] < sum_dict["AD"]:
            score = 67
        else:
            score = 66
    else:
        if (40 < dementia_proba <= 60):
            score = 62
        elif (60 < dementia_proba <= 69):
            score = 61
        elif (69 < dementia_proba):
            score = (100-dementia_proba)*2
        else:
            score = 1
    return score