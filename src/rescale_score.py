def rescale_30_score(dementia_class,dementia_proba,sum_dict):
    dementia_proba=int(dementia_proba*100)
    if dementia_class == "NOR":
        if (93 < dementia_proba ):
            score = 30
        elif (86 < dementia_proba <= 93):
            score = 29
        elif (79 < dementia_proba <= 86):
            score = 28
        elif (72 < dementia_proba <= 79):
            score = 27
        elif (65 < dementia_proba <= 72):
            score = 26
        elif (58 < dementia_proba <= 65):
            score = 25
        else:
            score = 24
    elif dementia_class == "MCI":
        if (40 < dementia_proba <= 80) and sum_dict["NOR"] > sum_dict["AD"]:
            score =23
        elif (80 < dementia_proba):
            score = 22
        elif (40 < dementia_proba <= 80) and sum_dict["NOR"] < sum_dict["AD"]:
            score = 21
        else:
            score = 20
    else:
        if (40 < dementia_proba <= 60): #
            score = 19
        elif (60 < dementia_proba <= 65):
            score = 18
        elif (65 < dementia_proba <= 68):
            score = 17
        elif (68 < dementia_proba <= 70):
            score = 16
        elif (70 < dementia_proba <= 71):
            score = 15
        elif (71 < dementia_proba <= 72):
            score = 14
        elif (72 < dementia_proba <= 73):
            score = 13
        elif (73 < dementia_proba <= 74):
            score = 12
        elif (74 < dementia_proba <= 75):
            score = 11
        elif (75 < dementia_proba <= 76):
            score = 10
        elif (76 < dementia_proba <= 77):
            score = 9
        elif (77 < dementia_proba <= 78):
            score = 8
        elif (78 < dementia_proba <= 79):
            score = 7
        elif (79 < dementia_proba <= 80):
            score = 6
        elif (80 < dementia_proba <= 82):
            score = 5
        elif (82 < dementia_proba <= 85):
            score = 4
        elif (85 < dementia_proba <= 90):
            score = 3
        elif (90 < dementia_proba):
            score = 2
        else:
            score = 1
    return score

def rescale_100_score(dementia_class,dementia_proba,sum_dict):
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
        elif (66 < dementia_proba <= 68):
            score = 83
        elif (64 < dementia_proba <= 66):
            score = 82
        elif (62 < dementia_proba <= 64):
            score = 81
        elif (60 < dementia_proba <= 62):
            score = 80
        elif (57 < dementia_proba <= 60):
            score = 79
        elif (54 < dementia_proba <= 57):
            score = 78
        else:
            score = 77
    elif dementia_class == "MCI":
        if (40 < dementia_proba <= 60) and sum_dict["NOR"] > sum_dict["AD"]:
            score =76
        elif (60 < dementia_proba <= 70) and sum_dict["NOR"] > sum_dict["AD"]:
            score = 75
        elif (70 < dementia_proba <= 72) and sum_dict["NOR"] > sum_dict["AD"]:
            score = 74
        elif (90 < dementia_proba):
            score = 73
        elif (87 < dementia_proba <= 90):
            score = 72
        elif (84 < dementia_proba <= 87):
            score = 71
        elif (81 < dementia_proba <= 84):
            score = 70
        elif (78 < dementia_proba <= 81):
            score = 69
        elif (75 < dementia_proba <= 78):
            score = 68
        elif (72 < dementia_proba <= 75):
            score = 67
        elif (70 < dementia_proba <= 72) and sum_dict["NOR"] < sum_dict["AD"]:
            score = 66
        elif (60 < dementia_proba <= 70) and sum_dict["NOR"] < sum_dict["AD"]:
            score = 65
        elif (40 < dementia_proba <= 60) and sum_dict["NOR"] < sum_dict["AD"]:
            score = 64
        else:
            score = 63
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
# test
# dementia_class="AD"
# dementia_proba=0.99
# sum_dict={"NOR":68,"AD":77}
# score=rescale_100_score(dementia_class,dementia_proba,sum_dict)
# print(score)