
from waitress import serve
from flask import Flask, request, Response
from functools import wraps
import time
import configure, utils, diagnosis

app = Flask('dementia')
logger = configure.getLogger('web')


sw_version = 'Release 0.0.041'


def as_json(f):
    '''
    json 응답을 위한 래퍼
    '''
    @wraps(f)
#    @profile
    def decorated_function(*args, **kwargs):
        stime = time.time()
        
        logger.info('========== api request [{}] {} =========='.format(stime, f.__name__))
        logger.info(str(request.get_data(True))[:1000])
        
        try:
            res = f(*args, **kwargs)
        except Exception as ex:
            logger.exception("exception")
            res = str(ex), 500
        
        logger.info('========== api response [{}] {} =========='.format(stime, f.__name__))
        if isinstance(res, tuple): # 내부적으로 응답코드를 컨트롤하기 위한 루틴
            logger.error(str(res))
            return Response(response=res[0], status=res[1])
        else: # 정상 응답처리. 모든 응답은 json응답을 원칙으로 함.
            res = utils.json_dumps(res, ensure_ascii=False)
            logger.debug(res[:1000])
            return Response(res, content_type='application/json; charset=utf-8')

    return decorated_function

def notEmptyValidate(obj, params): # 필수체크 루틴
    for item in params:
        if item not in obj:
            return item
        
    return False

@app.route('/diagnosis', methods=['POST'])
@as_json
def request_diagnosis():
    '''
        진단
    '''
    obj_request = request.get_json()
    obj_response = {
        'resultCode': '0000'
    }
    
    valid_result = notEmptyValidate(obj_request, ['birthYear', 'gender', 'fileList'])
    if valid_result:
        return valid_result, 400
    
    birthYear = obj_request['birthYear']
    gender = obj_request['gender']
    fileList = obj_request['fileList']
    
    resultData = diagnosis.execute(fileList,birthYear,gender)

    if 'resultCode' in resultData:
        obj_response = resultData
    else:
        obj_response['resultData'] = resultData
        
        
    return obj_response

@app.route('/version', methods=['POST'])
@as_json
def request_version():
    obj_response = {
        'resultCode': '0000',
        'resultData': sw_version
    }
    return obj_response

def webStart():
    logger.info('start dementia')
#    opt = dict(host='0.0.0.0', port=configure.port, max_request_body_size=1024*1024*1024*50)
    opt = dict(host='0.0.0.0', port=configure.port)
    if configure.threadcount:
        opt['threads'] = configure.threadcount
    serve(app, **opt)

if __name__ == '__main__':
    pass