# -*- coding: utf-8 -*-


import os, sys

def main():
    os.environ['DEMENTIA_CONFIG_FILE'] = sys.argv[1] if len(sys.argv) > 1 else 'configure.json'
    import web
    
    web.webStart()

def test():
    pass
    
if __name__ == '__main__':
    main()
#    test()
