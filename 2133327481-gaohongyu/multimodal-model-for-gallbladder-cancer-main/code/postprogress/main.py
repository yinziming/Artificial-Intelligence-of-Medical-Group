import os

from logic import Main

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    while Main.menu():
        pass
