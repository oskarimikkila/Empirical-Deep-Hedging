import json

class Settings():
    def __init__(self):
        self.data = json.load(open('settings.json'))
        
    def save(self, name):
        with open('settings/' + name + '.json', 'w') as f:
            json.dump(self.data, f)
            
    def load(self, name):
        with open('settings/' + name + '.json', 'r') as f:
            self.data = json.load(f)

s = Settings()

def getSettings():
    return s.data

def setSettings(fname):
    s.load(fname)

def saveSettings(fname):
    s.save(fname)