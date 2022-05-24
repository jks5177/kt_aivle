from flask import Flask

app = Flask(__name__)

@app.route('/')
def index() :
    return 'hello world~!'

@app.route('/user/<Name>')
def hello_user(Name) :
    return f'Hello, {Name}~!'

if __name__ == '__main__' :
    app.run()