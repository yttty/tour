from time import strftime, localtime


class Logger:
    def __init__(self, app_name: str, out: str = 'stdout'):
        self.app_name = app_name
        if out == 'stdout':
            self.log_file = None
        else:
            self.log_file = open(out, 'w')

    def __del__(self):
        if self.log_file:
            self.log_file.close()

    def log(self, msg: str, level: str):
        msg = f'''[{level} {strftime("%Y-%m-%d %H:%M:%S", localtime())} {self.app_name}] {msg}'''
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()
        else:
            print(msg)

    def info(self, msg):
        self.log(msg, level="I")

    def debug(self, msg):
        self.log(msg, level="D")
