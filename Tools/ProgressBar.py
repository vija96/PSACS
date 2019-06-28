class ProgressBar:
    def __init__(self, message, iterations, fill='█'):
        self._message = message
        self._iterations = iterations
        self._iteration = 0
        self._fill = fill
        print(self.__str__(), end='')

    def __str__(self):
        if self._iteration == self._iterations+1:
            raise Exception('To many calls to progress bar!')
        length = 50
        spacing = 25-len(self._message)
        filledLength = int(length * self._iteration // self._iterations)
        bar = self._fill * filledLength + '-' * (length - filledLength)
        end = '\r' if self._iteration != self._iterations else '\n'
        percent = ("{0:." + '2' + "f}").format(100 * (self._iteration / float(self._iterations)))
        self._iteration += 1
        return '\r{} |{}| {}% {} {}'.format(self._message+' '*spacing, bar, percent, 'Complete', end)

if __name__ == '__main__':
    from time import sleep
    numberOfSteps = 5
    prgBar = ProgressBar('test', numberOfSteps)
    for i in range(numberOfSteps):
        sleep(1)
        print(prgBar, end='')
