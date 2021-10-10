#!/usr/bin/env python3
import sys


def read_input(file):
    for line in file:
        yield float(line)


def main(k=50):
    stream = read_input(sys.stdin)
    counter = 0
    mean = variance = 0
    cash = []
    for data in stream:
        if counter < k:
            if data == 0:
                continue
            mean += data
            cash.append(data)
            counter += 1
        else:
            mean /= counter
            for var in cash:
                variance += (var - mean) ** 2
            variance /= counter
            print('({},{},{})'.format(counter,mean,variance))
            counter = mean = variance = 0
            cash = []
    mean /= counter
    for var in cash:
        variance += (var - mean) ** 2
    variance /= counter
    print('({},{},{})'.format(counter,mean,variance))


if __name__ == '__main__':
    main()
