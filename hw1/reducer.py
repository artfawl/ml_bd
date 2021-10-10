#!/usr/bin/env python3

import sys


def read_input(file):
    for line in file:
        yield line[1:-3]


def main():
    stream = read_input(sys.stdin)
    res_mean = 0
    res_var = 0
    total = 0
    for data in stream:
        mas = data.split(',')
        try:
            ck, mk, vk = int(mas[0]), float(mas[1]), float(mas[2])

        except Exception:
            print(mas[0], mas[1], mas[2])
            continue
        res_var = (total * res_var + ck * vk) / (total + ck) + total * ck * ((res_mean - mk) / (total + ck))**2
        res_mean = (res_mean * total + ck * mk) / (total + ck)
        total += ck
    print(res_mean, res_var)


if __name__ == '__main__':
    main()


