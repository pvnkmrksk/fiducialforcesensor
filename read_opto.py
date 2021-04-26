
'''
Sanity check, can we read optoforce using Shadow Robot's library.
The library: # https://github.com/shadow-robot/optoforce/blob/indigo-devel/optoforce/src/optoforce/optoforce_node.py

Date: 16 Sep 2019
Author: nouyang

This was used for debugging only, not for collecting data for the paper.
'''

import serial
import sys
import logging
import optoforcelibrary as optoforce
import numpy as np
import pprint

from datetime import datetime


def main():
    port = "/dev/ttyACM0"
    sensor_type = "s-ch/6-axis"
    starting_index = 0
    scaling_factors =  [[1,1,1,1,1,1]] #one per axis

    strtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    o_fname = strtime + '_optoforceData.csv'

    if len(sys.argv) > 1:
        port = "/dev/" + sys.argv[1]

    # Initialize optoforce driver
    try:
        driver = optoforce.OptoforceDriver(port,
                                         sensor_type,
                                         scaling_factors)

    except serial.SerialException as e:
        print('fail!')
        raise

    fields = ['Fx: ', 'Fy: ', 'Fz: ', 'Mx: ', 'My: ', 'Mz: ']

    while True:
        data = driver.read()
        if isinstance(data, optoforce.OptoforceData):
            a = ['{0: <10}'.format(x) for x in data.force[0]]
            # pprint.pprint(' '.join(a))
            pprint.pprint(''.join([val for pair in zip(fields, a) for val in \
                                   pair]))
            # print(data.force)
            # with open(o_fname,'a') as outf:
                # outf.write(opto_data_str)
                # outf.flush()
        elif isinstance(data, optoforce.OptoforceSerialNumber):
            print("The sensor's serial number is " + str(data))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        server.close()
        server_thread.join()
        ser.close()
