#!/usr/bin/env python
"""
@author: Crake
"""
import smtplib
import time
from time import gmtime, strftime, localtime

# Credentials (if needed)
# I recomend you create a new email account just for this
username = 'michalmessenger1234@gmail.com'
password = 'messaging1234'

fromaddr = 'michaelala25@gmail.com'
toaddrs  = '5196160451@vmobile.ca'

def get_Time() :
    return time.time()

def errorTextSend(errorName) :
    '''
    function takes in an error name (as a string), and sends
    a text message alerting the user of the error
    '''
    msg = ('\nERROR\nProcess: Errored Out\n' + str(errorName))

    # The actual mail send
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()


def doneTextSend(start_Time, end_Time, process) :
    '''
    function takes the start and end time (both are floating points) of whatever
    your function is, and the function title (string).
    '''
    timeUsed = end_Time - start_Time
    hoursHold = int(timeUsed / 3600)
    minutesHold = int(timeUsed / 60 - int(hoursHold * 60))
    secondsHold = int(timeUsed - int(minutesHold*60))
    formatedTime = str(hoursHold) + ':' + str(minutesHold) + ':' + str(secondsHold)


    msg = ('\nDONE\nProcess: ' + str(process) + '\nTime required: '  + str(formatedTime))


    # The actual mail send
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()