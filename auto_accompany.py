import pretty_midi
import time
import threading, time
import matplotlib.pyplot as plt
import fluidsynth
# print(dir(fluidsynth))
# pyFluidSynth	1.2.5	1.2.5
import sys
from scipy import stats

# next week
# 1 change beats by 0.5 or less, look back 2 beats
# 2 diff and 1-x pitch
# 3 check back several seconds for onset
# 4 how to interact score following and vocal tracking

import pretty_midi
import scikits.audiolab
import pyaudio
import analyse
import time
import copy
from scipy.integrate import quad
import wave
import numpy as np
import math
import sys
import pretty_midi
import matplotlib.pyplot as plt
from madmom.features.onsets import CNNOnsetProcessor
# import librosa
# from google.cloud import speech
# from google.cloud.speech import enums
# from google.cloud.speech import types
# import io
import scipy.stats
import os
import statsmodels.api as sm



# name of the song!!!
name = "1"


# file is for no gap
# file1 is for with gap
time_beat_file = 'time_beat_file{}.txt'.format(name)
confidence_file = 'confidence_queue_file{}.txt'.format(name)
ACC_FILE = 'midi{}.mid'.format(name)
time_beat_file = open(time_beat_file, 'r')
confidence_queue_file = open(confidence_file, 'r')
beat_lines = time_beat_file.readlines()
confidence_lines = confidence_queue_file.readlines()
time_list_for_beat = []
confidence_queue = [0.001, 0.001, 0.001, 0.001, 0.001]
for item in beat_lines:
    time_list_for_beat.append(float(item))
for item2 in confidence_lines:
    confidence_queue.append(float(item2))
time_list_for_beat.pop(0)
# print time_list_for_beat
# print confidence_queue
#
#
# time_list_for_beat = [1.0245804988662135, 2.017142857142858, 2.817596371882087, 4.194376417233562,
#                       6.8839002267573735, 8.356734693877556, 9.73351473922903, 10.565986394557829, 11.398458049886628,
#                       13.735782312925178, 14.568253968253977, 15.848979591836745, 17.25777777777779, 18.122267573696156,
#                       18.954739229024955, 19.787210884353755, 21.228027210884367, 22.66884353741498, 23.661405895691622,
#                       24.653968253968266, 26.222857142857155, 27.2154195011338, 28.175963718820874, 29.392653061224504,
#                       30.32117913832201, 31.249705215419514, 32.94666666666668, 34.41950113378686, 35.4120634920635,
#                       36.85287981859412, 38.54984126984129, 39.38231292517008, 40.182766439909315, 42.1678911564626,
#                       43.67274376417235, 45.625850340136076, 46.58639455782315, 47.70702947845807, 48.955736961451265]
#
# confidence_queue = [0.001, 0.001, 0.001, 0.001, 0.001, 0.01680439691005099, 0.025696367422475337, 0.023696269443428553,
#                     0.01579219497480256,
#                     0.31387006961386604, 0.5409981636348146, 0.03655208705668157, 0.029721971509192053,
#                     0.018465036511719247, 0.42286285523358685, 0.021179943689310872, 0.014904403134944578,
#                     0.014888044056185469, 0.02525525650413246, 0.18538281793488567, 0.10982516131219708,
#                     0.04167974031295919, 0.027697851131602762, 0.07481549920656361, 0.017420712846597124,
#                     0.38857813020598003, 0.01914642937567863, 0.029331458094393352, 0.01672554756282718,
#                     0.04547653549621592, 0.040467227954676444, 0.053486221134292126, 0.039381361198063985,
#                     0.7076970553846177, 0.02344335259103856, 0.014321139113852315, 0.040491705237417915,
#                     0.03318184415616462, 0.026175802267463313, 0.0437447015503212, 0.39219560497441486,
#                      0.029941874651415755, 0.0199186483771654, 0.018797349712122945]
#
# # no gap
# time_list_for_beat = [1.0245804988662135, 2.017142857142858, 2.817596371882087, 3.8741950113378696,
#                       4.930793650793652, 6.371609977324265, 7.8124263038548785, 8.612879818594108, 9.413333333333338,
#                       10.213786848072568, 11.014240362811798, 12.45505668934241, 13.895873015873022, 14.696326530612252,
#                       15.496780045351482, 16.937596371882094, 18.378412698412706, 19.178866213151935,
#                       19.979319727891163, 21.420136054421775, 22.860952380952387, 23.693424036281186,
#                       24.525895691609986, 25.326349206349214, 26.126802721088442, 27.567619047619054,
#                       29.008435374149666, 29.808888888888895, 30.609342403628123, 31.537868480725628,
#                       32.466394557823136, 33.90721088435375, 35.34802721088437, 36.1484807256236, 36.94893424036283,
#                       38.38975056689345, 39.83056689342406, 40.66303854875286, 41.495510204081654, 42.45605442176873,
#                       43.4165986394558, 44.21705215419503, 45.01750566893426, 46.45832199546488, 47.899138321995494,
#                       49.33995464852611]
#
# confidence_queue = [0.06598673990718819, 0.031787593524997576, 0.023684094501744243, 0.016735330013392717,
#                     0.19503043446128776, 0.019648306456893402, 0.10237781775335043, 0.12215676333391315,
#                     0.39814179347270157, 0.02261208586731391, 0.21804067783121658, 0.015528138887145364,
#                     0.0388119726669978, 0.3337597551499176, 0.045003733485178575, 0.0590415064647253,
#                     0.0199473227686229, 0.15194815063764344, 0.23678576669563586, 0.4480942787515414,
#                     0.016159605442821003, 0.0636828866441907, 0.023422364889580476, 0.04907478226829136,
#                     0.4844734953660967, 0.018003773108546093, 0.18336506474122072, 0.09010382558746237,
#                     0.1608517163214111, 0.050420703451330436, 0.09418395011746532, 0.061422914318270454,
#                     0.07079889135257326, 0.01938946747707268, 0.05863424788387084, 0.05081029642317815,
#                     0.26215571592689885, 0.4266033648695883, 0.06842547809717366, 0.022556043118477465,
#                     0.040515361550726894, 0.20454609504890475, 0.7120396359289378, 0.0, 0.0, 0.0]


# ACC_FILE = 'scale.mid'
BPM = 60
BPS = BPM / float(60)  # beat per second
original_begin = time.clock()
global_tempo = 0
# fluidsynth.init("soundfont.sf2")
# fluidsynth need some time to load sound file otherwise there is a
# chance there is no sound
# tmp  = raw_input('press return key to begin this program')

# weight is for 0.5beats for 2 beats
weight_judge = True
beat_back = 4
pressed_key = "lol"
timeQueue = []
stop_thread = False
sQueue = []
latency_end = -1
# for simulation
# speed = 0.8
# timeQueue = [-4,-3,-2,-1,0]
# x = timeQueue[-4:]
# y = range(1, 5)
# s0, intercept, r_value, p_value, std_err = stats.linregress(x, y)
# sQueue.append(s0)
# simu 2
# simulation_times = [1, 2, 3, 4, 5, 6, 7, 8]
simulation_times = [1, 2, 3, 4, 5]
for i in time_list_for_beat:
    simulation_times.append(i + 5)
# simulation_times = time_list_for_beat
# tmp = 8
# for i in range(28):
#     tmp = tmp + 2
#     simulation_times.append(tmp)

'''
function: press_key_thread:
--------------------------------------------------------
init another thread, take in keyboard input
each tap on return/enter key is registered as a beat
and save to global queue timeQueue
'''
from playsound import playsound


def press_key_thread():
    global pressed_key
    global stop_thread
    global latency_end
    cnt = 0
    while not stop_thread:
        # pressed_key = sys.stdin.readline()
        # if pressed_key=='\n':
        # origianl version
        # if cnt < len(simulation_times) and (
        #         abs(time.clock() - original_begin - simulation_times[cnt])) <= 0.003 or time.clock() - original_begin >= \
        #         simulation_times[cnt]:
        # new version:
        if cnt < len(simulation_times) and (
                 abs(time.clock() - original_begin - simulation_times[cnt])) <= 0.003 or time.clock() - original_begin >= \
                 simulation_times[cnt]:
            timeQueue.append(time.clock() - original_begin)
            cnt = cnt + 1
            # print timeQueue
            if len(timeQueue) == 5:
                print "timeQueue == 5"
                if weight_judge:
                    b0 = 1
                    t0 = timeQueue[-1]
                    x = timeQueue[-beat_back:]
                    y = range(1, beat_back + 1)
                    # print x
                    # print y
                    confidence_block = confidence_queue[len(timeQueue) - beat_back:len(timeQueue) - 0]
                    sum_confidence = sum(confidence_block)
                    confidence_count = []
                    for index in range(len(confidence_block)):
                        confidence_count.append(round(confidence_block[index] / sum_confidence, 1))
                    # print confidence_count
                    for index in range(len(confidence_count)):
                        print confidence_count[index]
                        times = int(confidence_count[index] * 10)
                        for i in range(times - 1):
                            x.append(x[index])
                            y.append(y[index])
                    # print x
                    # print y
                    s0, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    sQueue.append(s0)
                else:
                    b0 = 1
                    t0 = timeQueue[-1]
                    s0 = float(1) / (timeQueue[-2] - timeQueue[-3])
                    x = timeQueue[-4:]
                    y = range(1, 5)
                    # s0, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    sQueue.append(s0)
            if len(timeQueue) % 2 == 1 and len(timeQueue) > 5:
                if latency_end == -1:
                    l = 0.1
                else:
                    l = max(0, latency_end - time.clock())
                if weight_judge:
                    b0, t0, s0 = compute_tempo_ratio_weighted(b0, t0, s0, l)
                else:
                    b0, t0, s0 = compute_tempo_ratio(b0, t0, s0, l)
            # extra
            # if len(timeQueue) % 2 == 0 and len(timeQueue) > 5:
            #     if latency_end == -1:
            #         l = 0.1
            #     else:
            #         l = max(0, latency_end - time.clock())
            #     if weight_judge:
            #         b0, t0, s0 = compute_tempo_ratio_weighted(b0, t0, s0, l)
            #     else:
            #         b0, t0, s0 = compute_tempo_ratio(b0, t0, s0, l)
            pressed_key = 'lol'


'''
function: compute_tempo_ratio:
----------------------------------------------------
scheduleing algo by NIME 2011 paper
'''


def compute_tempo_ratio(b0, t0, s0, l):
    t1 = timeQueue[-1]
    b = b0 + (t1 + l - t0) * s0
    tn = t1 + l
    bn = b
    te = timeQueue[-2]
    # be = len(timeQueue) - 5
    be = len(timeQueue) - 5
    se = float(1) / (timeQueue[-1] - timeQueue[-2])
    # for normal regression
    sn = (float(4) / (te * se - tn * se - be + bn + 4)) * se
    # sn = (float(4) * se / (te * se - tn * se - be + bn + 4))
    # print(sn)
    sQueue.append(sn)
    return bn, tn, sn


def compute_tempo_ratio_weighted(b0, t0, s0, l):
    t1 = timeQueue[-1]
    b = b0 + (t1 + l - t0) * s0
    tn = t1 + l
    bn = b
    te = timeQueue[-2]
    # se = float(1) / (timeQueue[-1] - timeQueue[-2])
    # for long break to stop regression
    # global_tempo = float(1) / (timeQueue[-1] - timeQueue[-2])
    be = len(timeQueue) - 5
    x = timeQueue[-beat_back:]
    y = range(len(timeQueue) - beat_back - 3, len(timeQueue) - 3)
    confidence_block = confidence_queue[len(timeQueue) - beat_back:len(timeQueue) - 0]
    # print x
    # print y
    # print confidence_block
    # naive version
    # sum_confidence = sum(confidence_block)
    # confidence_count = []
    # for index in range(len(confidence_block)):
    #     confidence_count.append(round(confidence_block[index] / sum_confidence, 1))
    # # print confidence_count
    # for index in range(len(confidence_count)):
    #     # print confidence_count[index]
    #     times = int(confidence_count[index] * 10)
    #     for i in range(times - 1):
    #         x.append(x[index])
    #         y.append(y[index])
    # se, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # print "after weight"
    # # # print confidence_count
    # true weighted
    # print "------------------------------x timeQueue y range-----------------------"
    # print x
    # print y
    # print "----------------------------confidence_block"
    # print confidence_block
    x = sm.add_constant(x)
    if y[0] == 0:
        print "first one -------------------------------------"
        wls_model = sm.WLS(y, x)
        results = wls_model.fit()
        se = results.params[1]
    else:
        wls_model = sm.WLS(y, x, weights=confidence_block)
        results = wls_model.fit()
        se = results.params[1]
    print "------------------------------sk-----------------------"
    print se

    # weird version
    # print "------------------------------x timeQueue y range-----------------------"
    # print x
    # print y
    # # # x = sm.add_constant(x)
    # print "----------------------------confidence_block"
    # print confidence_block
    # # x = sm.add_constant(x)
    # if y[0] == 0:
    #     print "first one -------------------------------------"
    #     wls_model = sm.WLS(y, x)
    #     results = wls_model.fit()
    #     se = results.params[0]
    # else:
    #     wls_model = sm.WLS(y, x, weights=confidence_block)
    #     results = wls_model.fit()
    #     se = results.params[0]
    # print "------------------------------sk-----------------------"
    # print se


    # for normal regression
    sn = (float(4) / (te * se - tn * se - be + bn + 4)) * se
    # sn = (float(4) * se / (te * se - tn * se - be + bn + 4))
    # print(sn)
    sQueue.append(sn)
    return bn, tn, sn


old_midi = pretty_midi.PrettyMIDI(ACC_FILE)
new_midi = pretty_midi.PrettyMIDI()
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program)


class Player:
    def __init__(self, ACC_FILE, original_begin, BPM, fs):
        self.ACC_FILE = ACC_FILE
        self.original_begin = original_begin
        self.midi_data = pretty_midi.PrettyMIDI(ACC_FILE)
        self.notes = sorted(self.midi_data.instruments[0].notes, key=lambda x: x.start, reverse=False)
        self.playTimes = []
        self.noteTimes = []
        self.midi_start_time = self.notes[0].start
        # self.BPS = BPM / float(60)
        self.BPS = BPM / float(60)
        self.fs = fs

    '''
    function: follow:
    -----------------------------------------------
    follow score from start point
    '''

    def follow(self, start):
        global sQueue
        global timeQueue
        global latency_end
        begin = time.clock()

        for i in range(start, len(self.notes)):
            # for simulation
            # if len(timeQueue) == 5:
            #     b0 = 1
            #     t0 = timeQueue[-1]
            #     s0 = float(1)/(timeQueue[-2]-timeQueue[-3])
            #     x = timeQueue[-4:]
            #     y = range(1,5)
            #     # s0, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            #     sQueue.append(s0)
            # if len(timeQueue)%2==1 and len(timeQueue)>5:
            #     print(latency_end)
            #     if latency_end == -1:
            #        l = 0.1
            #     else:
            #        l = max(0,latency_end - time.clock())
            #     b0,t0,s0=compute_tempo_ratio(b0,t0,s0,l)
            # if len(timeQueue)%2==0 and len(timeQueue)>5:
            #     print(latency_end)
            #     if latency_end == -1:
            #        l = 0.1
            #     else:
            #        l = max(0,latency_end - time.clock())
            #     b0,t0,s0=compute_tempo_ratio(b0,t0,s0,l)

            note = self.notes[i]
            cur_time = time.clock() - begin
            wait_delta = note.start - cur_time
            if cur_time > 49:
                break
            tempo_ratio = float(self.BPS) / sQueue[-1]
            if tempo_ratio < 1:
                begin -= wait_delta * (1 - tempo_ratio)
                wait_delta = wait_delta * tempo_ratio
            elif tempo_ratio > 1:
                begin += wait_delta * (tempo_ratio - 1)
                wait_delta = wait_delta * tempo_ratio

            target_start_time = time.clock() + wait_delta
            latency_end = target_start_time
            while time.clock() < target_start_time:
                # if time.clock() - old_target_start_time > 4 * (1 / tempo_ratio):
                #     tempo_ratio = global_tempo
                #     sQueue[-1] = global_tempo
                #     break
                pass

            delta_time = note.end - (time.clock() - begin)
            # print 'delta_time%f'%(delta_time-note.end+note.start)

            self.playTimes.append(time.clock() - original_begin)
            self.noteTimes.append(note.start)

            tempo_ratio = float(self.BPS) / sQueue[-1]
            if tempo_ratio < 1:
                print('tempo faster with ratio %f' % tempo_ratio)
                begin -= delta_time * (1 - tempo_ratio)
                delta_time = delta_time * tempo_ratio
            elif tempo_ratio >= 1:
                print('tempo slower with ratio %f' % tempo_ratio)
                begin += delta_time * (tempo_ratio - 1)
                delta_time = delta_time * tempo_ratio

            # old_note = self.midi_data.instruments[0].notes[i]
            # dur = old_note.end-old_note.start
            # new_note = pretty_midi.Note(velocity=old_note.velocity, pitch=old_note.pitch, start=old_note.start, end=old_note.end)
            # piano.notes.append(new_note)

            # normal version

            cur_time = time.clock() - begin
            dur = note.end - note.start
            new_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=cur_time, end=cur_time + dur)
            piano.notes.append(new_note)

            # self.fs.noteon(0, note.pitch, 100)
            target_time = time.clock() + delta_time
            latency_end = target_time
            while time.clock() < target_time:
                pass
            # self.fs.noteoff(0, note.pitch)

            # for count for long break
            old_target_start_time = target_start_time + 0
            # for simulation
            # new_gap = speed * (timeQueue[-1] - timeQueue[-2])
            # new_gap = 0.5
            # timeQueue.append(timeQueue[-1]+new_gap)
            # print(sQueue)
            # print(timeQueue)

        tap_time = [t for t in timeQueue]
        tap_beat = [(t - 4) / float(self.BPS) for t in range(len(tap_time))]
        # for simulation
        # tap_beat = [(t) / float(self.BPS) for t in range(len(tap_time))]
        # print(tap_time)
        # print(self.playTimes)
        # print(tap_beat)
        # print(self.noteTimes)
        # tap_beat = tap_beat[3::]
        # tap_time = tap_time[3::]
        # print(tap_beat)

        new_midi.instruments.append(piano)
        new_midi.write("auto_accompany.mid")

        plt.scatter(x=self.playTimes, y=self.noteTimes, c='b', s=10, marker='o')
        plt.plot(tap_time, tap_beat, marker='+')
        plt.xlabel('audio time (seconds)')
        plt.ylabel('score time (seconds)')
        plt.show()

    '''
    function:jump
    ---------------------------------------
    jump to specified ith note
    '''

    def jump(self, i):
        self.follow(i)


#
if __name__ == '__main__':
    pk_thread = threading.Thread(target=press_key_thread)
    pk_thread.start()
    fs = fluidsynth.Synth()
    sfid = fs.sfload("soundfont.sf2")
    fs.start("coreaudio")
    fs.program_select(0, sfid, 0, 0)
    try:
        # print('tap four times to start')
        while True:
            if len(timeQueue) >= 5:
                break
        player = Player(ACC_FILE, original_begin, BPM, fs)
        player.follow(0)
    except KeyboardInterrupt:
        stop_thread = True
        pk_thread.killed = True
        pk_thread.join()
        fs.delete()
    finally:
        stop_thread = True
        pk_thread.killed = True
        pk_thread.join()
        fs.delete()
