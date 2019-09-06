#!/usr/bin/env python
# -*- coding: utf-8 -*- 
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
import io
import scipy.stats
import os


std = 5
resolution = 0.01
CHUNK = 1412
time_int= float(1412)/44100
alpha = 10
score_start_time = 0

#Rc = 74
DIRECTORY = '/Users/rtchen/Downloads/sing_dataset/001/'
#DIRECTORY = '/Users/rtchen/Downloads/onset_detection/changba/0006 - 嘴巴嘟嘟/'

FILTER =  'gate'

# need to change
# AUDIOFILE = os.path.join(DIRECTORY,'audio.wav')
# MIDIFILE = os.path.join(DIRECTORY,'midi.mid')
# PITCHFILE = os.path.join(DIRECTORY,'pitch.txt')
# NEWFILE = os.path.join(DIRECTORY,'generated_{}filter_nodtw.mid'.format(FILTER))
# CONF_FILE = os.path.join(DIRECTORY,'{}filter_confidence_nodtw.txt'.format(FILTER))
# TEMPO_FILE = os.path.join(DIRECTORY,'tempo.txt')
# Rc =  int(open(TEMPO_FILE, "r").readline().strip('\n'))

# to replace
name = "1"

# AUDIOFILE = os.path.join(DIRECTORY,'audio.wav')
# MIDIFILE = os.path.join(DIRECTORY,'midi.mid')
# PITCHFILE = os.path.join(DIRECTORY,'pitch.txt')
AUDIOFILE = 'audio{}.wav'.format(name)
MIDIFILE = 'midi{}.mid'.format(name)
PITCHFILE = 'pitch3.txt'
# NEWFILE = 'generated_{}filter.mid'.format(FILTER)
NEWFILE = 'score_generated{}.mid'.format(name)
CONF_FILE = '{}filter_confidence.txt'.format(FILTER)

# shuo san jiu san
# Rc = 70
# vineyard
Rc = 80
# # zhui guang zhe
# Rc = 74

# PITCHFILE = os.path.join(DIRECTORY,'pitch.txt')
# AUDIOFILE = 'audio{}.wav'.format(name)
# MIDIFILE = 'midi{}.mid'.format(name)
# PITCHFILE = 'pitch3.txt'
# NEWFILE = 'generated_{}filter.mid'.format(FILTER)
# NEWFILE = 'score_generated{}.mid'.format(name)
# CONF_FILE = '{}filter_confidence.txt'.format(FILTER)

#---------
performance_start_time = 0
score_end_time =184
WINSIZE = 1
BREAK_TIME = 184
#WEIGHT = [0.1,0.1,0.2,0.2,0.4]
WEIGHT = [0.5]
'''
Rc = 120
AUDIOFILE = 'happybday.wav'
MIDI_NAME = 'happybday.mid'
performance_start_time = 0
score_end_time = 12.3
'''
#score_end_time = 11.05
#score_end_time= 29



def normpdf(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def get_time_axis(resolution,start_time, end_time,filename):
    axis_start_time = 0
    axis_end_time = math.floor(end_time/resolution)*resolution
    size = (axis_end_time - axis_start_time)/resolution+1
    axis = np.linspace(axis_start_time,axis_end_time,size)
    if abs(axis_end_time-end_time) > 1e-5:
        axis = np.concatenate((axis, [end_time]))
    scoreLen = len(axis)
    score_midi = np.zeros(scoreLen)
    midi_data = pretty_midi.PrettyMIDI(filename)
    score_onsets = np.zeros(scoreLen)
    onsets = []
    onset0 = np.inf
    for note in midi_data.instruments[0].notes:
        start = int(math.floor(note.start/resolution))
        end = int(math.ceil(note.end/resolution))+1
        if start < len(score_onsets):
            score_onsets[start] = 1
        onsets.append(start)
        if start < onset0:
            onset0 = start
        for j in range(start,end):
            if j < len(score_midi):
                score_midi[j] = note.pitch
    #plt.plot(score_onsets)
    #plt.show()
    return score_midi,axis,score_onsets,onsets,onset0

def diff(score):
    diff_score = np.zeros(len(score)) 
    for i in range(len(score)-1):
        diff_score[i] = score[i+1]-score[i]
    return diff_score

def sigmoid(x):
     return 1 / (1 + math.e ** -x)
     


def similarity(onset_prob,score_onset):
    sim = float(min(onset_prob,score_onset)+1e-6)/(max(onset_prob,score_onset)+1e-6)
    return sim


def compute_f_V_given_I(pitch,pitches,scoreLen,score_midi,onset_prob,score_onsets,alpha,feature,w1,w2,w3):
    f_V_given_I = np.zeros(scoreLen)
    sims = np.zeros(scoreLen)
    
    if len(pitches) > WINSIZE:
        pitch = pitch - np.dot(pitches[-1-WINSIZE:-1],WEIGHT)
    elif len(pitches)>1:
        pitch = pitch - float(sum(pitches[:-1]))/(len(pitches)-1)
    else:
        pitch = 0
    '''
    if len(pitches) > WINSIZE:
        pitch = np.dot(pitches[-WINSIZE:],WEIGHT) - np.dot(pitches[-1-WINSIZE:-1],WEIGHT)
    elif len(pitches)>1:
        pitch = pitch - float(sum(pitches[:-1]))/(len(pitches)-1)
    else:
        pitch = 0
    ''' 
    
    for i in range(scoreLen):
        
        if i >= WINSIZE:
            score_pitch = score_midi[i] - np.dot(score_midi[i-WINSIZE:i],WEIGHT)           
        elif i>0:
            score_pitch = score_midi[i] - float(sum(score_midi[:i]))/i
        else:
            score_pitch = 0
        '''
        if i > WINSIZE:
            score_pitch = np.dot(score_midi[i-WINSIZE+1:i+1],WEIGHT) - np.dot(score_midi[i-WINSIZE:i],WEIGHT)           
        elif i>1:
            score_pitch = score_midi[i] - float(sum(score_midi[:i]))/i
        else:
            score_pitch = 0
        '''
        score_onset = score_onsets[i]
        if feature == 'onset':
            f_V_given_I[i] = math.pow(math.pow(normpdf(pitch, score_pitch, std),w1) *math.pow(similarity(onset_prob,score_onset),w2),w3)
            #f_V_given_I[i] = normpdf(pitch, score_pitch, std)
            #f_V_given_I[i] =  similarity(onset_prob,score_onset)
        elif feature == 'uniform':
            f_V_given_I[i] =  1
        elif feature == 'only':
            f_V_given_I[i] =  similarity(onset_prob,score_onset)
        elif feature == 'both':
            f_V_given_I[i] = math.pow(normpdf(pitch, score_pitch, std),0.5) *math.pow(similarity(onset_prob,score_onset),0.5)
        else:
            f_V_given_I[i] = normpdf(pitch, score_pitch, std)
        sims[i] = similarity(onset_prob,score_onset)

    #plt.plot(f_V_given_I)
    #plt.show()
    
    return f_V_given_I,sims


def compute_f_I_J_given_D(score_axis,estimated_tempo,elapsed_time,beta):
    if estimated_tempo > 0:
       rateRatio = float(Rc) / float(estimated_tempo)
    else:
       rateRatio = Rc/0.00001
    sigmaSquare = math.log(float(1)/float(alpha*elapsed_time )+1)
    sigma = math.sqrt(sigmaSquare)
    tmp1 = 1/(score_axis*sigma*math.sqrt(2*math.pi))
    tmp2 = (np.log(score_axis)-math.log(rateRatio*elapsed_time)+beta*sigmaSquare)
    tmp2 = np.exp(-tmp2*tmp2/(2*sigmaSquare))
    distribution = tmp1* tmp2
    distribution[score_axis <=0] = 0
    distribution = distribution /sum(distribution)
    #plt.plot(distribution)
    #plt.show()
    return distribution


def pitch_detection(data):
    samps = np.fromstring(data,dtype= np.int16)
    pitch = analyse.musical_detect_pitch(samps)
    if analyse.loudness(samps) > -25 and pitch != None:
       return pitch
    else:
       return -1


def tempo_estimate(estimated_tempo,elapsed_time,cur_pos,old_pos): 
    #print 'cur pos %d old pos %d'%(cur_pos,old_pos)
    #print 'elapsed_time %f'%elapsed_time
    return float(cur_pos-old_pos)*Rc*resolution/elapsed_time   



def compute_f_I_given_D(fsource,f_I_J_given_D,cur_pos,scoreLen):
    
    left = max(0,cur_pos-1000)
    right = min(scoreLen,cur_pos+1000)
    f_I_given_D = np.zeros(scoreLen)
    fsource_w = fsource[left:right]
    f_I_J_given_D_w = f_I_J_given_D[:right-left]
    f_I_given_D_w = np.convolve(fsource_w, f_I_J_given_D_w)
    f_I_given_D_w = f_I_given_D_w/sum(f_I_given_D_w)
    if left+len(f_I_given_D_w) > scoreLen:
        end = scoreLen
    else:
        end = left+len(f_I_given_D_w)
    f_I_given_D[left:end]  = f_I_given_D_w[:(end-left)]
    
    #f_I_given_D = np.convolve(fsource, f_I_J_given_D)
    return f_I_given_D


def plot_onsets_prob(onset_audio_file,scoreLen):

    #audio_onsets =plot_score(plot_audio_file = onset_audio_file)
    
    proc = CNNOnsetProcessor()
    audio_onsets = proc(onset_audio_file)
    
    '''
    audio_onsets = np.zeros(scoreLen)
    client = speech.SpeechClient()

    with io.open(onset_audio_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='zh',
        enable_word_time_offsets=True)

    response = client.recognize(config, audio)

    for result in response.results:
        alternative = result.alternatives[0]
        print(u'Transcript: {}'.format(alternative.transcript))

        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            print('start_time: %f, end_time: %f'%(
                start_time.seconds + start_time.nanos * 1e-9,
                end_time.seconds + end_time.nanos * 1e-9))
            onset_center = int((start_time.seconds+start_time.nanos * 1e-9)/0.01)
            if onset_center < scoreLen: 
                audio_onsets[onset_center]=1
    '''
    #plt.plot(audio_onsets)
    #plt.show()
    return audio_onsets

 

def find_next_onset(cur_pos,score_midi,skip):  
    while True:
        if cur_pos+ skip> len(score_midi) -1:
            cur_pos  = len(score_midi)-1
            break
        cur_pos = cur_pos+skip
        #print  'cur_pos  is  -------- %d'%cur_pos
        while score_midi[cur_pos]==0 and cur_pos < len(score_midi)-1:
            cur_pos+=1
        silence_cnt = 0
        total_cnt = 0
        for i in range(cur_pos,cur_pos+200):
            if cur_pos < len(score_midi) and score_midi[cur_pos] ==0:
                silence_cnt+=1
            elif cur_pos < len(score_midi):
                total_cnt +=1
        if float(silence_cnt)/float(total_cnt+1e-5) <=  0.8:
            break

    return cur_pos



def create_gaussian_mask(cur_pos,end_time,resolution):
    axis_end_time = math.floor(end_time/resolution)*resolution
    size = (axis_end_time - 0)/resolution+1
    axis = np.linspace(0,axis_end_time,size)
    if abs(axis_end_time-end_time) > 1e-5:
        axis = np.concatenate((axis, [end_time]))
   
    mean = cur_pos*resolution
    # std can chanage
    std = 0.7
    gaussian_mask = scipy.stats.norm.pdf(axis,mean,std)

    #plt.plot(axis,gaussian_mask, color='black')
    #plt.show()
    return gaussian_mask


def create_gate_mask(cur_pos,scoreLen):
    mask = np.zeros(scoreLen)
   
    for  i in range(-50,51):
        if cur_pos+i < scoreLen and cur_pos+i>=0:
             mask[cur_pos+i] = 1

    #plt.plot(range(scoreLen),mask, color='black')
    #plt.show()
    return mask


def score_follow(audio_file,midi_file ,feature,mask):
    old_midi = pretty_midi.PrettyMIDI(MIDIFILE)
    new_midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)


    score_midi,score_axis,score_onsets,onsets,onset0 = get_time_axis(resolution, score_start_time,score_end_time,midi_file)
    #score_midi = diff(score_midi)
    #plt.plot(score_midi)
    #plt.show()
    onset_idx = 0
    pitches = []
    mapped_onset_times = []
    mapped_onset_notes = []
    mapped_detect_notes = []
    scoreLen = len(score_axis)
    fsource = np.zeros(scoreLen)
    confidence = np.zeros(scoreLen)
    pitchfile = np.zeros(scoreLen)
    fsource[onset0] = 1
    old_pos = 0
    cur_pos = 0
    tempo_estimate_elapsed_time = 0
    estimated_tempo = Rc
    matched_score = []
    detected_pitches = []
    time_axis = []
    mapped_time = []
    tempos = []
    wf = wave.open(AUDIOFILE, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
    old_time = performance_start_time
    cur_time = performance_start_time
    firstReadingFlag = True
    n_frames = int(performance_start_time * wf.getframerate())
    wf.setpos(n_frames)
    data = wf.readframes(CHUNK)
    if feature ==  'onset':
        audio_onsets = plot_onsets_prob(audio_file,scoreLen)
    else:
        audio_onsets = np.zeros(scoreLen)
    #np.savetxt('/Users/rtchen/Downloads/sing_dataset/002/onsets.txt',audio_onsets)
    
    
    #print audio_onsets
    '''
    plt.plot(audio_onsets)
    plt.show()
    '''
    pitches = []
    last_silence_time = 0
    silence_cnt =0
    datas = [data]
    while wf.tell() < wf.getnframes():  
        '''
        if len(datas)>=3:
            c_data = datas[-3:]
            c_data = ''.join(c_data)
        else:
            c_data = data
        '''
        c_data = data
        pitch = pitch_detection(c_data)   
        data = wf.readframes(CHUNK)
        datas.append(data)
        if firstReadingFlag:
            old_time = cur_time
        else:
            tempo_estimate_elapsed_time+=time_int
        cur_time +=time_int

        if cur_time>BREAK_TIME:
            break
        if pitch == -1:
            if cur_time-last_silence_time <= time_int+0.1 and not firstReadingFlag:
                silence_cnt+=1
                last_silence_time = cur_time
            elif not firstReadingFlag:
                silence_cnt = 1
                last_silence_time = cur_time
            if silence_cnt == 35:
                fsource = np.zeros(scoreLen)
                new_pos = find_next_onset(cur_pos,score_midi,100)
                print 'new pos %d'%new_pos
                last_pos = cur_pos
                old_pos = new_pos
                cur_pos = new_pos
                fsource[new_pos] =1
            if silence_cnt >= 35:
                old_time = cur_time
                tempo_estimate_elapsed_time = 0
                #print 'old tempo %f'%tempo
            if silence_cnt ==100 and cur_pos - last_pos<300:
                fsource = np.zeros(scoreLen)
                new_pos = find_next_onset(cur_pos,score_midi,200)
                print 'new pos ------%d'%new_pos
                old_pos = new_pos
                cur_pos = new_pos
                fsource[new_pos] =1
            if silence_cnt ==200 and cur_pos - last_pos<600:
                fsource = np.zeros(scoreLen)
                new_pos = find_next_onset(cur_pos,score_midi,300)
                print 'new pos ------- %d'%new_pos
                old_pos = new_pos
                cur_pos = new_pos
                fsource[new_pos] =1
            continue  
        #pitch  = pitch-int(pitch)/12*12

        print 'detected pitch is %f'%pitch
        #print 'last detected pitch is %f'%last_pitch
        pitches.append(pitch)
        
        detected_pitches.append(-pitch)
        firstReadingFlag = False
        elapsed_time = cur_time - old_time
        tempo = estimated_tempo
        if tempo*tempo_estimate_elapsed_time >2*Rc:
            tempo=tempo_estimate(estimated_tempo,tempo_estimate_elapsed_time,cur_pos,old_pos)
            # 0.9 或 1.1
            if tempo/float(Rc) < 0.7:
                tempo = Rc*0.7
            elif tempo/float(Rc) > 1.3:
                tempo = Rc*1.3
            tempo_estimate_elapsed_time = 0
            estimated_tempo = tempo     
            old_pos = cur_pos     
        print 'tempo %f'%tempo
        
        if int((cur_time)/0.01) < len(audio_onsets):
            onset_prob = audio_onsets[int((cur_time)/0.01)]
        else:
            onset_prob = 0



        if fsource[cur_pos]>0.1:
            beta = 0.5
            w1 = 0.95
            w2 = 0.05
            w3 = 0.5
            print '-----------------------------'
        else:
            beta = 0.5
            w1 = 0.8
            w2 = 0.2
            w3 = 0.3
            
        '''
        beta = 0.5
        w1 = 0.95
        w2 = 0.05
        w3 = 0.7
        '''
        
        f_I_J_given_D = compute_f_I_J_given_D(score_axis,tempo,elapsed_time,beta)
        f_I_given_D = compute_f_I_given_D(fsource,f_I_J_given_D,cur_pos,scoreLen)   
        cur_pos = np.argmax(f_I_given_D)

        '''
        if similarity(pitch,score_midi[cur_pos]) < 0.9:
            alpha = 0.01
        else:
            alpha = 1
        '''
        #alpha = 1
        
        f_V_given_I,sims = compute_f_V_given_I(pitch,pitches,scoreLen,score_midi,onset_prob,score_onsets,alpha,feature,w1,w2,w3)
        #plt.plot(f_I_given_D[:scoreLen])
        #plt.show()
        print 'elapsed_time %f'%elapsed_time
        fsource = f_V_given_I*f_I_given_D[:scoreLen]
        if mask == 'gaussian':
            gaussian_mask = create_gaussian_mask(cur_pos,score_end_time,resolution)
            fsource  = fsource * gaussian_mask

        elif mask == 'gate':
            gate_mask = create_gate_mask(cur_pos,scoreLen)
            fsource  = fsource * gate_mask
            

        fsource = fsource/sum(fsource)

        cur_pos = np.argmax(fsource)
        
        if fsource[cur_pos] > confidence[int(cur_time/resolution)]:
            confidence[int(cur_time/resolution)] = fsource[cur_pos]
        
        pitchfile[int(cur_time/resolution)] = pitch
        old_time = cur_time
        
        
        
        old_idx = onset_idx
        while onset_idx < len(onsets) and cur_pos >= onsets[onset_idx]:
            onset_idx +=1
        '''
        if abs(onset_idx-old_idx)>1  and cur_pos>1 and feature == 'onset':
            if normpdf(pitch,score_midi[cur_pos]-0.5*score_midi[cur_pos-1],std) < 0.5:
                print 'here at %f compute onset only'%cur_time
                #plt.plot(fsource)
                #plt.show()
                f_V_given_I,sims = compute_f_V_given_I(pitch,pitches,scoreLen,score_midi,onset_prob,score_onsets,alpha,'only')            
                fsource = f_V_given_I*f_I_given_D[:scoreLen]
                fsource = fsource/sum(fsource)
                cur_pos = np.argmax(fsource)
                onset_idx = old_idx
                while onset_idx < len(onsets) and cur_pos >= onsets[onset_idx]:
                    onset_idx +=1
        '''

        if old_idx < onset_idx:
            #for i in range(old_idx,onset_idx):
            old_note = old_midi.instruments[0].notes[onset_idx-1]
            dur = old_note.end-old_note.start
            new_note = pretty_midi.Note(velocity=old_note.velocity, pitch=old_note.pitch, start=cur_time, end=cur_time+dur)
            piano.notes.append(new_note)

        for i in range(len(onsets)):
            if cur_pos < onsets[i]:
                 break
            old_idx = i-1
            
        
        print 'currently at %d'%cur_pos
        print 'cur pitch %f'%score_midi[cur_pos]
        print 'cur_time %f'%cur_time
        #print 'score is %d'%score_midi[cur_pos]
        #print 'onset prob %f'%onset_prob
        '''
        if onset_idx < len(onsets) and cur_pos >= onsets[onset_idx]:
            onset_idx +=1
            mapped_onset_times.append(cur_time-performance_start_time)
            mapped_onset_notes.append(cur_pos*resolution)
            mapped_detect_notes.append(-math.pow(2,float(pitch)/30)+2.5)
        
        plt.clf()
        
        plt.plot(score_axis,fsource)

        if onset_idx < len(onsets) and cur_pos >= onsets[onset_idx]:
            onset_idx +=1
            mapped_onset_times.append(cur_time-performance_start_time)
            mapped_onset_notes.append(cur_pos*resolution)
            mapped_detect_notes.append(-float(pitch)/10)
            plt.title('time {} onset {} score{}'.format(cur_time,onset_idx,score_midi[cur_pos]))
            plt.xlabel('score_axis')
            plt.ylabel('score')
            plt.draw()
            if onset_idx != 1:
                plt.pause(10)
            else:
                plt.pause(0.5)
        else:
            plt.title('time {}'.format(cur_time))
            plt.xlabel('score_axis')
            plt.ylabel('score')
            plt.draw()
            plt.pause(0.3)
        '''
        matched_score.append(score_midi[cur_pos])
        time_axis.append(cur_time-performance_start_time)
        mapped_time.append(cur_pos*resolution)
        tempos.append(tempo)
    #p.terminate()
    #plt.plot(pitches)
    #plt.show()
    new_midi.instruments.append(piano)
    new_midi.write(NEWFILE)
    np.savetxt(CONF_FILE,confidence)
    #---------------------------------------------------------------------------------------
    #plt.plot(pitchfile,'o',markersize = 1)
    np.savetxt(PITCHFILE,pitchfile)
    score_plt = [min(-x/5,0) for x in score_midi]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position(('data',0))
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    s,=plt.plot(score_plt,score_axis,color = 'r')
    dp,=plt.plot(time_axis,detected_pitches,'o',color = 'g',label = 'detected pitch',markersize = 1)
    mp,=plt.plot(time_axis,mapped_time,'o',color = 'b',label = 'mapped position',markersize = 1)
    '''
    for i in range(len(mapped_onset_notes)):
        score_pos = int(mapped_onset_notes[i]/resolution)
        plt.scatter(x=score_plt[score_pos],y=mapped_onset_notes[i],c = 'b',s = 10,marker = 'o')
        plt.plot([0,score_plt[score_pos]],[mapped_onset_notes[i],mapped_onset_notes[i]],linestyle='-.',color = 'b',linewidth = 0.75)
        plt.plot([mapped_onset_times[i],mapped_onset_times[i]],[0,mapped_onset_notes[i]],linestyle = '-.',color = 'b',linewidth= 0.75)
        plt.plot([0,mapped_onset_times[i]],[mapped_onset_notes[i],mapped_onset_notes[i]],linestyle = '-.',color = 'b',linewidth = 0.75)
        plt.plot([mapped_onset_times[i],mapped_onset_times[i]],[0,mapped_detect_notes[i]],linestyle= '-.',color = 'g',linewidth = 0.75)
    
    for i in range(len(ground_truths)):
        score = score_midi[onsets[i]]
        score = min(float(-x),5)/3-1         
        plt.plot([0,score],[onsets[i]*resolution,onsets[i]*resolution],linestyle='-.',color = 'k',linewidth = 0.4)
        plt.plot([0,ground_truths[i]],[onsets[i]*resolution,onsets[i]*resolution],linestyle = '-.',color = 'k',linewidth = 0.4)
        if i < len(mapped_detect_notes):
            plt.plot([ground_truths[i],ground_truths[i]],[0,mapped_detect_notes[i]],linestyle= '-.',color = 'k',linewidth = 1)
        plt.plot([ground_truths[i],ground_truths[i]],[0,onsets[i]*resolution],linestyle= '-.',color = 'k',linewidth = 0.75  )
    '''
    plt.xlabel('audio time (seconds)')
    plt.ylabel('score time (seconds)')
    plt.grid()
    plt.show()
    


        

if __name__== "__main__" :
    score_follow(audio_file = AUDIOFILE,midi_file = MIDIFILE,feature = 'onset',mask =FILTER)



