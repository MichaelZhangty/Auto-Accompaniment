# Auto-Accompaniment
File:
     score_following_onset_diff_old.py
Function:
     It compares audio.wav with midi.mid to follow the position of the singers' singing.
     It generate time list according to each beat in file time_beat_file.txt
     It also saves the confidence queue in confidence_queue_file.txt
     
File:
     auto_accompany.py
function:
     It reads time_beat_file.txt and confidence_queue_file.txt.
     It adjust the speed of the midi.mid to generate the final version of our accompanyment midi according to     time_beat_file.txt and confidence_queue_file.txt.
     
File: 
score_following_onset_diff_DTW.py
score_following_onset_diff_normal.py

function:
     They are improved version of score_following_onset_diff_.py.
     
Usage:
     python2 score_following_onset_diff_old.py
     python2 auto_accompany.py
   
 
