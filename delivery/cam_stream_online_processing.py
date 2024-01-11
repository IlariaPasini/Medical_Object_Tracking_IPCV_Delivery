#------------------------------------------------------------------------------
# This script receives encoded video from the HoloLens cameras and plays it.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import multiprocessing as mp
import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import output.Project.segmentation as sgt

# Settings --------------------------------------------------------------------

# HoloLens address
host = '169.254.50.241' #'169.254.58.146' 

# Ports
ports = [
    hl2ss.StreamPort.RM_VLC_LEFTFRONT,
    #hl2ss.StreamPort.RM_VLC_LEFTLEFT,
    hl2ss.StreamPort.RM_VLC_RIGHTFRONT,
    #hl2ss.StreamPort.RM_VLC_RIGHTRIGHT,
    #hl2ss.StreamPort.RM_DEPTH_AHAT,
    hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
    hl2ss.StreamPort.PERSONAL_VIDEO,
    #hl2ss.StreamPort.RM_IMU_ACCELEROMETER,
    #hl2ss.StreamPort.RM_IMU_GYROSCOPE,
    #hl2ss.StreamPort.RM_IMU_MAGNETOMETER,
    #hl2ss.StreamPort.MICROPHONE,
    #hl2ss.StreamPort.SPATIAL_INPUT,
    #hl2ss.StreamPort.EXTENDED_EYE_TRACKER,
    ]

# PV parameters
pv_width     = 760
pv_height    = 428
pv_framerate = 30

#LT parameters
lt_width     = 320
lt_height    = 288
lt_framerate = 5


# Maximum number of frames in buffer
buffer_elements = 150

#------------------------------------------------------------------------------

if __name__ == '__main__':
    if ((hl2ss.StreamPort.PERSONAL_VIDEO in ports) and (hl2ss.StreamPort.RM_DEPTH_AHAT in ports)):
        print('Error: Simultaneous PV and RM Depth AHAT streaming is not supported. See known issues at https://github.com/jdibenes/hl2ss.')
        quit()

    if ((hl2ss.StreamPort.RM_DEPTH_LONGTHROW in ports) and (hl2ss.StreamPort.RM_DEPTH_AHAT in ports)):
        print('Error: Simultaneous RM Depth Long Throw and RM Depth AHAT streaming is not supported. See known issues at https://github.com/jdibenes/hl2ss.')
        quit()

    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Start PV Subsystem if PV is selected ------------------------------------
    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start streams -----------------------------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.RM_VLC_LEFTFRONT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_LEFTFRONT))
    producer.configure(hl2ss.StreamPort.RM_VLC_LEFTLEFT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_LEFTLEFT))
    producer.configure(hl2ss.StreamPort.RM_VLC_RIGHTFRONT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_RIGHTFRONT))
    producer.configure(hl2ss.StreamPort.RM_VLC_RIGHTRIGHT, hl2ss_lnm.rx_rm_vlc(host, hl2ss.StreamPort.RM_VLC_RIGHTRIGHT))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate))
    producer.configure(hl2ss.StreamPort.RM_IMU_ACCELEROMETER, hl2ss_lnm.rx_rm_imu(host, hl2ss.StreamPort.RM_IMU_ACCELEROMETER))
    producer.configure(hl2ss.StreamPort.RM_IMU_GYROSCOPE, hl2ss_lnm.rx_rm_imu(host, hl2ss.StreamPort.RM_IMU_GYROSCOPE))
    producer.configure(hl2ss.StreamPort.RM_IMU_MAGNETOMETER, hl2ss_lnm.rx_rm_imu(host, hl2ss.StreamPort.RM_IMU_MAGNETOMETER))
    producer.configure(hl2ss.StreamPort.MICROPHONE, hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE))
    producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT))
    producer.configure(hl2ss.StreamPort.EXTENDED_EYE_TRACKER, hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER))

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sinks = {}

    for port in ports:
        producer.initialize(port, buffer_elements)
        producer.start(port)
        sinks[port] = consumer.create_sink(producer, port, manager, None)
        sinks[port].get_attach_response()
        while (sinks[port].get_buffered_frame(0)[0] != 0):
            pass
        print(f'Started {port}')        
        
    # Create Display Map ------------------------------------------------------
    def display_pv(port, payload):
        if (payload.image is not None and payload.image.size > 0):
            cv2.imshow(hl2ss.get_port_name(port), payload.image)

    def display_basic(port, payload):
        if (payload is not None and payload.size > 0):
            cv2.imshow(hl2ss.get_port_name(port), payload)

    def display_depth_lt(port, payload):
        cv2.imshow(hl2ss.get_port_name(port) + '-depth', payload.depth * 8) # Scaled for visibility
        cv2.imshow(hl2ss.get_port_name(port) + '-ab', payload.ab)

    def display_depth_ahat(port, payload):
        if (payload.depth is not None and payload.depth.size > 0):
            cv2.imshow(hl2ss.get_port_name(port) + '-depth', payload.depth * 64) # Scaled for visibility
        if (payload.ab is not None and payload.ab.size > 0):
            cv2.imshow(hl2ss.get_port_name(port) + '-ab', payload.ab)

    def display_null(port, payload):
        pass

    DISPLAY_MAP = {
        hl2ss.StreamPort.RM_VLC_LEFTFRONT     : display_basic,
        hl2ss.StreamPort.RM_VLC_LEFTLEFT      : display_basic,
        hl2ss.StreamPort.RM_VLC_RIGHTFRONT    : display_basic,
        hl2ss.StreamPort.RM_VLC_RIGHTRIGHT    : display_basic,
        hl2ss.StreamPort.RM_DEPTH_AHAT        : display_depth_ahat,
        hl2ss.StreamPort.RM_DEPTH_LONGTHROW   : display_depth_lt,
        hl2ss.StreamPort.PERSONAL_VIDEO       : display_pv,
        hl2ss.StreamPort.RM_IMU_ACCELEROMETER : display_null,
        hl2ss.StreamPort.RM_IMU_GYROSCOPE     : display_null,
        hl2ss.StreamPort.RM_IMU_MAGNETOMETER  : display_null,
        hl2ss.StreamPort.MICROPHONE           : display_null,
        hl2ss.StreamPort.SPATIAL_INPUT        : display_null,
        hl2ss.StreamPort.EXTENDED_EYE_TRACKER : display_null,
    }

    # Store -------------------------------------------------------------------
    #cambiare path se serve
    pv_path = 'C:/Users/marti/Desktop/Poli/Image Processing and Computer Vision/Progetto/hl2ss-computer-vision-class/viewer/output/stereo/alongZ/pv/'
    lf_path = 'C:/Users/marti/Desktop/Poli/Image Processing and Computer Vision/Progetto/hl2ss-computer-vision-class/viewer/output/stereo/alongZ/lf/'
    rf_path = 'C:/Users/marti/Desktop/Poli/Image Processing and Computer Vision/Progetto/hl2ss-computer-vision-class/viewer/output/stereo/alongZ/rf/'
    lt_path = 'C:/Users/marti/Desktop/Poli/Image Processing and Computer Vision/Progetto/hl2ss-computer-vision-class/viewer/output/stereo/alongZ/lt/'

    def store_pv(port, payload, c):
        if (payload.image is not None and payload.image.size > 0):
            # store pv image
            filename = f"pv_frame{c}.png"
            cv2.imwrite(pv_path + filename, payload.image) 
            pass

    def store_lf(port, payload, c):
        if (payload is not None and payload.size > 0):
            # store rf & lf image
            filename = f"lf_frame{c}.png"
            cv2.imwrite(lf_path + filename, payload) 
            pass

    def store_rf(port, payload, c):
        if (payload is not None and payload.size > 0):
            # store rf & lf image
            filename = f"rf_frame{c}.png"
            cv2.imwrite(rf_path + filename, payload) 
            pass
        
    def store_lt(port, payload, c):
        if (payload.depth is not None and payload.depth.size > 0):
            # store rf & lf image
            filename = f"lt_frame{c}.png"
            cv2.imwrite(lt_path + filename, payload.depth * 8) 
            pass

    STORE_MAP = {
        hl2ss.StreamPort.RM_VLC_LEFTFRONT     : store_lf,
        hl2ss.StreamPort.RM_VLC_RIGHTFRONT    : store_rf,
        hl2ss.StreamPort.PERSONAL_VIDEO       : store_pv,
        hl2ss.StreamPort.RM_DEPTH_LONGTHROW   : store_lt
    }

    def cv_lf(port, payload):
        pass

    def cv_rf(port, payload):
        pass

    def cv_pv(port, payload):
        if (payload.image is not None and payload.image.size > 0):
            res = sgt.Blob.FindCirclesFine(payload.image, applyMorph=True, blobMethod = sgt.Blob.Config.SIMPLE_BLOB)
            cv2.imshow(hl2ss.get_port_name(port), res)
            
    def cv_lt(port, payload):
        #displays long throw depth
        if (payload.depth is not None and payload.depth.size > 0):
            cv2.imshow(hl2ss.get_port_name(port) + '-depth', payload.depth * 8) # Scaled for visibility
            
    def cv_ah(port, payload):
        if (payload.depth is not None and payload.depth.size > 0):
            cv2.imshow(hl2ss.get_port_name(port) + '-depth', payload.depth * 64) # Scaled for visibility
        #if (payload.ab is not None and payload.ab.size > 0):
        #   cv2.imshow(hl2ss.get_port_name(port) + '-ab', payload.ab)

    CV_MAP = {
        hl2ss.StreamPort.RM_VLC_LEFTFRONT     : cv_lf,
        hl2ss.StreamPort.RM_VLC_RIGHTFRONT    : cv_rf,
        hl2ss.StreamPort.PERSONAL_VIDEO       : cv_pv,
        hl2ss.StreamPort.RM_DEPTH_LONGTHROW   : cv_lt,
        hl2ss.StreamPort.RM_DEPTH_AHAT        : cv_ah
        
    }

    # Main loop ---------------------------------------------------------------
    counter = 0
    while (enable):
        for port in ports:
            _, data = sinks[port].get_most_recent_frame()
            if (data is not None):
                DISPLAY_MAP[port](port, data.payload)
                #CV_MAP[port](port, data.payload)
                STORE_MAP[port](port, data.payload,counter)
        counter += 1
        cv2.waitKey(1)

    # Stop streams ------------------------------------------------------------
    for port in ports:
        sinks[port].detach()
        producer.stop(port)
        print(f'Stopped {port}')

    # Stop PV Subsystem if PV is selected -------------------------------------
    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Stop keyboard events ----------------------------------------------------
    listener.join()
