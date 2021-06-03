import numpy as np
import cv2 as cv
import time
import json
import random as r

btn_scc, btn_err, btn_nxt = "success.jpg", "tabu.jpg", "next.jpg"
btn_s = cv.imread(btn_scc)
btn_sr = cv.resize(btn_s, (150, 150))
btn_e = cv.imread(btn_err)
btn_er = cv.resize(btn_e, (150, 150))
btn_n = cv.imread(btn_nxt)
btn_nr = cv.resize(btn_n, (150, 150))

def getWord(color, shape):
    filename = "data.json"
    with open(filename) as f:
        data = json.load(f)
        rndIndx = r.randint(0, 2)
        clr = data[color]
        shp = clr[shape]
        res = shp[rndIndx]
        return res['word'], res['forbidden1'], res['forbidden2'], res['forbidden3'], res['forbidden4']

def getColor(bm, ym, gm):
    area_min_limit = 11000
    area_max_limit = 200000
    
    ret , tresh = cv.threshold(bm, 200, 255, cv.CHAIN_APPROX_NONE)
    contoursBlue , hierarchy = cv.findContours(tresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contourBlue in contoursBlue:
        c_area = cv.contourArea(contourBlue)
        if c_area >= area_min_limit and c_area <= area_max_limit:
            #approx = cv.approxPolyDP(contourBlue, 0.01* cv.arcLength(contourBlue, True), True)
            return 'Blue', contoursBlue

    ret , tresh = cv.threshold(ym, 200, 255, cv.CHAIN_APPROX_NONE)
    contoursYellow, hierarchy = cv.findContours(tresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contourYellow in contoursYellow:
        c_area = cv.contourArea(contourYellow)
        if c_area >= area_min_limit and c_area <= area_max_limit:
            #approx = cv.approxPolyDP(contourYellow, 0.01* cv.arcLength(contourYellow, True), True)
            return 'Yellow', contoursYellow

    ret , tresh = cv.threshold(gm, 200, 255, cv.CHAIN_APPROX_NONE)
    contoursGreen , hierarchy = cv.findContours(tresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contourGreen in contoursGreen:
        c_area = cv.contourArea(contourGreen)
        if c_area >= area_min_limit and c_area <= area_max_limit:
            #approx = cv.approxPolyDP(contourGreen, 0.01* cv.arcLength(contourGreen, True), True)
            return 'Green', contoursGreen

    return None, None

def getShape(frame, contours):
    area_min_limit = 11000
    area_max_limit = 200000
    
    if contours is not None:
        for contour in contours:
            c_area = cv.contourArea(contour)
            if c_area >= area_min_limit and c_area <= area_max_limit:
                approx = cv.approxPolyDP(contour, 0.01* cv.arcLength(contour, True), True)
                cv.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                x = approx.ravel()[0]
                y = approx.ravel()[1] - 5
                if len(approx) == 3:
                    cv.putText(frame, "Triangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    return 'Triangle'
                elif len(approx) == 4:
                    x1 ,y1, w, h = cv.boundingRect(approx)
                    aspectRatio = float(w)/h
                    if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                        cv.putText(frame, "square", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                        return 'Square'
                    else:
                        cv.putText(frame, "rectangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                        return 'Square'
                else:
                    cv.putText(frame, "Circle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                    return 'Circle'

    return None
    

def Game():
    # Game environment variables.
    cap = cv.VideoCapture(0)
    frame_tw = np.zeros((480, 1280, 3), np.uint8)

    total_points = 000

    isTimeTaken = False
    firstObjectTime = 0
    isTimer = False
    detectedShape = None
    isNeedToDetect = True

    sqrCntr = 0
    trgCntr = 0
    crcCntr = 0
    while(True):
        ret, frame = cap.read()
        

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        bl = np.array([72,158,134],np.uint8)
        bu = np.array([120,255,243],np.uint8)

        yl = np.array([0,181,0],np.uint8)
        yu = np.array([76,255,255],np.uint8)

        gl = np.array([33,88,100],np.uint8)
        gu = np.array([76,208,204],np.uint8)
         
        bmask = cv.inRange(hsv, bl, bu)
        ymask = cv.inRange(hsv, yl, yu)
        gmask = cv.inRange(hsv, gl, gu)

        col, con = getColor(bmask, ymask, gmask)
        shape = getShape(frame, con)
        
        if isNeedToDetect:
            if col is not None and shape is not None:
                #print(str(col) + ' - ' + str(shape))
                isTimer = True
                if not isTimeTaken:
                    firstObjectTime = time.time()
                    isTimeTaken = True
                cv.rectangle(frame_tw,(820,0),(1090,30),(0,0,0),-1)
            else:
                #print('No Shape Found!')
                cv.rectangle(frame_tw,(820,0),(1090,30),(50,50,50),-1)
                cv.putText(frame_tw, "No Game Card Found!", (820, 25), cv.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 1)
                isTimer = False
                isTimeTaken = False
                cv.rectangle(frame_tw,(820,65),(1090,120),(0,0,0),-1)

            if isTimer:
                if time.time() - firstObjectTime >= 1.0:
                    clrs = {'Blue':(255,0,0), 'Yellow':(0,255,255), 'Green':(0,255,0)}
                    vals = [sqrCntr, trgCntr, crcCntr]
                    shps = ['Square', 'Triangle', 'Circle']
                    detectedShape = shps[np.argmax(vals)]
                    
                    # handle what will happend when object is detected.
                    if col == 'Blue':
                        cv.putText(frame_tw, detectedShape, (910, 85), cv.FONT_HERSHEY_DUPLEX, 0.75, clrs['Blue'], 1)
                    if col == 'Yellow':
                        cv.putText(frame_tw, detectedShape, (910, 85), cv.FONT_HERSHEY_DUPLEX, 0.75, clrs['Yellow'], 1)
                    if col == 'Green':
                        cv.putText(frame_tw, detectedShape, (910, 85), cv.FONT_HERSHEY_DUPLEX, 0.75, clrs['Green'], 1)

                    # clear Panel.
                    cv.rectangle(frame_tw,(750,130),(1230,260),(0,255,0),-1)

                    # print word infos.
                    w, f1, f2, f3, f4 = getWord(col, detectedShape)
                    cv.putText(frame_tw, w, (910, 155), cv.FONT_HERSHEY_DUPLEX, 0.75, (0,235,0), 2)
                    cv.putText(frame_tw, f1, (910, 178), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1)
                    cv.putText(frame_tw, f2, (910, 198), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1)
                    cv.putText(frame_tw, f3, (910, 218), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1)
                    cv.putText(frame_tw, f4, (910, 238), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1)

                    # reset variables which are used to detect card.
                    isTimeTaken = False
                    firstObjectTime = 0
                    isTimer = False
                    isObjectDetected = False
                    detectedShape = None
                    sqrCntr = 0
                    trgCntr = 0
                    crcCntr = 0
                    isNeedToDetect = False
                    
                else:
                    if shape == 'Square':
                        sqrCntr += 1
                    if shape == 'Triangle':
                        trgCntr += 1
                    if shape == 'Circle':
                        crcCntr += 1        

        frame_tw[0:480, 0:640] = frame
        cv.imshow('Frame', frame_tw)
        key = cv.waitKey(1)
        if key == ord('a'):
            cv.rectangle(frame_tw,(820,45),(1090,75),(0,0,0),-1)
            isNeedToDetect = True
            total_points -= 10
        elif key == ord('s'):
            cv.rectangle(frame_tw,(820,45),(1090,75),(0,0,0),-1)
            isNeedToDetect = True
        elif key == ord('d'):
            cv.rectangle(frame_tw,(820,45),(1090,75),(0,0,0),-1)
            isNeedToDetect = True
            total_points += 10

        # print score on screen.
        cv.putText(frame_tw, "Score: " + str(total_points), (910, 60), cv.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1)
        # put btns on screen.
        frame_tw[300:450, 800-75:950-75] = btn_er
        frame_tw[300:450, 960-75:1110-75] = btn_nr
        frame_tw[300:450, 1120-75:1270-75] = btn_sr
        
    # When everything done, release the capture
    cap.release()

Game()