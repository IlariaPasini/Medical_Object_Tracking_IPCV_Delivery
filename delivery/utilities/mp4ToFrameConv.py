import cv2

class mp4Converter:

    def mp4ToPng(videoPath, framesPath):    
        #videoPath = 'Colored_Hololens.mp4'
        vidcap = cv2.VideoCapture(videoPath)
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(f"{framesPath}/frame%d.png" % count, image)
            success, image = vidcap.read()
            count+=1