import cv2

#Simple utility class to convert mp4 to png frames. This is useful for converting the video stream we get when using Windows Device Portal 
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