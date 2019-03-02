# Drowsiness_Detection
This system will help people who go for long drive to detect their drowsiness.

* [Research Paper Reference](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)

* ![GitHub](https://img.shields.io/github/license/ghrahul/Drowsiness_Detection.svg)

# Face and Eye detection in a static image

* Run 
```
python face_eye_detect_static_image.py
```
![face_eye_detect](https://user-images.githubusercontent.com/22416933/53569160-89cd4400-3b89-11e9-912e-a63d50cf1f90.jpg)


# Face and Eye detection in a real time using haarcascade

* Run 
```
python realtime_face_eye_detection.py
```
![20190228_195850](https://user-images.githubusercontent.com/22416933/53573576-c0a85780-3b93-11e9-8d09-3a214f527bac.gif)

* In this application we will check the eye aspect ratio.If this ratio is less than the threshold for a good amount of time application will trigger the alarm

![blink_detection_plot](https://user-images.githubusercontent.com/22416933/53685610-fa6b9080-3d42-11e9-8aa0-8aba4f5de0b4.jpg)

* The facial landmarks detected by dlib are indexible.
<br>
* Visualization of 68 facial landmark coordinates from the iBUG 300-W dataset

![facial_landmarks_68markup-768x619](https://user-images.githubusercontent.com/22416933/53685652-94333d80-3d43-11e9-8249-570bc87de58b.jpg)

# Detection of drowsiness

![20190302_234711](https://user-images.githubusercontent.com/22416933/53685846-145aa280-3d46-11e9-9349-aebf2d1a8348.gif)

        




