<h1>  OOPproject4 </h1>
<strong> This is for OOP class, project4 </strong>

<h5> Our topic is to create a parking lot management system through car plate recognition. We work with Python, using the openCV library.
<br> It will be helpful if you already know about image processing.</h5>
<hr>
1. you need to install openCV, pytesseract library and just connect with our code
(just follow your compiler and google says 🤞)

2. we're working on it, maybe many things will be changed... please do not start yet !!! 😂😂

3. You can visit the official docs for openCV https://docs.opencv.org/4.x/index.html 

4. you can change this path to on your own! (if you need)
![image](https://user-images.githubusercontent.com/102032766/204998731-b5ab185a-a316-49bf-8cbf-1ecd8fade004.png)

<br><br>
<hr>
<br><br>

to execute program:
1. download cv2, matplotlib, pytesseract, pillow(perhaps there is more.. plz read error message)
2. 'path' in oopproject4.py : directory where py files & car images are(has to be in same directory)
3. in line 'pytesseract.pytesseract.tesseract_cmd =' in findcarinfo.py you should enter your tesseract file path
4. you should download trainded data from here: https://github.com/tesseract-ocr/tessdata/blob/main/kor.traineddata
    move this trained data file to tessdata directory in tesseract directory
5. the image file name format should be 0000-00-00 00;00;00 to correctly read the date and time info; there are examples on 
6. image result in 'res' directory under path

<br><br>
<hr>
<br><br>

 <h2> Problem1.</h2>

![image](https://user-images.githubusercontent.com/102032766/205343818-432b2639-eda7-44a1-b452-975ce3d7fcb2.png)
<br>
Can you see the image's name? <br> Some name are not correct. <br>
At now, some image can't be read well, so we need to make exepction case... <br> (like at first, it starts 2 numbers and next 1 한글 and next, 4 numbers.. we give it some rules)
