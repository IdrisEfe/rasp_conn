import cv2
import numpy as np
import serial
import time
from math import tan, radians

"""
FoV_dikey = 27 derece
FoV_yatay = 48 derece
FoV_diyagonal = 55 derce
s_diyagonal = 0.0375 derce/piksel 
s_dikey = 0.0375 derece/piksel
s_yatay = 0.0375 derce/piksel
"""

feedback = ""
dataSent = 0
dataGot = 0
orgRenk = ""
renk = ""
start = False

def getDistance(y):

    h = 26.5 #25
    teta = 75 # 55 ama 75 yapılabilir 
    FoV_dikey = 27
    #s_dikey = 0.0375

    y_height = frame_height - y
    center_y = frame_height / 2
    distanceToCenter_pixel = center_y - y_height
    fi = (distanceToCenter_pixel / frame_height) * FoV_dikey
    #fi = distanceToCenter_pixel * s_dikey

    a = teta + fi
    print(fi)
    print(a)
    print(tan(radians(a)))
    x = h * tan(radians(a)) / 2

    return x

def isCircle(area, perimeter):

    circularity = 0
    shape_name = ""
    isCircle = False

    if perimeter == 0:
        circularity = 0
    else:
        circularity = 4 * np.pi * (area / (perimeter ** 2))

    if 0.85 <= circularity <= 1.15:
        shape_name = "Daire"
    
    if shape_name == "Daire":
        isCircle = True

    return isCircle

def are_images_identical(img1, img2):
    if img1.shape != img2.shape:
        return False
    
    difference = cv2.subtract(img1, img2)
    b, g, r = cv2.split(difference)

    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        return True
    else:
        return False
        
def findAim(x):
    gap_size = frame_width / 3
    if x > frame_width - gap_size :
        return "Sağda"
    elif x < gap_size:
        return "Solda"
    else:
        return "Ortada"
    
def findAimDetailed(x):
    first_gap = frame_width * 2/5
    second_gap = first_gap + frame_width * 1/5

    if x < first_gap:
        return "Solda"
    elif x < second_gap:
        return "Ortada"
    else:
        return "Sagda"

def find_middle_pixel(contour, x=0, y=0):

    if (x == 0 and y == 0) or (x != 0 and y != 0):
        raise TypeError
    
    elif x!=0:
        i=0
        target_ = x

    elif y!=0:
        i=1
        target_ = y
    
    points = contour.resahpe(-1, 2) # 2 li böler ve diğer boyutu kendisi belirlemesini ister [len(liste) / param]

    filtered_points = [point for point in points if point[i] == target_]

    if not filtered_points:
        print(f"Belirtilen kordinatlı nokta bulunmamaktadır")
        return None
    
    filtered_points.sort(key=lambda p: p[0])
    middle_index = len(filtered_points) // 2
    middle_pixel = filtered_points[middle_index]

    return middle_pixel

def write(a):

    text = a+'\n'
    text = text.encode('utf-8')
    ser.write(text)
    dataSent += 1

def asyncron():

    global feedback, orgRenk, renk
    while True:
        if ser.in_waiting:
            feedback = ser.readline().decode('utf-8').strip()
            dataGot += 1
            if dataGot == 2:
                orgRenk = feedback
                renk = orgRenk
            print("Arduino:", feedback)
            
        
        if feedback.lower() == "end":
            break

port = "/dev/ttyACM0"
ser = serial.Serial(port, 9600, timeout=1)

asyncron()


# Kamera aç
cap = cv2.VideoCapture(2)

# Kamera yükseklik ve genişliğini değiştirme

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Normalde 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Normalde 480

# Kamera yükseklik ve genişlik bulma
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Yükselik: {frame_height}, Genişkil: {frame_width}")

# Kırmızı renk için HSV sınırları
lower_red1 = np.array([0, 130, 30])     # İlk kırmızı tonu alt sınırı
upper_red1 = np.array([10, 255, 255])   # İlk kırmızı tonu üst sınırı
lower_red2 = np.array([170, 130, 30])   # İkinci kırmızı tonu alt sınırı
upper_red2 = np.array([180, 255, 255])  # İkinci kırmızı tonu üst sınırı

# Mavi renk için HSV sınırları
lower_blue = np.array([100, 150, 50])   # Mavi alt sınırı
upper_blue = np.array([140, 255, 255])  # Mavi üst sınırı

lower_green = np.array([45, 80, 35])
upper_green = np.array([75, 255, 200])

frame_prev = None
frame_org_prev = None
mesafeler_pixel_prev = None

while True:
    mesafeler_pixel = []
    _, org_frame = cap.read()

    if ser.in_waiting:
        feedback = ser.readline().decode('utf-8').strip()

    if feedback == "start":
        start = True

    elif feedback == "stop":
        start = False

    if start:
        
        """if frame_org_prev is not None and are_images_identical(org_frame, frame_org_prev):
            frame = frame_prev.copy()
            mesafe_pixel = mesafeler_pixel_prev.copy()"""

        

        frame = org_frame.copy()

        # Görüntüyü HSV renk uzayına çevir
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Kırmızı maskesi oluştur
        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        # Mavi maskesi oluştur
        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # Yeşil maskesi oluştur
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # Morfolojik işlemler (Hem kırmızı hem mavi hem de yeşil için)
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        red_mask = cv2.erode(red_mask, kernel, iterations=1)

        blue_mask = cv2.dilate(blue_mask, kernel, iterations=2)
        blue_mask = cv2.erode(blue_mask, kernel, iterations=1)

        green_mask = cv2.dilate(green_mask, kernel, iterations=2)
        green_mask = cv2.erode(green_mask, kernel, iterations=1)

        # Konturları bul ve işleme
        masks = [("Red", red_mask, (0, 0, 255)), ("Blue", blue_mask, (255, 0, 0)), ("Green", green_mask, (0, 255, 0))]

        for color_name, mask, contour_color in masks:
            if color_name.lower() == renk:

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for i, contour in enumerate(contours):
                    # Konturun alanını hesapla
                    area = cv2.contourArea(contour)
                    if area < 800:  # Çok küçük alanları yok say
                        continue

                    # Sınır kutusu ve merkez noktası
                    x, y, w, h = cv2.boundingRect(contour)
                    y_alt = y + h
                    mesafe_pixel = frame_height - y_alt
                    perimeter = cv2.arcLength(contour, True)
                    M = cv2.moments(contour)
                    if M['m00'] != 0: # Merkez
                        cx = int(M['m10'] / M['m00']) # x ekseni ağırlık merkezi
                        cy = int(M['m01'] / M['m00']) # y ekseni ağırlık merkezi
                    else:
                        cx, cy = x + w // 2, y + h // 2

                    # Konturu şekil olarak algıla
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    if color_name == "Green" or color_name == "Red":
                        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
                    let = True
                    if len(approx) == 3:
                        shape = "Ucgen"
                    elif len(approx) == 4:
                        # Dikdörtgen mi yoksa kare mi kontrol et
                        aspect_ratio = float(w) / h
                        if 0.95 <= aspect_ratio <= 1.05:
                            shape = "Kare"
                        else:
                            shape = "Dikdortgen"
                    elif len(approx) == 5:
                        shape = "Besgen"
                    elif len(approx) > 6:
                        shape = "Daire"
                        isCirle = isCircle(area, perimeter)
                        let = isCircle
                        let = False
                    else:
                        shape = "Cokgen"
                        let = False

                    if let:
                        mesafeler_pixel.append((i+1, contour, shape, mesafe_pixel, area))

            mesafeler_pixel_arr = sorted(mesafeler_pixel, key = lambda x: x[4], reverse=True)          
            mesafe_pixel_arr = mesafeler_pixel_arr[0][3]
            middle_pixel_x_arr = find_middle_pixel(contour, y=frame_height - mesafe_pixel)
            loc_arr = findAimDetailed(middle_pixel_x_arr)
            write(loc_arr)
            

            
    elif feedback == "modg":
        renk = "green"

    elif feedback == "modd":
        if orgRenk == "red":
            renk = "blue"
        else:
            renk = "red"
        

    #frame_org_prev = org_frame.copy()
    #frame_prev = frame.copy()
    #mesafeler_pixel_prev = mesafeler_pixel.copy()

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()